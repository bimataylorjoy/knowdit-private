//! LLM-driven invariant generation — the code-grounded, protocol-specific
//! complement to the static [`InvariantMiner`](super::invariants::InvariantMiner).
//!
//! The static miner detects identities by variable-name heuristics
//! (`total*`/`shares`/`supply`), so it only finds the invariants a project
//! happens to name conventionally. A hardened target's novel-bug-bearing
//! invariants are the bespoke ones — earmark-weight conservation, a survival
//! ratio, a packed-struct accounting identity — whose variables match no generic
//! stem, so the miner misses them.
//!
//! This module hands an agent the target's own accounting surface — one contract
//! at a time, its state variables and the bodies of the functions that write
//! them — and asks what identity must always hold and how a violation is
//! demonstrated. The output is validated into ordinary
//! [`MinedInvariant`]s that flow through the *same* downstream as the static
//! ones: [`TargetHypothesisPlanner`](super::target_hypotheses) turns each into a
//! first-class hypothesis scheduled ahead of the knowledge-graph links, and the
//! handler/invariant-fuzz harness proves violations. Nothing downstream knows or
//! cares which source produced an invariant.
//!
//! Two guards keep this honest, mirroring the static miner's contract:
//! * every candidate must cite a `file:line` that exists in the project's facts,
//!   or it is dropped — "nothing here is invented";
//! * an economic invariant must carry a value-delta falsification (attacker gain
//!   + victim metric), or it is dropped — a bare state oddity is not a loss.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use color_eyre::eyre::Result;
use knowdit_repo_model::project_facts::is_vendored_path;
use knowdit_repo_model::{FunctionFacts, ProjectFacts};
use llmy::agent::tool::ToolBox;
use llmy::agent::tools::files::{FindFileTool, GrepDirectoryTool, ListDirectoryTool, ReadFileTool};
use llmy::client::client::LLM;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::reflect::agent_loop::{AttemptHandle, GraderOptions, drive_agent_loop};
use crate::scope::ProgramScope;
use super::index::ProjectIndex;
use super::invariants::{Falsification, InvariantLayer, MinedInvariant};
use super::target_hypotheses::TargetHypothesisPlanner;
use super::tools::{
    LookupCallGraphTool, LookupStateVariableXrefsTool, ReadContractSourceTool,
    ReadFunctionSourceTool,
};

/// How many state-variable declarations / writing-function bodies one contract
/// prompt carries before it is truncated. Bounds the prompt; the agent can read
/// the rest with its tools.
const MAX_STATE_VARS_PER_PROMPT: usize = 40;
const MAX_FUNCTIONS_PER_PROMPT: usize = 24;
const MAX_FUNCTION_BODY_CHARS: usize = 1_600;

/// Tunables for the LLM invariant stage.
#[derive(Debug, Clone)]
pub struct LlmInvariantOptions {
    /// Run the stage at all. The ablation switch: `--gen-specs-invariant-llm false`.
    pub enabled: bool,
    /// Cap on how many contracts get a generation call. Scope-insured contracts
    /// are prioritised, then the ones with the most accounting surface.
    pub max_contracts: usize,
    /// Agent-step budget per contract.
    pub max_agent_steps: usize,
    /// Contracts analysed concurrently. Follows `--gen-specs-concurrency`.
    pub concurrency: usize,
    /// Normalized contract keys of `--gen-specs-focus`. When set, only these
    /// contracts are analysed: a focused run discards every link outside them,
    /// so mining elsewhere only spends tokens (an insured-first ranking would
    /// otherwise pick the core contracts over the focused ones).
    pub focus_contracts: Vec<String>,
    pub debug_prefix: Option<String>,
    pub llm_settings: Option<llmy::client::settings::LLMSettings>,
}

impl Default for LlmInvariantOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            max_contracts: 8,
            max_agent_steps: 20,
            concurrency: 1,
            focus_contracts: Vec::new(),
            debug_prefix: None,
            llm_settings: None,
        }
    }
}

/// The generator. Stateless; all inputs are passed to [`Self::generate`].
pub struct LlmInvariantGenerator;

impl LlmInvariantGenerator {
    /// Generate protocol-specific invariants from the project's own code.
    ///
    /// One agent call per selected contract. Failures on a single contract are
    /// logged and skipped, never fatal: the static miner's output already stands
    /// on its own, and this stage only ever *adds*. Returns the validated,
    /// de-duplicated invariants (deduped against `already` by
    /// [`TargetHypothesisPlanner::id_for`]).
    pub async fn generate(
        facts: &ProjectFacts,
        scope: Option<&ProgramScope>,
        already: &[MinedInvariant],
        project_index: &Arc<ProjectIndex>,
        repo_root: &Path,
        llm: &LLM,
        options: &LlmInvariantOptions,
        dependency_block: &str,
    ) -> Vec<MinedInvariant> {
        let clusters = Self::clusters(facts, scope, options.max_contracts, &options.focus_contracts);
        if clusters.is_empty() {
            return Vec::new();
        }
        let grader_options = GraderOptions {
            max_agent_steps: options.max_agent_steps,
            debug_prefix: options.debug_prefix.clone(),
            llm_settings: options.llm_settings.clone(),
        };
        // Contracts are independent, so they run concurrently, bounded by
        // `options.concurrency`; results are re-ordered by cluster rank before
        // dedup so the output does not depend on which call finishes first.
        let permits = Arc::new(tokio::sync::Semaphore::new(options.concurrency.max(1)));
        let mut tasks = tokio::task::JoinSet::new();
        for (rank, cluster) in clusters.into_iter().enumerate() {
            let permits = permits.clone();
            let project_index = project_index.clone();
            let repo_root = repo_root.to_path_buf();
            let llm = llm.clone();
            let grader_options = grader_options.clone();
            let dependency_block = dependency_block.to_string();
            tasks.spawn(async move {
                let _permit = permits.acquire_owned().await;
                // One fresh retry: a failed attempt is usually the agent ending
                // a turn without a tool call or burning its budget exploring,
                // which a new conversation does not repeat deterministically.
                // Without it a single failure drops a contract outright.
                let mut result = Err(color_eyre::eyre::eyre!("not attempted"));
                for attempt in 0..2 {
                    result = Self::generate_one(
                        &cluster,
                        &project_index,
                        &repo_root,
                        &llm,
                        &grader_options,
                        &dependency_block,
                    )
                    .await;
                    match &result {
                        Ok(_) => break,
                        Err(err) if attempt == 0 => tracing::info!(
                            "invariant-llm: contract {} attempt 1 failed ({err}); retrying once",
                            cluster.contract
                        ),
                        Err(_) => {}
                    }
                }
                (rank, cluster.contract, result)
            });
        }
        let mut finished: Vec<(usize, String, Result<Vec<RawInvariant>>)> = Vec::new();
        while let Some(joined) = tasks.join_next().await {
            match joined {
                Ok(done) => finished.push(done),
                Err(err) => tracing::warn!("invariant-llm: a contract task panicked: {err}"),
            }
        }
        finished.sort_by_key(|(rank, ..)| *rank);

        let mut seen: BTreeSet<i32> = already.iter().map(TargetHypothesisPlanner::id_for).collect();
        let mut out: Vec<MinedInvariant> = Vec::new();
        for (_, contract, result) in finished {
            match result {
                Ok(raws) => {
                    for raw in raws {
                        if let Some(inv) = raw.into_mined(facts, &contract) {
                            let id = TargetHypothesisPlanner::id_for(&inv);
                            if seen.insert(id) {
                                out.push(inv);
                            }
                        }
                    }
                }
                Err(err) => {
                    tracing::warn!("invariant-llm: contract {contract} failed, skipping: {err}");
                }
            }
        }
        tracing::info!(
            "invariant-llm: {} new invariant(s) over {} contract(s)",
            out.len(),
            out.iter()
                .map(|i| i.contract.clone())
                .collect::<BTreeSet<_>>()
                .len()
        );
        out
    }

    /// One agent call for one contract cluster.
    async fn generate_one(
        cluster: &ContractCluster,
        project_index: &Arc<ProjectIndex>,
        repo_root: &Path,
        llm: &LLM,
        options: &GraderOptions,
        dependency_block: &str,
    ) -> Result<Vec<RawInvariant>> {
        let attempt: AttemptHandle<Vec<RawInvariant>> = AttemptHandle::new();
        let mut tools = ToolBox::new();
        tools.add_tool(LookupCallGraphTool {
            project: project_index.clone(),
        });
        tools.add_tool(LookupStateVariableXrefsTool {
            project: project_index.clone(),
        });
        tools.add_tool(ReadContractSourceTool {
            project: project_index.clone(),
        });
        tools.add_tool(ReadFunctionSourceTool {
            project: project_index.clone(),
        });
        tools.add_tool(ReadFileTool::new(repo_root.to_path_buf()));
        tools.add_tool(ListDirectoryTool::new_root(repo_root.to_path_buf()));
        tools.add_tool(FindFileTool::new(repo_root.to_path_buf()));
        tools.add_tool(GrepDirectoryTool::new(repo_root.to_path_buf()));
        tools.add_tool(EmitInvariantsTool {
            attempt: attempt.clone(),
        });
        let cache_suffix = format!("invariant-llm-{}", slugify(&cluster.contract));
        drive_agent_loop(
            project_index,
            llm,
            options,
            // Dependency ground truth first (stable prefix), then the role.
            format!("{dependency_block}{SYSTEM_PROMPT}"),
            cluster.user_prompt(),
            &cache_suffix,
            tools,
            &attempt,
        )
        .await?;
        Ok(attempt.take().await.unwrap_or_default())
    }

    /// Select and build the contract clusters. One per non-vendored contract
    /// that both declares state and has a function writing it, ranked so the
    /// engagement's insured contracts come first, then the widest accounting
    /// surface.
    fn clusters(
        facts: &ProjectFacts,
        scope: Option<&ProgramScope>,
        max_contracts: usize,
        focus_contracts: &[String],
    ) -> Vec<ContractCluster> {
        // Map state-variable id -> name, for rendering read/write sets.
        let var_name: BTreeMap<i32, String> = facts
            .state_variables
            .iter()
            .map(|v| (v.id, v.name.clone()))
            .collect();
        let insured: BTreeSet<String> = scope
            .map(|s| s.assets.iter().map(|a| a.name.clone()).collect())
            .unwrap_or_default();

        let mut clusters: Vec<ContractCluster> = Vec::new();
        for contract in &facts.contracts {
            if is_vendored_path(&contract.path) {
                continue;
            }
            let name = &contract.name;
            if !focus_contracts.is_empty()
                && !focus_contracts.contains(&crate::recall_eval::RecallScorer::contract_key(name))
            {
                continue;
            }
            let vars: Vec<&knowdit_repo_model::StateVarFacts> = facts
                .state_variables
                .iter()
                .filter(|v| v.contract.as_deref() == Some(name.as_str()))
                .collect();
            if vars.is_empty() {
                continue;
            }
            let var_ids: BTreeSet<i32> = vars.iter().map(|v| v.id).collect();
            let writers: Vec<&FunctionFacts> = facts
                .functions
                .iter()
                .filter(|f| f.contract.as_deref() == Some(name.as_str()))
                .filter(|f| f.writes.iter().any(|w| var_ids.contains(w)))
                .collect();
            if writers.is_empty() {
                continue;
            }
            clusters.push(ContractCluster {
                contract: name.clone(),
                insured: insured.contains(name),
                vars: vars
                    .iter()
                    .take(MAX_STATE_VARS_PER_PROMPT)
                    .map(|v| VarLine {
                        name: v.name.clone(),
                        type_name: v.type_name.clone(),
                        anchor: format!("{}:{}", v.path, v.line),
                    })
                    .collect(),
                functions: writers
                    .iter()
                    .take(MAX_FUNCTIONS_PER_PROMPT)
                    .map(|f| FnLine {
                        name: f.name.clone(),
                        anchor: format!("{}:{}", f.path, f.line),
                        writes: f
                            .writes
                            .iter()
                            .filter_map(|w| var_name.get(w).cloned())
                            .collect(),
                        body: truncate(&f.content, MAX_FUNCTION_BODY_CHARS),
                    })
                    .collect(),
                surface: vars.len().saturating_add(writers.len()),
            });
        }
        // Insured first, then widest accounting surface, then name for stability.
        clusters.sort_by(|a, b| {
            b.insured
                .cmp(&a.insured)
                .then(b.surface.cmp(&a.surface))
                .then(a.contract.cmp(&b.contract))
        });
        clusters.truncate(max_contracts);
        clusters
    }
}

/// One contract's accounting surface, ready to render into a prompt.
struct ContractCluster {
    contract: String,
    insured: bool,
    vars: Vec<VarLine>,
    functions: Vec<FnLine>,
    surface: usize,
}

struct VarLine {
    name: String,
    type_name: String,
    anchor: String,
}

struct FnLine {
    name: String,
    anchor: String,
    writes: Vec<String>,
    body: String,
}

impl ContractCluster {
    fn user_prompt(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("# Contract under analysis: {}\n", self.contract));
        if self.insured {
            s.push_str("(this contract is an insured asset in the engagement scope)\n");
        }
        s.push_str("\n## State variables (name : type — declaration)\n");
        for v in &self.vars {
            s.push_str(&format!("- {} : {} — {}\n", v.name, v.type_name, v.anchor));
        }
        s.push_str("\n## Functions that write this contract's state (body excerpts)\n");
        for f in &self.functions {
            s.push_str(&format!(
                "\n### {} — {} (writes: {})\n```solidity\n{}\n```\n",
                f.name,
                f.anchor,
                f.writes.join(", "),
                f.body
            ));
        }
        s.push_str(
            "\nDerive the invariants that must ALWAYS hold over this contract's state, then call \
             emit_invariants exactly once. Use the read tools first if you need a function body or \
             a cross-contract writer you cannot see above.\n",
        );
        s
    }
}

const SYSTEM_PROMPT: &str = r#"You derive protocol-specific INVARIANTS from a smart contract's own code —
identities its state must always satisfy, that a bug would break. You are not
looking for a known bug pattern; you are stating what the code is supposed to
keep true, so a fuzzer can search for a sequence that breaks it.

Good invariants are concrete and checkable from stored state after any sequence
of calls. Examples of the SHAPE (adapt to THIS contract, do not copy):
- conservation: the sum of every account's share of X equals the global X;
- no-double-count: a quantity credited once is not counted again on a second path;
- monotonicity: a price/index only moves in the allowed direction per operation;
- round-trip / differential: convert(x) then invert must return x within rounding;
- solvency: total claims never exceed backing (an ECONOMIC invariant).

Classify each by LAYER and how a violation is FALSIFIED:
- "accounting" / "cross_contract": falsified by STATE EQUALITY — the identity is
  false in stored state after a call sequence.
- "economic": falsified only by VALUE DELTA — you must name the attacker's gain
  metric and the victim's loss metric, both measured in one run. A bare state
  oddity is NOT an economic finding; do not label something economic unless you
  can name both metrics.

Rules:
- Every invariant MUST cite at least one `path:line` from the material you were
  shown (a state-var declaration or a function body). No citation → do not emit it.
- State the identity in THIS contract's own variable and function names.
- Prefer few sharp invariants over many vague ones. Skip anything you cannot tie
  to specific state.
- Call emit_invariants exactly once with all of them (an empty list is allowed if
  nothing crisp holds)."#;

/// The layer, as the LLM names it.
#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum RawLayer {
    Accounting,
    CrossContract,
    Economic,
}

/// How the LLM says the invariant is falsified.
#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum RawFalsification {
    /// Assert the identity fails in stored state after a sequence.
    StateEquality,
    /// The actor's net value rises and a victim metric falls, measured in one run.
    ValueDelta,
}

/// One invariant as the agent emits it, before validation into a [`MinedInvariant`].
#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct RawInvariant {
    /// `accounting` | `cross_contract` | `economic`.
    layer: RawLayer,
    /// A short kind slug describing the shape (`conservation`, `monotonic`,
    /// `no_double_count`, `round_trip`, `solvency`, `fee_bound`, ...).
    kind: String,
    /// The identity, in this contract's own terms.
    statement: String,
    /// `path:line` anchors from the shown material. At least one required.
    #[serde(deserialize_with = "lenient_strings")]
    #[schemars(with = "Vec<String>")]
    evidence: Vec<String>,
    /// `state_equality` (accounting / cross-contract) or `value_delta` (economic).
    falsification: RawFalsification,
    /// For `value_delta`: the attacker's gain metric. Required for economic.
    #[serde(default)]
    attacker_gain: Option<String>,
    /// For `value_delta`: the victim's loss metric. Required for economic.
    #[serde(default)]
    victim_metric: Option<String>,
}

impl RawInvariant {
    /// Validate and convert to a [`MinedInvariant`], or drop it (returns `None`)
    /// when it fails a guard: empty statement, no `file:line` evidence, or an
    /// economic invariant without both value-delta metrics.
    fn into_mined(self, facts: &ProjectFacts, contract: &str) -> Option<MinedInvariant> {
        let statement = self.statement.trim().to_string();
        if statement.is_empty() {
            return None;
        }
        let evidence: Vec<String> = self
            .evidence
            .into_iter()
            .map(|e| e.trim().to_string())
            .filter(|e| evidence_is_real(e, facts))
            .collect();
        if evidence.is_empty() {
            return None;
        }
        let layer = match self.layer {
            RawLayer::Accounting => InvariantLayer::Accounting,
            RawLayer::CrossContract => InvariantLayer::CrossContract,
            RawLayer::Economic => InvariantLayer::Economic,
        };
        let falsification = match self.falsification {
            RawFalsification::StateEquality => Falsification::StateEquality,
            RawFalsification::ValueDelta => {
                let gain = self.attacker_gain.unwrap_or_default().trim().to_string();
                let victim = self.victim_metric.unwrap_or_default().trim().to_string();
                if gain.is_empty() || victim.is_empty() {
                    return None;
                }
                Falsification::ValueDelta {
                    attacker_gain: gain,
                    victim_metric: victim,
                }
            }
        };
        // An economic invariant that isn't a value-delta proves a state oddity,
        // not a loss — the same guard the static miner enforces.
        if layer == InvariantLayer::Economic
            && !matches!(falsification, Falsification::ValueDelta { .. })
        {
            return None;
        }
        Some(MinedInvariant {
            layer,
            kind: kind_slug(&self.kind),
            contract: contract.to_string(),
            statement,
            evidence,
            falsification,
            asset: None,
            value_usd: None,
        })
    }
}

/// Map the LLM's free-text kind onto a fixed `&'static str` slug so a spec, a
/// log line and a test can name the same thing (the `MinedInvariant::kind`
/// contract). Unknown shapes fall back to a generic slug.
fn kind_slug(raw: &str) -> &'static str {
    let r = raw.to_lowercase();
    if r.contains("conserv") || r.contains("sum") {
        "llm-conservation"
    } else if r.contains("double") {
        "llm-no-double-count"
    } else if r.contains("monoton") {
        "llm-monotonic"
    } else if r.contains("round") || r.contains("differential") || r.contains("invert") {
        "llm-round-trip"
    } else if r.contains("solven") || r.contains("backing") {
        "llm-solvency"
    } else if r.contains("dilut") {
        "llm-non-dilution"
    } else if r.contains("fee") {
        "llm-fee-bound"
    } else {
        "llm-invariant"
    }
}

/// Whether an evidence anchor (`path:line` or a bare `path`) names a file the
/// project facts actually contain — the "nothing here is invented" guard.
fn evidence_is_real(anchor: &str, facts: &ProjectFacts) -> bool {
    let path = anchor.split(':').next().unwrap_or(anchor).trim();
    if path.is_empty() {
        return false;
    }
    facts
        .contracts
        .iter()
        .map(|c| c.path.as_str())
        .chain(facts.state_variables.iter().map(|v| v.path.as_str()))
        .chain(facts.functions.iter().map(|f| f.path.as_str()))
        .any(|p| p == path || p.ends_with(path) || path.ends_with(p))
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let taken: String = s.chars().take(max).collect();
    format!("{taken}\n// … (truncated; use read_function_source for the full body)")
}

fn slugify(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect()
}

// ---------------------------------------------------------------------------
// emit_invariants tool
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct EmitInvariantsArgs {
    /// Every invariant derived for this contract. May be empty when nothing
    /// crisp holds. Each must cite at least one `path:line` from the material.
    #[serde(deserialize_with = "lenient_invariants")]
    #[schemars(with = "Vec<RawInvariant>")]
    invariants: Vec<RawInvariant>,
}

/// Models regularly hand a list over as something else: a JSON-encoded string
/// (`"[\"src/A.sol:10\"]"`), a lone string, or a wrapper object
/// (`{"item": [...]}`). Rejecting those shapes made agents burn their whole step
/// budget on retries. Collect every string the value holds instead;
/// the `file:line` guard in [`RawInvariant::into_mined`] still drops anything
/// that is not a real anchor.
pub(crate) fn lenient_strings<'de, D>(deserializer: D) -> std::result::Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    fn collect(value: serde_json::Value, out: &mut Vec<String>) {
        match value {
            serde_json::Value::String(text) => {
                let trimmed = text.trim();
                if (trimmed.starts_with('[') || trimmed.starts_with('{'))
                    && let Ok(inner) = serde_json::from_str::<serde_json::Value>(trimmed)
                {
                    collect(inner, out);
                } else if !trimmed.is_empty() {
                    out.push(trimmed.to_string());
                }
            }
            serde_json::Value::Array(items) => items.into_iter().for_each(|v| collect(v, out)),
            serde_json::Value::Object(map) => map.into_iter().for_each(|(_, v)| collect(v, out)),
            _ => {}
        }
    }
    let value = <serde_json::Value as serde::Deserialize>::deserialize(deserializer)?;
    let mut out = Vec::new();
    collect(value, &mut out);
    Ok(out)
}

/// The invariant list itself, accepted as an array, a JSON-encoded array
/// string, or a single object.
fn lenient_invariants<'de, D>(deserializer: D) -> std::result::Result<Vec<RawInvariant>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let mut value = <serde_json::Value as serde::Deserialize>::deserialize(deserializer)?;
    if let serde_json::Value::String(text) = &value
        && let Ok(inner) = serde_json::from_str::<serde_json::Value>(text.trim())
    {
        value = inner;
    }
    if value.is_object() {
        value = serde_json::Value::Array(vec![value]);
    }
    serde_json::from_value(value).map_err(serde::de::Error::custom)
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = EmitInvariantsArgs,
    invoke = invoke,
    description = "Commit the invariants you derived for this contract and end the run. Call EXACTLY once. Each invariant must state an identity in this contract's own terms, cite at least one path:line you were shown or read, and set layer + falsification (economic REQUIRES attacker_gain and victim_metric). An empty list is allowed if nothing crisp holds.",
    name = "emit_invariants",
)]
struct EmitInvariantsTool {
    attempt: AttemptHandle<Vec<RawInvariant>>,
}

impl EmitInvariantsTool {
    async fn invoke(
        &self,
        args: EmitInvariantsArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let n = args.invariants.len();
        match self.attempt.try_set(args.invariants).await {
            Ok(()) => Ok(format!(
                "ok: {n} invariant(s) committed; the runtime will end the agent now"
            )),
            Err(_) => Ok("error: emit_invariants already called for this run".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use knowdit_repo_model::project_facts::{ContractFacts, StateVarFacts};

    fn facts_with(path: &str) -> ProjectFacts {
        ProjectFacts {
            contracts: vec![ContractFacts {
                id: 1,
                name: "Vault".to_string(),
                path: path.to_string(),
                line: 1,
            }],
            state_variables: vec![StateVarFacts {
                id: 1,
                name: "totalShares".to_string(),
                type_name: "uint256".to_string(),
                path: path.to_string(),
                line: 10,
                contract: Some("Vault".to_string()),
            }],
            functions: vec![],
            inherits: vec![],
        }
    }

    fn raw(layer: RawLayer, fals: RawFalsification, evidence: &[&str]) -> RawInvariant {
        RawInvariant {
            layer,
            kind: "conservation".to_string(),
            statement: "sum of shares equals totalShares".to_string(),
            evidence: evidence.iter().map(|s| s.to_string()).collect(),
            falsification: fals,
            attacker_gain: None,
            victim_metric: None,
        }
    }

    #[test]
    fn accepts_state_equality_with_real_evidence() {
        let facts = facts_with("src/Vault.sol");
        let inv = raw(RawLayer::Accounting, RawFalsification::StateEquality, &["src/Vault.sol:10"])
            .into_mined(&facts, "Vault")
            .expect("valid invariant");
        assert_eq!(inv.kind, "llm-conservation");
        assert_eq!(inv.layer, InvariantLayer::Accounting);
        assert_eq!(inv.contract, "Vault");
    }

    #[test]
    fn drops_invariant_with_no_real_evidence() {
        let facts = facts_with("src/Vault.sol");
        let inv = raw(RawLayer::Accounting, RawFalsification::StateEquality, &["src/Ghost.sol:1"])
            .into_mined(&facts, "Vault");
        assert!(inv.is_none(), "evidence names a file not in the project");
    }

    #[test]
    fn drops_economic_without_value_delta() {
        let facts = facts_with("src/Vault.sol");
        let inv = raw(RawLayer::Economic, RawFalsification::StateEquality, &["src/Vault.sol:10"])
            .into_mined(&facts, "Vault");
        assert!(inv.is_none(), "economic must be value-delta");
    }

    #[test]
    fn economic_with_metrics_is_kept() {
        let facts = facts_with("src/Vault.sol");
        let mut r = raw(RawLayer::Economic, RawFalsification::ValueDelta, &["src/Vault.sol:10"]);
        r.attacker_gain = Some("caller BEAN balance".to_string());
        r.victim_metric = Some("vault backing".to_string());
        let inv = r.into_mined(&facts, "Vault").expect("valid economic invariant");
        assert!(matches!(inv.falsification, Falsification::ValueDelta { .. }));
    }

    #[test]
    fn emit_args_accept_the_shapes_models_actually_send() {
        let canonical = r#"{"invariants":[{"layer":"accounting","kind":"sum","statement":"s","evidence":["src/A.sol:1"],"falsification":"state_equality"}]}"#;
        let stringified_evidence = r#"{"invariants":[{"layer":"accounting","kind":"sum","statement":"s","evidence":"[\"src/A.sol:1\",\"src/A.sol:2\"]","falsification":"state_equality"}]}"#;
        let wrapped_evidence = r#"{"invariants":[{"layer":"accounting","kind":"sum","statement":"s","evidence":{"item":["src/A.sol:1"]},"falsification":"state_equality"}]}"#;
        let stringified_list = r#"{"invariants":"[{\"layer\":\"accounting\",\"kind\":\"sum\",\"statement\":\"s\",\"evidence\":[\"src/A.sol:1\"],\"falsification\":\"state_equality\"}]"}"#;
        for (raw, want) in [(canonical, 1), (stringified_evidence, 2), (wrapped_evidence, 1), (stringified_list, 1)] {
            let args: EmitInvariantsArgs = serde_json::from_str(raw).expect(raw);
            assert_eq!(args.invariants.len(), 1, "{raw}");
            assert_eq!(args.invariants[0].evidence.len(), want, "{raw}");
        }
    }

    #[test]
    fn kind_slug_maps_known_shapes() {
        assert_eq!(kind_slug("no_double_count"), "llm-no-double-count");
        assert_eq!(kind_slug("round_trip differential"), "llm-round-trip");
        assert_eq!(kind_slug("something weird"), "llm-invariant");
    }
}
