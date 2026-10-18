//! Shared data types + parsers + small helpers used by both Solidity
//! harness modes ([`super::harness`] handler-invariant; [`super::poc`]
//! deterministic PoC). Anything here must be free of mode-specific
//! agent / prompt logic.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use knowdit_repo_model::{CodeGenRecord, CoverageEntry, HarnessRunRecord, RunKind};
use llmy::client::settings::LLMSettings;
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::types::AuditSpecification;

/// Monotonic process-local counter that uniquifies the per-spec scratch
/// dir inside the agent runtime. Two concurrent regen attempts on the
/// same `spec_id` would otherwise share `harness_dir/spec_{id}/` and
/// race on harness file writes + forge's build cache. `Relaxed` is
/// fine — we only need uniqueness, not ordering between writes.
pub(super) static SPEC_ATTEMPT_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(super) fn next_attempt_id() -> u64 {
    SPEC_ATTEMPT_COUNTER.fetch_add(1, Ordering::Relaxed)
}

// =============================================================================
// Options + outcome types
// =============================================================================

/// Configuration knobs for one fuzz pass. Every value must come from the CLI
/// (no hidden env reads).
#[derive(Debug, Clone)]
pub struct FuzzOptions {
    /// Repository root on disk (`out_git/contracts/<id>/...`).
    pub repo_root: PathBuf,
    /// Where to write `.sol` harness files. Resolved by [`Self::resolve_harness_dir`]
    /// when None — `<repo>/test/knowdit_harness` if the repo is Foundry,
    /// otherwise `<repo>/knowdit_harness`.
    pub harness_dir: Option<PathBuf>,
    /// Path to the `forge` binary. `None` selects docker mode (see
    /// [`Self::docker_image`]); `Some(path)` selects local mode and runs
    /// that binary directly. There is intentionally no `"forge"` default —
    /// callers must opt in to local mode by passing the binary explicitly.
    pub forge_bin: Option<PathBuf>,
    /// Docker image used when [`Self::forge_bin`] is `None`. Defaults to
    /// `lazymio/knowdit:foundry`.
    pub docker_image: String,
    /// Per-invocation memory ceiling, in bytes. `None` disables the cap.
    /// In local mode this is enforced via cgroup v2 (when available);
    /// in docker mode it's plumbed through `--memory` / `--memory-swap`.
    pub forge_mem_cap_bytes: Option<u64>,
    /// Hard timeout for a `forge test` subprocess (seconds).
    pub forge_timeout_secs: u64,
    /// Hard timeout for a `forge coverage` subprocess (seconds). Coverage now
    /// runs after every test attempt, so this bounds what a hopeless or hung
    /// attempt can cost; kept below `forge_timeout_secs` because coverage is
    /// evidence, not the verdict.
    pub forge_coverage_timeout_secs: u64,
    /// Hard timeout for the one-off environment preflight (seconds).
    /// Separate from `forge_timeout_secs`: preflight runs one trivial
    /// test, but on a `via_ir = true` project its fresh compilation
    /// unit pulls forge-std through the IR pipeline every invocation
    /// and no artifact cache shortcuts it.
    pub forge_preflight_timeout_secs: u64,
    /// `--fuzz-runs` for the test invocation.
    pub forge_test_runs: u64,
    /// `--fuzz-runs` for the coverage invocation (usually smaller, since
    /// `forge coverage` is much slower per sample).
    pub forge_coverage_runs: u64,
    /// Hard cap on agent steps across all restarts for one spec.
    pub max_agent_steps: usize,
    /// Per-link cap; 0 = unbounded.
    pub max_specs: usize,
    /// Worker pool size.
    pub concurrency: usize,
    /// When true, clear `code_gen` / `harness_run` / `line_coverage` and
    /// reprocess every spec from scratch.
    pub regenerate: bool,
    /// Optional debug-prefix forwarded to `llmy` for LLM-call dumps.
    pub debug_prefix: Option<String>,
    /// Optional override for the `LLMSettings` (mostly to forward
    /// reasoning_effort, temperature, etc.). None lets `llmy` defaults apply.
    pub llm_settings: Option<LLMSettings>,
    /// When the agent's approximate context size hits this many tokens
    /// the orchestrator drops the conversation and restarts a fresh
    /// agent (with a small "previous attempt" bootstrap). `None` means
    /// "never restart on tokens" — only the step budget terminates.
    pub window_restart_threshold_tokens: Option<usize>,
    /// Maximum number of agent restarts per spec. Once exhausted the
    /// spec is abandoned even if the step budget still has room.
    pub max_restarts: usize,
    /// Maximum number of *content-filter* restarts per spec, budgeted
    /// separately from `max_restarts` because a provider refusal is not the
    /// agent's work going wrong — the request never reached the model. The
    /// cap exists so a spec whose data sits permanently on the wrong side of
    /// the filter still terminates.
    pub max_filter_restarts: usize,
    /// Gate 2 fidelity threshold passed to the inline coverage gate that
    /// runs inside `run_forge`. Same semantic as `agentic reflect`'s
    /// `--gate2-fidelity-threshold`. Defaults to 0.5 in the CLI layer.
    pub gate2_fidelity_threshold: f64,
    /// The project is built with `viaIR = true`. Affects `forge
    /// coverage`: when the runtime supports `--force-via-ir`, the
    /// coverage subcommand opts in (sacrificing some line-mapping
    /// accuracy for compile parity with the project's viaIR build);
    /// when not, coverage runs WITHOUT viaIR and may fail to
    /// compile via-IR-only projects — the agent prompt is told
    /// this is expected and not to keep retrying.
    pub via_ir: bool,
    /// Recursion budget for the `list_test_files` agent tool's directory
    /// scan. Wired from `--harness-list-test-files-max-depth`. 0 disables
    /// the scan entirely.
    pub list_test_files_max_depth: usize,
    /// Fork endpoints handed to harnesses as
    /// [`crate::harness::forge::ForgeRunner::FORK_RPC_ENV`], one per chain;
    /// each is probed at startup to learn which chain it serves. Empty
    /// confines the run to synthetic deployments, and the prompts tell the
    /// agent so.
    pub fork_rpc_urls: Vec<String>,
    /// Block every forked harness is pinned to, overriding whatever the
    /// endpoint reports as its head. `None` uses the probed head, which is the
    /// normal case; set it when the engagement's snapshot is older than the
    /// endpoint's head (the code under audit may already have changed since).
    pub fork_block: Option<u64>,
    /// Skip specs whose call sequence names a contract the forked chain does
    /// not deploy. Off by default: a specification often names an abstract
    /// base or an interface while the deployed instance is a concrete
    /// subclass under a different name (`AbstractOTokenZapper` is deployed as
    /// `OETHZapper`), so a name the deployment tree does not list is a hint
    /// to check the address by hand — not proof the target is unreachable.
    /// The run warns either way; this only decides whether it also skips.
    pub skip_undeployed_specs: bool,
}

impl FuzzOptions {
    /// Resolve [`Self::harness_dir`] given an inspection of the repo root.
    /// Returns the absolute path the agent should write harness files to.
    ///
    /// When `foundry.toml` exists, we honor its `test = '<dir>'` directive so
    /// `forge` actually picks our harness file up — projects like Panoptic
    /// configure `test = 'test/foundry'` and will silently ignore files we
    /// drop into the conventional `test/`. Falls through to `<repo>/test/` if
    /// `foundry.toml` has no explicit `test` directive, and finally to
    /// `<repo>/` if no foundry config exists at all.
    pub fn resolve_harness_dir(&self) -> PathBuf {
        if let Some(dir) = &self.harness_dir {
            return dir.clone();
        }
        let foundry_toml = self.repo_root.join("foundry.toml");
        if foundry_toml.exists() {
            if let Ok(text) = std::fs::read_to_string(&foundry_toml) {
                // Prefer `[profile.default].test` over the legacy
                // top-level `[default].test`. Both writes target the
                // conventional default profile only; if a project
                // pins a non-default profile we'd need a CLI flag.
                if let Ok(cfg) = toml::from_str::<knowdit_project::foundry::FoundryToml>(&text) {
                    let test_subdir = cfg
                        .profile
                        .get("default")
                        .and_then(|p| p.test.clone())
                        .or(cfg.default.test);
                    if let Some(sub) = test_subdir {
                        return self.repo_root.join(sub).join("knowdit_harness");
                    }
                }
            }
            // foundry.toml exists but no explicit `test = ...`; default test/.
            return self.repo_root.join("test").join("knowdit_harness");
        }
        self.repo_root.join("knowdit_harness")
    }
}

/// Top-level outcome: counts + per-link records (in DB, not returned in
/// memory — we only return summary numbers).
#[derive(Debug, Default)]
pub struct FuzzOutcome {
    pub processed_count: usize,
    pub completed_count: usize,
    pub abandoned_count: usize,
    pub error_count: usize,
    pub violated_count: usize,
    pub skipped_resumed: usize,
    /// Specs dropped before any agent ran because their sequence calls a
    /// contract the forked chain does not deploy.
    pub skipped_undeployed: usize,
}

/// What one agent run produced for one link, before persistence. Crossed
/// the mode boundary: both [`super::harness`] and [`super::poc`] return
/// this type so the caller can persist them identically.
#[derive(Debug)]
pub struct LinkFuzzOutcome {
    pub spec_id: i32,
    pub code_gen: CodeGenRecord,
    pub coverage_per_run: Vec<Vec<CoverageEntry>>,
    /// Whether at least one of the `code_gen.runs` had `violated == true`.
    pub any_violation: bool,
}

/// Why the orchestrator stopped iterating for one spec. Mapped to
/// [`knowdit_repo_model::CodeGenStatus`] downstream, but kept separate
/// so we can fold the "no violation but the loop ran cleanly" / "no
/// violation and we ran out of restarts" distinction without forcing a
/// new DB enum value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LoopOutcome {
    Violation,
    StepsExhausted,
    RestartsExhausted,
    AgentError,
}

// =============================================================================
// Forge JSON parsers
// =============================================================================

/// Structured digest of one `forge test --json` invocation. Captures the
/// runtime signals that drive Gate 1 inline (setUp success, fuzzer
/// effective-call count, counter-example contract+function decoding)
/// instead of just `violated`.
///
/// Two flavors of violation:
///
/// * **Invariant counter-example** (handler+invariant mode): forge's
///   fuzzer found an input sequence that breaks the invariant. Surfaces
///   as `counterexample_calls.non_empty()` + `violated=true`.
/// * **Deterministic test failure** (PoC mode): the agent wrote a
///   `function test_…()` whose body deterministically reproduces the
///   attack and asserts with reason starting `KNOWDIT_POST_ATTACK_REACHED:`.
///   Surfaces as `deterministic_test_failure=true` + `violated=true`,
///   without a counterexample sequence.
#[derive(Debug, Clone, Default)]
pub struct ForgeTestSummary {
    pub setup_failed: bool,
    pub failure_reason: Option<String>,
    pub total_calls: i64,
    pub total_reverts: i64,
    pub violated: bool,
    pub counterexample_calls: Vec<CounterCall>,
    pub counterexample_json: Option<serde_json::Value>,
    /// True iff this run was a deterministic `test_…` PoC that
    /// reverted with the `KNOWDIT_POST_ATTACK_REACHED:` prefix. The
    /// reflect Gate 1 (sequence-coverage check) skips runs flagged
    /// deterministic — by construction there's no counterexample
    /// sequence to validate.
    pub deterministic_test_failure: bool,
}

/// One decoded call from a counter-example sequence (`BaseCounterExample`
/// in forge). Field names are forge's `--json` output verbatim.
#[derive(Debug, Clone, Default)]
pub struct CounterCall {
    pub contract: Option<String>,
    pub function: Option<String>,
    pub args: Option<String>,
}

impl ForgeTestSummary {
    pub fn revert_rate(&self) -> f64 {
        if self.total_calls == 0 {
            0.0
        } else {
            self.total_reverts as f64 / self.total_calls as f64
        }
    }

    /// The `(contract, function)` sequence of the counter-example, as forge
    /// decoded it. Two runs of one harness agree only when this matches: the
    /// same violation reached by a different sequence is a different
    /// experiment, and for a finding that matters it is not the same claim.
    pub fn counterexample_signature(&self) -> Vec<(String, String)> {
        self.counterexample_calls
            .iter()
            .map(|call| {
                (
                    call.contract.clone().unwrap_or_else(|| "?".to_string()),
                    call.function.clone().unwrap_or_else(|| "?".to_string()),
                )
            })
            .collect()
    }

    /// The fuzz seed forge reported for this counter-example, when it reported
    /// one. Forge carries it as an extra field on each counter-example call
    /// (`fuzz_seed`), either as a number or as a string; both spellings appear
    /// across versions, so both are accepted.
    ///
    /// Used to re-run the *same* experiment: a replay without the seed would be
    /// a fresh fuzz campaign and could only ever confirm that the harness fuzzes
    /// at all.
    pub fn counterexample_seed(&self) -> Option<i64> {
        let value = self.counterexample_json.as_ref()?;
        Self::find_seed(value)
    }

    fn find_seed(value: &serde_json::Value) -> Option<i64> {
        match value {
            serde_json::Value::Object(map) => {
                if let Some(seed) = map.get("fuzz_seed").and_then(Self::seed_as_i64) {
                    return Some(seed);
                }
                map.values().find_map(Self::find_seed)
            }
            serde_json::Value::Array(items) => items.iter().find_map(Self::find_seed),
            _ => None,
        }
    }

    fn seed_as_i64(value: &serde_json::Value) -> Option<i64> {
        match value {
            serde_json::Value::Number(number) => number
                .as_i64()
                .or_else(|| number.as_u64().and_then(|n| i64::try_from(n).ok())),
            serde_json::Value::String(text) => {
                let text = text.trim();
                match text.strip_prefix("0x") {
                    Some(hex) => i64::from_str_radix(hex, 16).ok(),
                    None => text.parse::<i64>().ok(),
                }
            }
            _ => None,
        }
    }
}

impl std::fmt::Display for ForgeTestSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "calls={} reverts={} revert_rate={:.3} violated={} setup_failed={}",
            self.total_calls,
            self.total_reverts,
            self.revert_rate(),
            self.violated,
            self.setup_failed,
        )?;
        if let Some(reason) = &self.failure_reason {
            write!(f, " failure_reason={reason:?}")?;
        }
        if !self.counterexample_calls.is_empty() {
            let preview = self
                .counterexample_calls
                .iter()
                .take(3)
                .map(|c| {
                    format!(
                        "{}::{}",
                        c.contract.as_deref().unwrap_or("?"),
                        c.function.as_deref().unwrap_or("?"),
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            write!(
                f,
                " counterexample={}[{}{}]",
                self.counterexample_calls.len(),
                preview,
                if self.counterexample_calls.len() > 3 {
                    ",…"
                } else {
                    ""
                }
            )?;
        }
        Ok(())
    }
}

impl ForgeTestSummary {
    /// Parse `forge test --json` stdout into a [`ForgeTestSummary`].
    /// Best-effort: when stdout isn't a JSON suite map (compile errors,
    /// mixed text mode), serde returns an error and we fall back to a
    /// default summary — Gate 1 then sees `setup_failed=false, calls=0`
    /// and passes, leaving the agent to react to the raw exit code.
    pub fn parse(stdout: &str) -> Self {
        use crate::harness::forge_json::SuiteMap;
        serde_json::from_str::<SuiteMap>(stdout.trim())
            .map(Self::from)
            .unwrap_or_default()
    }
}

impl From<crate::harness::forge_json::SuiteMap> for ForgeTestSummary {
    /// Reduce a deserialized `SuiteMap` into the summary. Splitting
    /// this out keeps [`ForgeTestSummary::parse`] a one-liner and lets
    /// callers that already have a `SuiteMap` (e.g. tests, future
    /// callers replaying a captured run) skip the JSON layer.
    fn from(suites: crate::harness::forge_json::SuiteMap) -> Self {
        use crate::harness::forge_json::{CounterExample, TestStatus};

        let mut summary = Self::default();
        for suite in suites.values() {
            for (test_name, tr) in &suite.test_results {
                if let Some(stats) = tr.invariant_stats() {
                    summary.total_calls += stats.calls;
                    summary.total_reverts += stats.reverts;
                }
                if tr.is_setup_failure(test_name) {
                    summary.setup_failed = true;
                }
                if tr.status == TestStatus::Failure
                    && let Some(reason) = &tr.reason
                    && !reason.is_empty()
                    && summary.failure_reason.is_none()
                {
                    summary.failure_reason = Some(reason.clone());
                }
                let counterexample = tr.counterexample.as_ref().or_else(|| {
                    tr.invariant_failures
                        .iter()
                        .find_map(|failure| failure.counterexample.as_ref())
                });
                if let Some(CounterExample::Sequence(_orig, calls)) = counterexample
                    && !calls.is_empty()
                {
                    summary.violated = true;
                    summary.counterexample_json = serde_json::to_value(counterexample).ok();
                    for call in calls {
                        summary.counterexample_calls.push(CounterCall {
                            contract: call.contract_name.clone(),
                            function: call.func_name.clone(),
                            args: call.args.clone(),
                        });
                    }
                }
                // Deterministic PoC violation: a `test_…` function (not
                // invariant) reverted with the magic prefix. The PoC
                // agent is told to use this exact prefix on its
                // post-attack assertion so we can distinguish "intended
                // exploit reproduction" from "random revert".
                if test_name.starts_with("test")
                    && !tr.is_setup_failure(test_name)
                    && tr
                        .reason
                        .as_deref()
                        .is_some_and(is_deterministic_violation_reason)
                {
                    summary.violated = true;
                    summary.deterministic_test_failure = true;
                }
            }
        }
        summary
    }
}

/// String form callers can paste into prompts.
pub(super) const POC_VIOLATION_MARKER: &str = "KNOWDIT_POST_ATTACK_REACHED:";

/// `KNOWDIT_POST_ATTACK_REACHED:` is the contract between the PoC
/// agent's `require(..., "KNOWDIT_POST_ATTACK_REACHED: <reason>")`
/// assertion and the orchestrator's "this is a real violation" gate.
/// Kept as a substring check (not a prefix) so the agent can prepend
/// `vm.expectRevert` framing without breaking detection.
pub(super) fn is_deterministic_violation_reason(reason: &str) -> bool {
    reason.contains(POC_VIOLATION_MARKER)
}

/// Parse a minimal subset of LCOV: every `DA:<line>,<hits>` after a
/// `SF:<source-file-path>` block. Path is taken verbatim from the SF line.
pub(super) fn parse_lcov(text: &str) -> Vec<CoverageEntry> {
    let mut current_file: Option<String> = None;
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim_end();
        if let Some(rest) = line.strip_prefix("SF:") {
            current_file = Some(rest.trim().to_string());
        } else if line == "end_of_record" {
            current_file = None;
        } else if let Some(rest) = line.strip_prefix("DA:") {
            let Some(file) = &current_file else { continue };
            let mut parts = rest.split(',');
            let (Some(line_str), Some(hits_str)) = (parts.next(), parts.next()) else {
                continue;
            };
            let Ok(line_num) = line_str.trim().parse::<i32>() else {
                continue;
            };
            let Ok(hits) = hits_str.trim().parse::<i64>() else {
                continue;
            };
            // lcov's `DA:` lists every *instrumented* line, scoring the ones a
            // run never reached as `,0` — on real harnesses that is ~88% of the
            // records. A zero says only "this particular harness didn't walk
            // here", which no consumer wants: Gate 2 filters `hit_count > 0`
            // itself, and so does every downstream reader. Keeping them cost
            // ~6x the rows in `line_coverage` (623k rows encoding 5.4k distinct
            // positions) and forced any per-file lookup to wade through them,
            // so drop them at the parse boundary and let the table mean exactly
            // "this line was executed".
            if hits <= 0 {
                continue;
            }
            out.push(CoverageEntry {
                relative_contract_path: file.clone(),
                line_number: line_num,
                hit_count: hits,
            });
        }
    }
    out
}

// =============================================================================
// Agent state held inside each mode's runtime
// =============================================================================

/// In-memory accumulator for one agent run. Tools mutate this through an
/// `Arc<Mutex>` handle so the agent's tool-call loop can share state across
/// turns. Both modes use the same shape — harness writes a Solidity file
/// and runs forge; PoC writes a deterministic test and runs forge. The
/// orchestrator drains this into a `CodeGenRecord` when the loop ends.
#[derive(Debug, Default, Clone)]
pub(super) struct HarnessAttempt {
    pub harness_source: String,
    pub harness_filename: Option<String>,
    pub runs: Vec<RecordedRun>,
    pub coverage: Vec<Vec<CoverageEntry>>,
    pub violation_observed: bool,
    pub last_gate_passed: bool,
    pub last_test_calls: u64,
    pub restarts: usize,
}

#[derive(Debug, Clone)]
pub(super) struct RecordedRun {
    pub record: HarnessRunRecord,
    pub coverage_idx: Option<usize>,
}

#[derive(Clone)]
pub(super) struct AttemptHandle(pub(super) Arc<Mutex<HarnessAttempt>>);

impl std::fmt::Debug for AttemptHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AttemptHandle")
    }
}

impl AttemptHandle {
    pub(super) fn new() -> Self {
        Self(Arc::new(Mutex::new(HarnessAttempt::default())))
    }

    pub(super) async fn snapshot(&self) -> HarnessAttempt {
        self.0.lock().await.clone()
    }

    /// Validate `filename` is a basename + write `content` to
    /// `dest_dir/filename`, then stamp the new source + filename onto
    /// the underlying `HarnessAttempt` so the agent loop's restart
    /// bootstrap can reference it. Errors are returned as `Ok(String)`
    /// rather than propagated — the agent sees the failure as a normal
    /// tool result and can re-issue with a corrected filename.
    pub(super) async fn write_agent_file(
        &self,
        dest_dir: &Path,
        filename: &str,
        content: &str,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        if filename.is_empty() || filename.contains('/') || filename.contains('\\') {
            return Ok(
                "error: filename must be a basename (no directory separators) and non-empty"
                    .to_string(),
            );
        }
        if content.trim().is_empty() {
            return Ok("error: content must be non-empty".to_string());
        }
        let path = dest_dir.join(filename);
        if let Some(parent) = path.parent()
            && let Err(e) = tokio::fs::create_dir_all(parent).await
        {
            return Ok(format!(
                "error: failed to create harness dir {}: {e}",
                parent.display()
            ));
        }
        if let Err(e) = tokio::fs::write(&path, content).await {
            return Ok(format!("error: failed to write {}: {e}", path.display()));
        }
        let mut state = self.0.lock().await;
        state.harness_source = content.to_string();
        state.harness_filename = Some(filename.to_string());
        Ok(format!(
            "ok: wrote {} bytes to {}",
            state.harness_source.len(),
            path.display()
        ))
    }
}

// =============================================================================
// Fork guidance — shared by both mode prompts
// =============================================================================

// =============================================================================
// Deployment address book — shared by both mode prompts
// =============================================================================

/// Deployed addresses read out of the repository's own deployment
/// artifacts, keyed by chain id.
///
/// The artifacts are the right source and the wrong thing to hand an
/// agent: a mid-sized protocol keeps a hundred-odd files per network, each
/// tens of kilobytes of ABI and storage layout, to carry one address. This
/// keeps that field and drops the rest, so a forked harness can stand on a
/// deployed instance without spending a spec's whole context finding it.
#[derive(Debug, Default)]
pub(crate) struct DeploymentBook {
    /// chain id -> artifact name -> address.
    by_chain: BTreeMap<u64, BTreeMap<String, String>>,
}

/// The one field worth keeping out of a deployment artifact. Everything
/// else in the file — ABI, bytecode, storage layout — is what makes them
/// too large to hand an agent.
#[derive(Deserialize)]
struct DeploymentArtifact {
    address: Option<String>,
}

/// The part of a Foundry broadcast log this book reads. The rest of the file is
/// transaction and receipt detail: what matters here is which contract the run
/// put on chain and where.
#[derive(Deserialize)]
struct BroadcastLog {
    #[serde(default)]
    transactions: Vec<BroadcastTransaction>,
}

/// One entry of a broadcast log, named as forge writes it. Names and addresses
/// are both optional because a log interleaves contract creations with calls to
/// contracts that already existed.
#[derive(Deserialize)]
struct BroadcastTransaction {
    #[serde(default, rename = "contractName")]
    contract_name: Option<String>,
    #[serde(default, rename = "contractAddress")]
    contract_address: Option<String>,
}

impl DeploymentBook {
    /// Conventional homes for per-network deployment artifacts. Both
    /// spellings are in use across the projects this pipeline audits.
    const ROOTS: [&'static str; 2] = ["deployments", "deploy"];

    /// Where Foundry writes what a deploy script actually did:
    /// `broadcast/<Script>.s.sol/<chainId>/run-latest.json`, with a `dry-run/`
    /// variant when the script was simulated against a fork instead of
    /// broadcast. Read as a second source because a Foundry repository often
    /// carries no `deployments/` tree at all — and a run that cannot resolve a
    /// single address has no way to prove anything about the deployed system.
    const BROADCAST_ROOT: &'static str = "broadcast";

    /// How many artifacts one spec's block may list. A spec names a
    /// handful of contracts; the cap only bounds the pathological case
    /// where a name matches a whole family of artifacts.
    const MAX_ENTRIES: usize = 24;

    /// Read every `<root>/<network>/*.json` under `repo_root`. A network
    /// directory without a parseable `.chainId` is skipped rather than
    /// guessed at: an address attributed to the wrong chain is worse than
    /// an absent one, because a fork of the right chain would silently
    /// call an empty account.
    pub(crate) async fn load(repo_root: &Path) -> Self {
        let mut me = Self::default();
        for root in Self::ROOTS {
            let Ok(mut networks) = tokio::fs::read_dir(repo_root.join(root)).await else {
                continue;
            };
            while let Ok(Some(network)) = networks.next_entry().await {
                if !network.path().is_dir() {
                    continue;
                }
                let Ok(text) = tokio::fs::read_to_string(network.path().join(".chainId")).await
                else {
                    continue;
                };
                let Ok(chain_id) = text.trim().parse::<u64>() else {
                    continue;
                };
                me.load_network(&network.path(), chain_id).await;
            }
        }
        // Second source, filling only what the artifacts above did not answer:
        // an explicit artifact is a deployment record, a broadcast log is a
        // side effect of one.
        me.load_broadcast(repo_root).await;
        tracing::info!(
            networks = me.by_chain.len(),
            addresses = me.by_chain.values().map(BTreeMap::len).sum::<usize>(),
            "DeploymentBook: artifacts loaded"
        );
        me
    }

    /// Read every `broadcast/<Script>.s.sol/<chainId>/run-latest.json` under
    /// `repo_root`, falling back to the `dry-run/` variant when a script was
    /// simulated rather than broadcast.
    ///
    /// The chain comes from the directory name, which is how forge names it;
    /// `.chainId` markers are a Hardhat-deploy convention and do not exist here.
    async fn load_broadcast(&mut self, repo_root: &Path) {
        let broadcast_root = repo_root.join(Self::BROADCAST_ROOT);
        let Ok(mut scripts) = tokio::fs::read_dir(&broadcast_root).await else {
            return;
        };
        while let Ok(Some(script)) = scripts.next_entry().await {
            if !script.path().is_dir() {
                continue;
            }
            let Ok(mut chains) = tokio::fs::read_dir(script.path()).await else {
                continue;
            };
            while let Ok(Some(chain)) = chains.next_entry().await {
                let Ok(chain_id) = chain.file_name().to_string_lossy().parse::<u64>() else {
                    // Forge also writes non-numeric entries here; a chain id
                    // that cannot be read is skipped rather than guessed.
                    continue;
                };
                if !chain.path().is_dir() {
                    continue;
                }
                let latest = chain.path().join("run-latest.json");
                let text = match tokio::fs::read_to_string(&latest).await {
                    Ok(text) => text,
                    Err(_) => {
                        let dry_run = chain.path().join("dry-run").join("run-latest.json");
                        match tokio::fs::read_to_string(&dry_run).await {
                            Ok(text) => text,
                            Err(_) => continue,
                        }
                    }
                };
                let Ok(log) = serde_json::from_str::<BroadcastLog>(&text) else {
                    continue;
                };
                for transaction in log.transactions {
                    let (Some(name), Some(address)) = (
                        transaction.contract_name.filter(|name| !name.trim().is_empty()),
                        transaction
                            .contract_address
                            .filter(|address| address.starts_with("0x")),
                    ) else {
                        continue;
                    };
                    self.by_chain
                        .entry(chain_id)
                        .or_default()
                        .entry(name)
                        .or_insert(address);
                }
            }
        }
    }

    async fn load_network(&mut self, dir: &Path, chain_id: u64) {
        let Ok(mut files) = tokio::fs::read_dir(dir).await else {
            return;
        };
        while let Ok(Some(file)) = files.next_entry().await {
            let path = file.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let Some(name) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let Ok(text) = tokio::fs::read_to_string(&path).await else {
                continue;
            };
            let Ok(artifact) = serde_json::from_str::<DeploymentArtifact>(&text) else {
                continue;
            };
            if let Some(address) = artifact.address.filter(|a| a.starts_with("0x")) {
                self.by_chain
                    .entry(chain_id)
                    .or_default()
                    .insert(name.to_string(), address);
            }
        }
    }

    /// Contract names one specification is about: everything it declares
    /// in `setup` / `pre_attack` / `post_attack` plus every call in
    /// `sequence`. Qualified `Contract.field` keys contribute the contract.
    fn spec_contract_names(spec: &AuditSpecification) -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        for state in [&spec.setup, &spec.pre_attack, &spec.post_attack] {
            names.extend(state.contracts.keys().cloned());
            names.extend(
                state
                    .states_variables
                    .keys()
                    .filter_map(|key| key.split_once('.').map(|(c, _)| c.to_string())),
            );
        }
        names.extend(spec.sequence.iter().map(|step| step.contract.clone()));
        names.sort();
        names.dedup();
        names
    }

    /// A name as it should be compared: deployment artifacts distinguish
    /// a proxy from its implementation with a suffix (`VaultCore` vs
    /// `VaultCoreProxy`), while a specification names whichever it means.
    /// Matching on the stem keeps the pair together without a prefix
    /// match, which would let a short name drag in an unrelated family.
    fn stem(name: &str) -> &str {
        for suffix in ["Proxy", "Implementation", "Impl"] {
            if let Some(stripped) = name.strip_suffix(suffix)
                && !stripped.is_empty()
            {
                return stripped;
            }
        }
        name
    }

    /// The address block for one specification on one chain. `None` when
    /// the book knows nothing about that chain, so the prompt can fall
    /// back to its own instructions instead of showing an empty list.
    pub(super) fn render_for_spec(&self, chain_id: u64, spec: &AuditSpecification) -> Option<String> {
        let entries = self.by_chain.get(&chain_id)?;
        let wanted: Vec<String> = Self::spec_contract_names(spec);
        let matched = Self::matches(entries, &wanted);

        let mut out = format!("### Deployed addresses (chain {chain_id})\n\n");
        if matched.is_empty() {
            out.push_str(&format!(
                "No artifact under this repository's deployment tree matches the contracts this \
                 specification names ({}). They are still on chain — `grep` that chain's \
                 deployment directory for the contract name to read its `\"address\"`.\n",
                wanted.join(", ")
            ));
            return Some(out);
        }
        for (name, address) in matched {
            out.push_str(&format!("- `{name}` → `{address}`\n"));
        }
        out.push_str(
            "\nCall the deployed instance (through its proxy when one is listed). An \
             implementation address is where the code lives, not what the protocol exposes.\n",
        );
        Some(out)
    }

    /// Contracts a specification names that have no artifact on `chain_id`.
    ///
    /// A specification whose calls must go through a contract the chain does
    /// not deploy cannot be reproduced against deployed state — whatever the
    /// harness agent builds for it is a synthetic stand-in, which cannot
    /// support a reportable finding. Callers use this to decide before
    /// spending an agent on it, rather than discovering it after a full
    /// budget of mocks.
    ///
    /// Only `sequence` counts: those are the calls the reproduction must
    /// issue. Contracts named in `setup` / `pre_attack` are frequently
    /// abstract bases or interfaces that no deployment lists under any name.
    pub(crate) fn undeployed_for_spec(
        &self,
        chain_id: u64,
        spec: &AuditSpecification,
    ) -> Vec<String> {
        let Some(entries) = self.by_chain.get(&chain_id) else {
            return Vec::new();
        };
        let called: Vec<String> = spec
            .sequence
            .iter()
            .map(|step| step.contract.clone())
            .collect();
        let matched = Self::matches(entries, &called);
        let resolved: Vec<&str> = matched.iter().map(|(name, _)| *name).collect();
        let resolved_stems: Vec<&str> = resolved.iter().map(|n| Self::stem(n)).collect();

        let mut missing: Vec<String> = called
            .into_iter()
            .filter(|name| !resolved_stems.contains(&Self::stem(name)))
            .collect();
        missing.sort();
        missing.dedup();
        missing
    }

    /// Artifacts whose chain-stem is one of `names`, sorted by name.
    fn matches<'a>(
        entries: &'a BTreeMap<String, String>,
        names: &[String],
    ) -> Vec<(&'a str, &'a str)> {
        let stems: Vec<&str> = names.iter().map(|n| Self::stem(n)).collect();
        entries
            .iter()
            .filter(|(name, _)| stems.contains(&Self::stem(name)))
            .map(|(name, address)| (name.as_str(), address.as_str()))
            .take(Self::MAX_ENTRIES)
            .collect()
    }

    /// Other chains in this book that deploy at least one of `names`.
    /// Distinguishes "wrong chain" from "not deployed here at all", which
    /// need different advice: the first is fixed by pointing the run at that
    /// chain, the second by checking the contract by hand.
    pub(super) fn chains_deploying(&self, chain_id: u64, names: &[String]) -> Vec<u64> {
        self.by_chain
            .iter()
            .filter(|(id, entries)| **id != chain_id && !Self::matches(entries, names).is_empty())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Whether this chain deploys every contract the specification's call
    /// sequence names — the precondition for a proof that exercises deployed
    /// state. A chain the book has never heard of cannot satisfy it: absence
    /// of evidence is not evidence.
    pub(super) fn deploys_all(&self, chain_id: u64, spec: &AuditSpecification) -> bool {
        self.by_chain.contains_key(&chain_id)
            && self.undeployed_for_spec(chain_id, spec).is_empty()
    }


    /// Whether this book read any artifact at all. A repository with no
    /// deployment tree is a different situation from one whose tree lacks the
    /// contract in question, and callers that gate on the deployment answer
    /// (the scope predicate does) must not treat the first as the second.
    pub(crate) fn is_empty(&self) -> bool {
        self.by_chain.is_empty()
    }

    /// Whether this book read artifacts for `chain_id`. A chain the book has
    /// never heard of cannot vouch for anything, in either direction.
    pub(crate) fn knows_chain(&self, chain_id: u64) -> bool {
        self.by_chain.contains_key(&chain_id)
    }

    /// Every `(artifact name, address)` on `chain_id` whose stem is `name`'s.
    /// The read-only chain tools resolve a contract name through this, so an
    /// agent asking about `VaultCore` lands on the same deployed instance the
    /// harness would fork.
    pub(crate) fn addresses_named(&self, chain_id: u64, name: &str) -> Vec<(String, String)> {
        let Some(entries) = self.by_chain.get(&chain_id) else {
            return Vec::new();
        };
        Self::matches(entries, &[name.to_string()])
            .into_iter()
            .map(|(name, address)| (name.to_string(), address.to_string()))
            .collect()
    }
}

/// The fork targets a run resolved to: one endpoint per chain, probed at
/// startup, plus the repository's deployed addresses.
///
/// Per chain rather than one endpoint for the whole run because a protocol
/// spread across chains does not deploy the same contracts everywhere:
/// Origin's cross-chain strategies live on Base while its vaults live on
/// mainnet, and a proof of one against the other's fork can only ever be
/// synthetic.
#[derive(Debug)]
pub(super) struct ForkContext {
    /// chain id -> endpoint, ascending by chain id.
    endpoints: BTreeMap<u64, String>,
    /// chain id -> block the endpoint reported (or `--fork-block`), for the
    /// chains where one was resolved. A chain absent here can still be forked;
    /// its harnesses just cannot be pinned.
    blocks: BTreeMap<u64, u64>,
    pub(super) book: Arc<DeploymentBook>,
}

impl ForkContext {
    pub(super) fn new(
        endpoints: BTreeMap<u64, String>,
        blocks: BTreeMap<u64, u64>,
        book: Arc<DeploymentBook>,
    ) -> Self {
        Self {
            endpoints,
            blocks,
            book,
        }
    }

    /// The chain this specification should fork: the lowest-numbered configured
    /// chain that deploys every contract its call sequence names — and, when the
    /// repository's artifacts do not answer that question, the lowest configured
    /// chain anyway.
    ///
    /// The fallback is the point. Whether the addresses resolve is a fact about
    /// this repository's deployment tree (a Hardhat project keeps them under
    /// `deployments/`, a Foundry one may have only `broadcast/`); whether a fork
    /// is *available* is a fact about the run's configuration. Conflating them
    /// meant a run with a working endpoint but an unfamiliar artifact layout told
    /// its harness agent there was no forked state at all, and every proof the
    /// agent then wrote was synthetic. The agent can find addresses itself — it
    /// has the trees, the existing tests and a grep tool. Callers that must know
    /// whether the deployed instances are *accounted for* ask
    /// [`Self::vouches_for_spec`].
    pub(super) fn chain_for_spec(&self, spec: &AuditSpecification) -> Option<u64> {
        self.vouches_for_spec(spec)
            .or_else(|| self.endpoints.keys().next().copied())
    }

    /// The chain whose deployment tree accounts for every contract the
    /// specification's call sequence names — the precondition for *requiring* a
    /// production-path proof, as opposed to merely offering one.
    pub(super) fn vouches_for_spec(&self, spec: &AuditSpecification) -> Option<u64> {
        self.endpoints
            .keys()
            .copied()
            .find(|chain_id| self.book.deploys_all(*chain_id, spec))
    }

    pub(super) fn endpoint_for(&self, chain_id: u64) -> Option<&str> {
        self.endpoints.get(&chain_id).map(String::as_str)
    }

    /// The block this run pins harnesses to on `chain_id`, when one was
    /// resolved. `None` means the harness cannot be pinned there (the endpoint
    /// reported no head and no override was given), so the run cannot require a
    /// pinned fork on that chain.
    pub(super) fn block_for(&self, chain_id: u64) -> Option<u64> {
        self.blocks.get(&chain_id).copied()
    }


    /// Why a specification has no fork target, for the run log: the first
    /// configured chain, the contracts that chain does not deploy, and the
    /// chains that do. `None` when no endpoint is configured at all, where
    /// there is nothing to explain.
    ///
    /// The three values answer the three different fixes: point the run at
    /// another chain, check the address under another name, or add the
    /// endpoint for a chain that has it.
    pub(super) fn unforkable_advice(
        &self,
        spec: &AuditSpecification,
    ) -> Option<(u64, Vec<String>, Vec<u64>)> {
        let probe_chain = self.endpoints.keys().next().copied()?;
        let missing = self.book.undeployed_for_spec(probe_chain, spec);
        let elsewhere = self.book.chains_deploying(probe_chain, &missing);
        Some((probe_chain, missing, elsewhere))
    }

    /// The guidance for one specification, for the chain it will fork. Built
    /// per spec rather than once per run so both the address list and the
    /// availability statement describe the chain this spec actually gets.
    pub(super) fn guidance_for(&self, spec: &AuditSpecification) -> ForkGuidance {
        let Some(chain_id) = self.chain_for_spec(spec) else {
            return ForkGuidance::new(None, None);
        };
        ForkGuidance::new(self.endpoint_for(chain_id), self.block_for(chain_id))
            .with_deployments(Some(chain_id), &self.book, spec)
    }
}

/// What the run can reach, rendered into both mode prompts.
///
/// The prompts otherwise describe an empty EVM the harness populates
/// itself, which is only true when no fork endpoint is configured. The
/// distinction decides what a proof is worth: state built inside the
/// harness can demonstrate a defect but cannot establish that the deployed
/// protocol reaches it, so a run without a fork can only ever produce
/// investigated candidates.
#[derive(Debug, Clone)]
pub(super) struct ForkGuidance {
    /// Whether `forge` is spawned with [`crate::harness::forge::ForgeRunner::FORK_RPC_ENV`]
    /// pointing at a reachable endpoint.
    available: bool,
    /// The block this run forked at, exposed to the harness as
    /// [`crate::harness::forge::ForgeRunner::FORK_BLOCK_ENV`]. `None` when the
    /// endpoint reported no head and no override was given — the harness then
    /// forks at head and the run cannot require a pinned fork.
    block: Option<u64>,
    /// Pre-rendered address block for this spec on the fork's chain.
    deployments: Option<String>,
}

impl ForkGuidance {
    pub(super) fn new(fork_rpc_url: Option<&str>, block: Option<u64>) -> Self {
        let available = fork_rpc_url.is_some_and(|u| !u.trim().is_empty());
        Self {
            available,
            block: available.then_some(block).flatten(),
            deployments: None,
        }
    }

    /// The fn body a harness should use to stand on deployed state, or `None`
    /// when this run cannot fork at all. One place builds it, so the prompt's
    /// example, the remedy line and the pin requirement cannot disagree about
    /// how a fork is written.
    fn fork_call(&self) -> Option<String> {
        if !self.available {
            return None;
        }
        let env = crate::harness::forge::ForgeRunner::FORK_RPC_ENV;
        Some(match self.block {
            Some(_) => format!(
                "vm.createSelectFork(vm.envString(\"{env}\"), vm.envUint(\"{}\"))",
                crate::harness::forge::ForgeRunner::FORK_BLOCK_ENV
            ),
            None => format!("vm.createSelectFork(vm.envString(\"{env}\"))"),
        })
    }

    /// Attach the deployed addresses this specification needs, resolved
    /// from the repository's own artifacts for the chain the fork
    /// resolved to. No-op without a fork: addresses of deployed contracts
    /// are only meaningful to a harness that can reach them.
    pub(super) fn with_deployments(
        mut self,
        chain_id: Option<u64>,
        book: &DeploymentBook,
        spec: &AuditSpecification,
    ) -> Self {
        if self.available
            && let Some(chain_id) = chain_id
        {
            self.deployments = book.render_for_spec(chain_id, spec);
        }
        self
    }

    /// The setup step both modes share: where the state the reproduction
    /// stands on comes from.
    pub(super) fn setup_rule(&self) -> String {
        if self.available {
            "**Deployed state, not a built world.** `setUp()` forks first (see \
             \"Forked state\" below); resolve every contract the spec names to its deployed \
             address instead of constructing it. Build only what the deployment cannot supply: \
             actors via `vm.deal` / `vm.addr`, and state the `pre_attack` needs that no \
             reachable transaction could have produced."
                .to_string()
        } else {
            "**No on-chain assumptions.** Treat the EVM as completely empty at \
             the start of setUp(). YOU construct every fact the spec.pre_attack \
             requires."
                .to_string()
        }
    }

    /// One sentence for `run_forge` feedback, where there is no room for the
    /// full block but the agent still has turns left to act on it. Names the
    /// same call the spawn path supports, so the agent never has to guess how a
    /// fork is written and lose a turn to a failing one.
    pub(super) fn remedy(&self) -> String {
        match self.fork_call() {
            Some(call) => format!(
                "It reproduces the behaviour in a synthetic deployment only, which is not evidence \
                 about the deployed protocol: fork with `{call}` and resolve the deployed addresses \
                 instead of constructing the project's own contracts."
            ),
            None => "It reproduces the behaviour in a synthetic deployment only, and this run has no \
                     fork endpoint, so the proof is recorded as an investigated candidate rather \
                     than a reportable finding."
                .to_string(),
        }
    }

    /// The prompt block both modes append to their conventions section.
    pub(super) fn render(&self) -> String {
        let env = crate::harness::forge::ForgeRunner::FORK_RPC_ENV;
        if self.available {
            let deployments = self.deployments.as_deref().unwrap_or(
                "### Deployed addresses\n\n\
                 This repository's deployment artifacts did not account for the contracts this \
                 specification names. The fork is live, so resolve them yourself: look in the \
                 project's own `deployments/`, `deploy/` or `broadcast/` trees, in an existing \
                 fork test, or in the project's docs, then cast each address to the contract's \
                 type and call it. Do not fall back to deploying the project's own contract.\n",
            );
            let call = self
                .fork_call()
                .unwrap_or_else(|| format!("vm.createSelectFork(vm.envString(\"{env}\"))"));
            let pin_note = match self.block {
                Some(block) => format!(
                    "- **Fork at this run's block ({}), not at head.** `{}` carries the block every \
                     harness in this run stands on. A proof whose chain state is not fixed cannot \
                     be re-run by anyone else, and happens to depend on state that keeps moving \
                     (oracle rounds, liquidity, whether a proxy has been upgraded since) — so a \
                     violation from a harness that forked at head is not accepted as the \
                     reproduction.\n",
                    block,
                    crate::harness::forge::ForgeRunner::FORK_BLOCK_ENV
                ),
                None => String::new(),
            };
            format!(
                r#"## Forked state is available (use it)

`forge` is spawned with `{env}` pointing at a live chain endpoint, so
`setUp()` can stand on deployed state:

```solidity
{call}
```

That changes what the reproduction must do:

{pin_note}- **Resolve deployed addresses; never redeploy the protocol.** The project's
  own contracts are already on chain. The addresses this specification is
  about are listed below, resolved from the repository's own deployment
  artifacts; when one is missing, find it in the repo's `deployments/`,
  `deploy/`, or `broadcast/` trees, or in an existing fork test
  (`list_test_files`, then `grep` for the contract name). Cast the address to
  the contract's type and call it; do NOT write `new <ProjectContract>(…)`.
- **Do not fabricate state.** `vm.store` / `vm.etch` against a project
  contract's storage or code replaces the very behaviour under audit. Fund
  and impersonate actors with `vm.deal` / `vm.prank` instead.
- **Keep the real dependencies.** Tokens, oracles, and vaults are already
  deployed — use them instead of stand-ins.
- **Verify every address before you call it.** A stale or wrong-chain address
  answers nothing at all — `code.length == 0` — and every call to it reverts or
  silently returns empty, which reads as a broken harness rather than a bad
  address. Check `code.length != 0`, or make one call only the real instance
  answers, before the sequence depends on it.
- **When an address cannot be resolved**, do not fall back to deploying a
  local copy of the project's contract. A reproduction standing on a
  stand-in only shows the defect in a synthetic deployment: it is recorded
  as an investigated candidate, not as a reportable finding.

{deployments}"#
            )
        } else {
            format!(
                r#"## No forked state in this run

`forge` is spawned WITHOUT a fork endpoint ({env} is unset), so the EVM is
empty and `setUp()` must construct everything the reproduction needs. Every
proof this run produces therefore demonstrates the behaviour in a synthetic
deployment: it is recorded as an investigated candidate, not as a reportable
finding, until it is reproduced against deployed code. Runs that need
reportable proofs must pass the fork endpoint option.
"#
            )
        }
    }
}

// =============================================================================
// Small string + filesystem helpers (shared between modes)
// =============================================================================

// =============================================================================
// Project conventions — shared by both mode prompts
// =============================================================================

/// Excerpt of the project's foundry config + remapping conventions,
/// loaded once per [`super::runtime::FuzzRuntime`] and surfaced into
/// every mode's system prompt so the agent picks the right import
/// paths and test-contract layout.
#[derive(Debug, Clone, Default)]
pub(super) struct ProjectConventions {
    foundry_toml_excerpt: String,
    remappings_txt: String,
    sample_test_filenames: Vec<String>,
    sample_test_imports: String,
    /// Project files that already stand on deployed state (they call a fork
    /// cheatcode). Relative to the repo root, capped and sorted.
    ///
    /// Worth surfacing because such a file is the cheapest possible answer to
    /// "where is this contract deployed and how do I fork it here": it was
    /// written by someone who knew, it already compiles against this project's
    /// remappings, and it carries the addresses. On a factory protocol it is
    /// often the *only* place the per-instance addresses exist — a deployments
    /// tree lists implementations, not the fund the engagement is about.
    fork_harness_paths: Vec<String>,
}

impl ProjectConventions {
    pub(super) fn foundry_toml_excerpt(&self) -> &str {
        &self.foundry_toml_excerpt
    }
    pub(super) fn remappings_txt(&self) -> &str {
        &self.remappings_txt
    }
    pub(super) fn sample_test_filenames(&self) -> &[String] {
        &self.sample_test_filenames
    }
    pub(super) fn sample_test_imports(&self) -> &str {
        &self.sample_test_imports
    }

    /// How many forks a scanned file may name before it counts as a fork test.
    /// One is enough: the question is whether the file stands on live state.
    const FORK_MARKERS: [&'static str; 5] = [
        "createSelectFork",
        "createFork",
        "selectFork",
        "rollFork",
        "activeFork",
    ];

    /// Prompt block naming the project's own fork tests, or empty when it has
    /// none. Empty rather than a "none found" line: a note that says nothing was
    /// found reads as an instruction not to fork.
    pub(super) fn fork_harness_note(&self) -> String {
        if self.fork_harness_paths.is_empty() {
            return String::new();
        }
        let mut out = String::from(
            "\nExisting fork tests in this project (start from one of these — it already forks the \
             deployed system, so it carries working addresses and a setUp that compiles):\n",
        );
        for path in &self.fork_harness_paths {
            out.push_str(&format!("- `{path}`\n"));
        }
        out.push_str(
            "Read the closest one with `read_file` before writing `setUp()`, and verify every \
             address you take from it (`code.length != 0`, or a call only the real instance \
             answers) — a stale address silently calls an empty account.\n",
        );
        out
    }

    /// How many files to read while looking for fork tests, and how many hits
    /// to keep. A prompt line, not an index: the first few are enough, and the
    /// bound keeps a large repository from costing a directory walk per spec.
    const FORK_SCAN_MAX_FILES: usize = 200;
    const FORK_SCAN_MAX_HITS: usize = 5;

    /// Find the project's own fork tests, breadth-first over the directories
    /// they conventionally live in.
    async fn scan_fork_harnesses(repo_root: &Path) -> Vec<String> {
        let mut hits: Vec<String> = Vec::new();
        let mut read: usize = 0;
        let mut queue: std::collections::VecDeque<PathBuf> = ["test", "script", "scripts", "deploy"]
            .iter()
            .map(|sub| repo_root.join(sub))
            .collect();
        while let Some(dir) = queue.pop_front() {
            if hits.len() >= Self::FORK_SCAN_MAX_HITS || read >= Self::FORK_SCAN_MAX_FILES {
                break;
            }
            let Ok(mut entries) = tokio::fs::read_dir(&dir).await else {
                continue;
            };
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                if path.is_dir() {
                    // Vendored dependencies and build output are not this
                    // project's tests.
                    if !matches!(name, "node_modules" | "lib" | "out" | "cache" | "artifacts") {
                        queue.push_back(path);
                    }
                    continue;
                }
                if path.extension().and_then(|e| e.to_str()) != Some("sol") {
                    continue;
                }
                read += 1;
                let Ok(text) = tokio::fs::read_to_string(&path).await else {
                    continue;
                };
                let code =
                    crate::reflect::admissibility::PocAdmissibility::strip_comments(&text);
                if !Self::FORK_MARKERS
                    .iter()
                    .any(|marker| code.contains(marker))
                {
                    continue;
                }
                let relative = path
                    .strip_prefix(repo_root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .into_owned();
                hits.push(relative);
            }
        }
        hits.sort();
        hits.dedup();
        hits
    }

    pub(super) async fn load(repo_root: &Path) -> Self {
        let mut me = Self::default();
        if let Ok(text) = tokio::fs::read_to_string(repo_root.join("foundry.toml")).await {
            me.foundry_toml_excerpt = truncate_str(&text, 4_096);
        }
        if let Ok(text) = tokio::fs::read_to_string(repo_root.join("remappings.txt")).await {
            me.remappings_txt = truncate_str(&text, 4_096);
        }
        // Prefer `[profile.default].test`, then legacy top-level
        // `[default].test`, falling back to the conventional `test/`.
        // Files that already fork, scanned where they live: the configured test
        // directory plus the conventional script directories.
        me.fork_harness_paths = Self::scan_fork_harnesses(repo_root).await;
        let test_dir =
            toml::from_str::<knowdit_project::foundry::FoundryToml>(&me.foundry_toml_excerpt)
                .ok()
                .and_then(|cfg| {
                    cfg.profile
                        .get("default")
                        .and_then(|p| p.test.clone())
                        .or(cfg.default.test)
                })
                .map(|td| repo_root.join(td))
                .unwrap_or_else(|| repo_root.join("test"));
        let mut tests: Vec<PathBuf> = Vec::new();
        if let Ok(mut entries) = tokio::fs::read_dir(&test_dir).await {
            while let Ok(Some(e)) = entries.next_entry().await {
                let p = e.path();
                if p.extension().and_then(|s| s.to_str()) == Some("sol") {
                    tests.push(p);
                }
            }
            if let Ok(mut top) = tokio::fs::read_dir(&test_dir).await {
                while let Ok(Some(e)) = top.next_entry().await {
                    let p = e.path();
                    if p.is_dir()
                        && let Ok(mut sub) = tokio::fs::read_dir(&p).await
                    {
                        while let Ok(Some(e2)) = sub.next_entry().await {
                            let p2 = e2.path();
                            if p2.extension().and_then(|s| s.to_str()) == Some("sol") {
                                tests.push(p2);
                            }
                        }
                    }
                }
            }
        }
        tests.sort();
        tests.dedup();
        me.sample_test_filenames = tests
            .iter()
            .take(8)
            .filter_map(|p| p.file_name().and_then(|s| s.to_str()).map(String::from))
            .collect();
        if let Some(first) = tests.first()
            && let Ok(text) = tokio::fs::read_to_string(first).await
        {
            // Tree-sitter–based header extractor: every byte
            // before the first contract / interface / library
            // declaration. Catches block comments, multi-line
            // imports, and weird pragma split-lines that the
            // previous `line.starts_with(...)` scan would
            // mis-classify.
            let header = knowdit_sol::header::extract_header(&text).unwrap_or_default();
            me.sample_test_imports = truncate_str(&header, 2_048);
        }
        me
    }
}

/// `~max`-byte prefix of `s` + "[N bytes truncated]" suffix — stored
/// verbatim in `HarnessRunRecord.stdout/stderr` and surfaced downstream to
/// the reflect grader. The cut is snapped DOWN to the nearest UTF-8 char
/// boundary so slicing forge trace output (which contains multi-byte box
/// characters like `┗`/`━`) never panics.
pub(super) fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{} … [{} bytes truncated]", &s[..end], s.len() - end)
}

/// Last `~n` bytes of `s` with a "…[N bytes earlier]" marker — only used
/// when assembling agent-facing tool feedback. The cut is snapped UP to the
/// nearest UTF-8 char boundary so slicing forge trace output (which contains
/// multi-byte box characters like `┗`/`━`) never panics.
pub(super) fn tail_str(s: &str, n: usize) -> String {
    if s.len() <= n {
        return s.to_string();
    }
    let cut = s.len() - n;
    let mut start = cut;
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    format!("…[{} bytes earlier]\n{}", cut, &s[start..])
}

/// Re-export of the kg::audit_finding map type used by the runtime
/// summary builders. Kept here so the rest of the module doesn't need
/// to re-import.
#[allow(dead_code)]
pub(super) type RunByKind = HashMap<RunKind, Vec<HarnessRunRecord>>;

#[cfg(test)]
mod deployment_book_tests {
    use super::{DeploymentBook, ForkContext};
    use crate::types::{AuditSpecification, SuqenceCallStep};
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    /// A specification whose only contract reference is `contract`.
    fn spec_naming(contract: &str) -> AuditSpecification {
        let mut spec = AuditSpecification::default();
        spec.sequence.push(SuqenceCallStep {
            contract: contract.to_string(),
            function: "entry".to_string(),
            intention: String::new(),
        });
        spec
    }

    /// A throwaway repository with the deployment layout both roots use.
    async fn repo_with_artifacts() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "knowdit-deployment-book-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let write = |rel: &str, body: &str| {
            let path = root.join(rel);
            std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
            std::fs::write(&path, body).expect("write artifact");
        };
        write("deployments/mainnet/.chainId", "1\n");
        write(
            "deployments/mainnet/VaultCore.json",
            r#"{"address":"0x1111111111111111111111111111111111111111","abi":[]}"#,
        );
        write(
            "deployments/mainnet/VaultCoreProxy.json",
            r#"{"address":"0x2222222222222222222222222222222222222222","abi":[]}"#,
        );
        write(
            "deployments/mainnet/Unrelated.json",
            r#"{"address":"0x3333333333333333333333333333333333333333"}"#,
        );
        // No address at all: bytecode-only artifact.
        write("deployments/mainnet/NoAddress.json", r#"{"abi":[]}"#);
        write("deployments/base/.chainId", "8453\n");
        write(
            "deployments/base/VaultCore.json",
            r#"{"address":"0x4444444444444444444444444444444444444444"}"#,
        );
        // Deployed on one chain only: the "wrong chain" case.
        write(
            "deployments/base/BaseOnly.json",
            r#"{"address":"0x6666666666666666666666666666666666666666"}"#,
        );
        // A network directory whose chain id cannot be read is skipped
        // rather than guessed at.
        write("deployments/mystery/VaultCore.json", r#"{"address":"0x55"}"#);
        // Foundry layout: the chain is the directory name, so no `.chainId`
        // marker is needed, and a name only the broadcast knows about
        // (`ScriptOnly`) is the point of reading it at all.
        write(
            "broadcast/Deploy.s.sol/8453/run-latest.json",
            r#"{"chain":8453,"transactions":[
                 {"transactionType":"CREATE","contractName":"ScriptOnly","contractAddress":"0x7777777777777777777777777777777777777777"},
                 {"transactionType":"CREATE","contractName":"VaultCore","contractAddress":"0x9999999999999999999999999999999999999999"},
                 {"transactionType":"CALL","contractName":"ExistingToken","contractAddress":"0x8888888888888888888888888888888888888888"}
               ]}"#,
        );
        // A script that was simulated against a fork rather than broadcast.
        write(
            "broadcast/Simulate.s.sol/1/dry-run/run-latest.json",
            r#"{"transactions":[
                 {"transactionType":"CREATE","contractName":"DryRunOnly","contractAddress":"0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
               ]}"#,
        );
        // Not a chain directory, and not JSON: both are skipped rather than
        // guessed at.
        write("broadcast/Notes.txt", "not a script directory");
        write("broadcast/Broken.s.sol/1/run-latest.json", "{ this is not json");
        root
    }

    #[tokio::test]
    async fn spec_addresses_resolve_per_chain_and_keep_proxy_pairs() {
        let root = repo_with_artifacts().await;
        let book = DeploymentBook::load(&root).await;
        let block = book
            .render_for_spec(1, &spec_naming("VaultCore"))
            .expect("chain 1 is in the book");

        assert!(block.contains("chain 1"), "{block}");
        assert!(block.contains("0x1111111111111111111111111111111111111111"), "{block}");
        assert!(
            block.contains("VaultCoreProxy"),
            "a proxy is the instance a caller reaches: {block}"
        );
        assert!(
            !block.contains("Unrelated"),
            "only the spec's own contracts belong here: {block}"
        );

        let base = book
            .render_for_spec(8453, &spec_naming("VaultCore"))
            .expect("chain 8453 is in the book");
        assert!(base.contains("0x4444444444444444444444444444444444444444"), "{base}");
        assert!(!base.contains("0x1111"), "addresses must not cross chains: {base}");

        assert!(
            book.render_for_spec(999, &spec_naming("VaultCore")).is_none(),
            "an unknown chain has no block to show"
        );
    }

    #[tokio::test]
    async fn unresolved_spec_points_at_the_deployment_tree() {
        let root = repo_with_artifacts().await;
        let book = DeploymentBook::load(&root).await;
        let block = book
            .render_for_spec(1, &spec_naming("SomethingElse"))
            .expect("chain 1 is in the book");
        assert!(block.contains("No artifact"), "{block}");
        assert!(block.contains("SomethingElse"), "{block}");
    }

    #[tokio::test]
    async fn undeployed_specs_are_detected_from_the_call_sequence() {
        let root = repo_with_artifacts().await;
        let book = DeploymentBook::load(&root).await;

        // Calls a deployed contract: nothing missing.
        assert_eq!(
            book.undeployed_for_spec(1, &spec_naming("VaultCore")),
            Vec::<String>::new()
        );

        // Calls something the chain does not deploy: named, so the caller
        // can say why it skipped the spec.
        assert_eq!(
            book.undeployed_for_spec(1, &spec_naming("NotDeployedAnywhere")),
            vec!["NotDeployedAnywhere".to_string()]
        );

        // A chain the book knows nothing about cannot vouch for anything,
        // and must not silently drop the whole run's specs.
        assert_eq!(
            book.undeployed_for_spec(999, &spec_naming("VaultCore")),
            Vec::<String>::new()
        );
    }

    #[tokio::test]
    async fn broadcast_logs_supply_addresses_the_artifact_tree_does_not() {
        let root = repo_with_artifacts().await;
        let book = DeploymentBook::load(&root).await;

        // A name only the broadcast log knows about is resolvable now.
        let block = book
            .render_for_spec(8453, &spec_naming("ScriptOnly"))
            .expect("chain 8453 is in the book");
        assert!(
            block.contains("0x7777777777777777777777777777777777777777"),
            "{block}"
        );

        // A simulated run counts too: those are the addresses the script would
        // create, and the alternative is resolving nothing.
        assert!(
            book.render_for_spec(1, &spec_naming("DryRunOnly"))
                .is_some_and(|block| block
                    .contains("0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")),
            "a dry-run broadcast is still a record of where the contracts go"
        );

        // Which flips the question the fork policy asks: this chain's records
        // now account for a specification that calls the script's contracts.
        assert!(book.deploys_all(8453, &spec_naming("ScriptOnly")));
        assert!(
            !book.deploys_all(1, &spec_naming("ScriptOnly")),
            "a name one chain's log carries does not vouch for another chain"
        );
    }

    #[tokio::test]
    async fn an_explicit_artifact_outranks_a_broadcast_log() {
        let root = repo_with_artifacts().await;
        let book = DeploymentBook::load(&root).await;

        // `VaultCore` appears in both sources on chain 8453. The artifact is
        // the deployment record; the log is a side effect of running a script
        // that may since have been superseded.
        let block = book
            .render_for_spec(8453, &spec_naming("VaultCore"))
            .expect("chain 8453 is in the book");
        assert!(
            block.contains("0x4444444444444444444444444444444444444444"),
            "the artifact address wins: {block}"
        );
        assert!(
            !block.contains("0x9999999999999999999999999999999999999999"),
            "the broadcast address must not replace it: {block}"
        );
    }

    #[tokio::test]
    async fn a_malformed_or_absent_broadcast_tree_is_not_an_error() {
        // No trees at all: the book loads, empty.
        let bare = std::env::temp_dir().join(format!(
            "knowdit-book-bare-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&bare).expect("create scratch dir");
        assert!(DeploymentBook::load(&bare).await.is_empty());
        std::fs::remove_dir_all(&bare).ok();

        // A tree that exists but is partly unreadable contributes what it can —
        // the fixture's `Broken.s.sol` and `Notes.txt` are both in that
        // category, and the good records still load.
        let root = repo_with_artifacts().await;
        let book = DeploymentBook::load(&root).await;
        assert!(
            book.render_for_spec(8453, &spec_naming("ScriptOnly"))
                .is_some(),
            "one unparseable log must not cost the whole tree"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn other_chains_are_named_when_they_deploy_the_missing_contracts() {
        let root = repo_with_artifacts().await;
        let book = DeploymentBook::load(&root).await;

        // Deployed on 8453 but not on 1: the advice is "wrong chain", not
        // "check by hand".
        assert_eq!(
            book.chains_deploying(1, &["BaseOnly".to_string()]),
            vec![8453]
        );
        assert_eq!(book.chains_deploying(8453, &["VaultCore".to_string()]), vec![1]);
        assert!(book.chains_deploying(1, &["Nowhere".to_string()]).is_empty());
    }

    #[tokio::test]
    async fn guidance_carries_addresses_only_with_a_fork() {
        let root = repo_with_artifacts().await;
        let book = std::sync::Arc::new(DeploymentBook::load(&root).await);
        let spec = spec_naming("VaultCore");
        let endpoints = BTreeMap::from([(1u64, "https://rpc.example.invalid".to_string())]);

        let forked =
            ForkContext::new(endpoints, BTreeMap::from([(1u64, 21_000_000u64)]), book.clone())
                .guidance_for(&spec);
        assert!(forked.render().contains("0x1111111111111111111111111111111111111111"));
        assert!(
            forked.render().contains("21000000"),
            "a spec whose chain has a resolved block is pinned to it: {}",
            forked.render()
        );

        let unforked = ForkContext::new(BTreeMap::new(), BTreeMap::new(), book).guidance_for(&spec);
        assert!(
            !unforked.render().contains("0x1111"),
            "addresses of deployed contracts are noise without a fork"
        );
    }

    #[tokio::test]
    async fn an_endpoint_without_an_artifact_tree_still_offers_the_fork() {
        // The failure this guards: a working endpoint plus a repository whose
        // addresses live where the book does not read (a Hardhat tree, a
        // Foundry `broadcast/`) used to render "no forked state in this run",
        // so the agent built a synthetic world while a fork was sitting right
        // there — and the finding could never be reported.
        let endpoints = BTreeMap::from([(1u64, "https://mainnet.example.invalid".to_string())]);
        let book = std::sync::Arc::new(DeploymentBook::default());
        let fork = ForkContext::new(endpoints, BTreeMap::from([(1u64, 21_000_000u64)]), book);

        let spec = spec_naming("VaultCore");
        assert_eq!(fork.chain_for_spec(&spec), Some(1));
        assert_eq!(
            fork.vouches_for_spec(&spec),
            None,
            "an empty book cannot vouch for anything"
        );

        let block = fork.guidance_for(&spec).render();
        assert!(block.contains("Forked state is available"), "{block}");
        assert!(block.contains("21000000"), "the pin survives: {block}");
        assert!(
            block.contains("resolve them yourself"),
            "the agent must be told to find the addresses, not that there is no fork: {block}"
        );
        assert!(!block.contains("No forked state"), "{block}");
    }

    #[tokio::test]
    async fn each_spec_forks_the_lowest_chain_that_deploys_its_contracts() {
        let root = repo_with_artifacts().await;
        let book = std::sync::Arc::new(DeploymentBook::load(&root).await);
        let endpoints = BTreeMap::from([
            (1u64, "https://mainnet.example.invalid".to_string()),
            (8453u64, "https://base.example.invalid".to_string()),
        ]);
        let fork = ForkContext::new(endpoints, BTreeMap::new(), book);

        // Deployed on both: the lowest configured chain wins, so a run's
        // choice does not drift between specs.
        assert_eq!(fork.chain_for_spec(&spec_naming("VaultCore")), Some(1));
        assert_eq!(fork.vouches_for_spec(&spec_naming("VaultCore")), Some(1));
        // Deployed on one chain only: that is the chain it forks, which is
        // the whole point of tracking endpoints per chain.
        assert_eq!(fork.chain_for_spec(&spec_naming("BaseOnly")), Some(8453));
        // Deployed nowhere the run can reach: the fork is still offered (an
        // endpoint exists, so a proof can stand on deployed state and find its
        // own addresses) while nothing vouches that the deployed instances are
        // accounted for — which is what decides whether a synthetic harness may
        // be *refused*.
        assert_eq!(fork.chain_for_spec(&spec_naming("Nowhere")), Some(1));
        assert_eq!(fork.vouches_for_spec(&spec_naming("Nowhere")), None);

        let base_spec = fork.guidance_for(&spec_naming("BaseOnly"));
        assert!(
            base_spec.render().contains("0x6666666666666666666666666666666666666666"),
            "the address block must come from the chain the spec forks"
        );
    }
}

#[cfg(test)]
mod fork_guidance_tests {
    use super::ForkGuidance;
    use crate::harness::forge::ForgeRunner;

    /// Prose assertions must not depend on where the source happened to
    /// wrap, so compare against whitespace-collapsed text.
    fn flat(s: &str) -> String {
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn configured_endpoint_names_the_variable_and_the_forbidden_shortcuts() {
        let fork = ForkGuidance::new(Some("https://rpc.example.invalid"), None);
        let block = flat(&fork.render());
        assert!(block.contains(ForgeRunner::FORK_RPC_ENV), "{block}");
        assert!(block.contains("createSelectFork"), "{block}");
        // The two shortcuts that make a proof non-reportable must be named.
        assert!(block.contains("vm.store"), "{block}");
        assert!(block.contains("new <ProjectContract>"), "{block}");
        assert!(!block.contains("No forked state"), "{block}");
        assert!(flat(&fork.setup_rule()).contains("Deployed state"));
    }

    #[test]
    fn a_resolved_block_is_rendered_into_the_fork_call() {
        let fork = ForkGuidance::new(Some("https://rpc.example.invalid"), Some(21_000_000));
        let block = flat(&fork.render());
        assert!(
            block.contains(ForgeRunner::FORK_BLOCK_ENV),
            "the pin must name the env var that carries it: {block}"
        );
        assert!(block.contains("21000000"), "{block}");
        assert!(
            block.contains("not at head"),
            "the reason a head fork is rejected must be stated: {block}"
        );
        assert!(
            flat(&fork.remedy()).contains(ForgeRunner::FORK_BLOCK_ENV),
            "the remedy must show the pinned form too"
        );
    }

    #[test]
    fn missing_endpoint_says_what_a_proof_is_then_worth() {
        let fork = ForkGuidance::new(None, None);
        let block = flat(&fork.render());
        assert!(block.contains("No forked state"), "{block}");
        assert!(block.contains("synthetic deployment"), "{block}");
        assert!(block.contains("investigated candidate"), "{block}");
        assert!(!block.contains("createSelectFork"), "{block}");
        assert!(flat(&fork.setup_rule()).contains("empty"));
    }

    #[test]
    fn the_resolved_fork_call_names_the_variables_and_the_pin() {
        let pinned = flat(&ForkGuidance::new(Some("https://rpc.example.invalid"), Some(1)).render());
        assert!(pinned.contains("envUint"), "{pinned}");
        assert!(
            pinned.contains("stale or wrong-chain address"),
            "the address-verification requirement belongs in the fork block: {pinned}"
        );
    }

    #[test]
    fn blank_endpoint_is_treated_as_absent() {
        assert!(
            !flat(&ForkGuidance::new(Some("   "), Some(1)).render()).contains("createSelectFork")
        );
    }

    #[test]
    fn remedy_points_at_the_configured_variable() {
        let remedy = ForkGuidance::new(Some("https://rpc.example.invalid"), None).remedy();
        assert!(flat(&remedy).contains(ForgeRunner::FORK_RPC_ENV), "{remedy}");
        let absent = flat(&ForkGuidance::new(None, None).remedy());
        assert!(absent.contains("investigated candidate"), "{absent}");
    }
}

#[cfg(test)]
mod project_conventions_tests {
    use super::ProjectConventions;

    /// A repository with a forking script, a non-forking script, a unit test,
    /// and a file that only *mentions* a fork in prose.
    async fn repo() -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!(
            "knowdit-conventions-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let write = |rel: &str, body: &str| {
            let path = root.join(rel);
            std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
            std::fs::write(&path, body).expect("write");
        };
        write(
            "script/ImpactFork.s.sol",
            "contract F is Test { function run() public { vm.createSelectFork(vm.envString(\"RPC\")); } }",
        );
        write("script/PlainDeploy.s.sol", "contract D is Script { function run() public {} }");
        write("test/Unit.t.sol", "contract U is Test { function test_x() public {} }");
        write(
            "test/Notes.t.sol",
            "// we could fork with vm.createSelectFork(url) here one day\ncontract N {}",
        );
        // Vendored and build directories must not be walked.
        write(
            "lib/dependency/Fork.t.sol",
            "contract L { function f() public { vm.createFork(\"x\"); } }",
        );
        root
    }

    #[tokio::test]
    async fn fork_harnesses_are_listed_and_prose_is_not() {
        let root = repo().await;
        let conventions = ProjectConventions::load(&root).await;
        let note = conventions.fork_harness_note();
        assert!(
            note.contains("script/ImpactFork.s.sol") && !note.contains("lib/dependency"),
            "only the file that actually forks, and not the vendored one: {note}"
        );
        assert!(note.contains("`script/ImpactFork.s.sol`"), "{note}");
        assert!(
            note.contains("verify every address"),
            "reading an address out of an old script needs a check: {note}"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn a_project_without_fork_tests_gets_no_note() {
        let root = std::env::temp_dir().join(format!(
            "knowdit-conventions-plain-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(root.join("script")).expect("mkdir");
        std::fs::write(root.join("script/Deploy.s.sol"), "contract D {}").expect("write");
        let conventions = ProjectConventions::load(&root).await;
        assert!(
            conventions.fork_harness_note().is_empty(),
            "an empty note, not a 'nothing found' line that reads as 'do not fork'"
        );
        std::fs::remove_dir_all(&root).ok();
    }
}

#[cfg(test)]
mod truncate_tests {
    use super::{tail_str, truncate_str};

    /// Build a string whose byte length forces the cut at `cut_byte` to land
    /// inside a 3-byte box character (`┗` = bytes `E2 94 97`), reproducing the
    /// forge-trace panic.
    fn box_line(repeat: usize) -> String {
        "┗".repeat(repeat)
    }

    #[test]
    fn tail_str_snaps_to_char_boundary() {
        // 10 × '┗' = 30 bytes. n=20 → cut at byte 10, which is mid-'┗'
        // (boundaries are multiples of 3). Must not panic, and the result
        // must be valid UTF-8 starting at a real char.
        let s = box_line(10);
        let out = tail_str(&s, 20);
        assert!(out.contains("bytes earlier"));
        assert!(out.ends_with(&box_line(6))); // snapped up: 12..30 = 6 chars
    }

    #[test]
    fn truncate_str_snaps_to_char_boundary() {
        // max=10 lands mid-'┗'; snap down to byte 9 = 3 chars kept.
        let s = box_line(10);
        let out = truncate_str(&s, 10);
        assert!(out.contains("bytes truncated"));
        assert!(out.starts_with(&box_line(3)));
    }

    #[test]
    fn short_strings_pass_through() {
        let s = "┗━┛";
        assert_eq!(tail_str(s, 1_000), s);
        assert_eq!(truncate_str(s, 1_000), s);
    }
}
