//! `agentic gen-specs` subcommand: run the Specification Generator agent.
//!
//! Reads the Knowledge Mapper output (extract↔historical↔finding links) from
//! the project's per-project SQLite database, runs a memory-equipped agent
//! per link to derive zero or more `AuditSpecification`s, and persists the
//! results back to the `specification` table. Optionally dumps a JSON
//! summary for offline review against the project's ground-truth report.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::{Args, ValueEnum};
use color_eyre::eyre::Result;
use knowdit_audit::harness::solidity::SolidityHarness;
use knowdit_audit::scope::{ProgramScope, ScopeGate};
use knowdit_audit::spec::coverage::LinkFocus;
use knowdit_audit::spec::economic_gate::{EconomicGate, EconomicGateOptions};
use knowdit_audit::spec::{
    LinkSource, LinkSpecSummary, SpecGenOptions, SpecGenOutcome, SpecificationGenerator,
};
use knowdit_kg_model::link_strength::LinkStrength;
use knowdit_repo_model::{MatchStrength, RepoDatabase};
use llmy::client::client::LLM;
use serde::Serialize;

use crate::cli::{DatabaseArgs, LoadedRepoDatabase, ProjectArgs};

/// CLI-layer mirror of the 3-tier strength tiers used by both
/// [`MatchStrength`] (mapper) and [`LinkStrength`] (global linker).
/// Those enums live in DB-facing crates that don't (and shouldn't)
/// depend on clap, so this thin wrapper carries the `ValueEnum`
/// derive while converting cleanly to either real enum at the
/// boundary.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum MinStrengthArg {
    High,
    Medium,
    Low,
}

impl From<MinStrengthArg> for MatchStrength {
    fn from(v: MinStrengthArg) -> Self {
        match v {
            MinStrengthArg::High => MatchStrength::High,
            MinStrengthArg::Medium => MatchStrength::Medium,
            MinStrengthArg::Low => MatchStrength::Low,
        }
    }
}

impl From<MinStrengthArg> for LinkStrength {
    fn from(v: MinStrengthArg) -> Self {
        match v {
            MinStrengthArg::High => LinkStrength::High,
            MinStrengthArg::Medium => LinkStrength::Medium,
            MinStrengthArg::Low => LinkStrength::Low,
        }
    }
}

/// Knobs for the gen-specs phase. Long-flag names carry a
/// `--gen-specs-*` prefix to disambiguate from other phases'
/// equivalents (`concurrency`, `regenerate`, `max_agent_steps`,
/// `debug_prefix`).
#[derive(Args, Clone, Debug)]
pub struct GenSpecsSharedArgs {
    /// Maximum agent steps per link before forced abandonment.
    #[arg(long = "gen-specs-max-agent-steps", default_value_t = 60)]
    pub gen_specs_max_agent_steps: usize,

    /// Maximum specifications a single link may commit before finalize.
    #[arg(long = "gen-specs-max-specs-per-link", default_value_t = 4)]
    pub gen_specs_max_specs_per_link: usize,

    /// Cap on number of links processed in this run. `0` means no cap.
    #[arg(long = "gen-specs-max-links", default_value_t = 0)]
    pub gen_specs_max_links: usize,

    /// Cap on findings per *(extract, historical)* pair. `0` means
    /// no cap. When multiple extracts match the same historical,
    /// each pair gets its own quota.
    #[arg(long = "gen-specs-max-findings-per-historical", default_value_t = 0)]
    pub gen_specs_max_findings_per_historical: usize,

    /// Cap on the total number of links processed *per extract*.
    /// `0` means no cap.
    #[arg(long = "gen-specs-max-links-per-extract", default_value_t = 0)]
    pub gen_specs_max_links_per_extract: usize,

    /// Number of links processed in parallel. Defaults to `1`
    /// (strict serial) when omitted. Typed `Option` so streamloop's
    /// `--default-concurrency` can tell "user passed it" from
    /// "default kicked in" — an explicit value wins over `-d`.
    #[arg(long = "gen-specs-concurrency")]
    pub gen_specs_concurrency: Option<usize>,

    /// Regenerate every spec from scratch. Default is to skip any
    /// `(semantic_id, finding_id)` pair that already has rows.
    #[arg(long = "gen-specs-regenerate", default_value_t = false)]
    pub gen_specs_regenerate: bool,

    /// Minimum mapper-emitted match strength to consider. Links with
    /// strength below this tier are dropped before any other capping
    /// is applied. Set to `medium` to also take `Medium` matches, or
    /// `low` to consider every link.
    ///
    /// Defaults to `high` because it is the one knowledge-graph property
    /// measured to predict whether a link yields anything. Over 5,838 links
    /// with ground truth, `High` matches produced a canonical finding 17.2% of
    /// the time against 10.9% for `Medium`. The advantage survives controlling
    /// for queue position (High links are scheduled earlier, so the raw gap is
    /// partly ordering): within the same quartile of a run it is 1.20x / 1.67x
    /// / 1.99x over the first three quarters, reversing only in the last
    /// quarter once the `High` pool is exhausted.
    ///
    /// Supply is not the binding constraint: on tare the `High` tier held 3,617
    /// candidate links while a $400 run gets through ~550, so a budgeted run
    /// stays in the most productive part of the tier and never reaches the
    /// reversal. Raise the budget far enough to drain it and `medium` becomes
    /// the better setting again.
    ///
    /// The other graph-side properties were measured and do NOT predict yield —
    /// link strength above the `Medium` cutoff (0.99x / 1.15x), finding
    /// out-degree (0.90–1.10x), and historical severity, which is mildly
    /// inverted (`Low` 17.9% vs `High` 11.6%: the severest historical bugs are
    /// the most project-specific, hence the least reproducible elsewhere).
    #[arg(long = "gen-specs-min-strength", value_enum, default_value_t = MinStrengthArg::High)]
    pub gen_specs_min_strength: MinStrengthArg,

    /// Ceiling on the resident project source, as a fraction of the model's
    /// input window.
    ///
    /// The block is filled up to this budget, contracts the project profile
    /// calls core first; whatever does not fit stays on demand behind
    /// `read_contract_source` / `read_function_source` and keeps its
    /// signature-index entry. So this is a spending limit rather than a
    /// threshold — a project larger than the budget still gets its core
    /// resident instead of falling back to reading everything. `0` disables
    /// residency entirely.
    ///
    /// It has to be a ceiling because the block is re-billed at the cached
    /// input rate on *every* call the stage makes. At the measured ~6 calls per
    /// link, one resident token costs about 60% of what one uncached token
    /// costs, so a block big enough to outweigh the reads it replaces loses
    /// money even when it fits the window comfortably.
    ///
    /// The default admits ~92k tokens. Residency itself is well supported —
    /// across 21 full-budget audit runs, every run with a resident block
    /// out-produced every run without one, with no overlap. The exact ceiling
    /// is not: the runs that measured it were on a project whose whole source
    /// was 55k tokens, so every setting at or above ~0.06 behaved identically
    /// there. 0.1 is chosen from the cost model — it covers a project of that
    /// size whole, covers the core of a much larger one, and caps the
    /// worst-case carry at roughly half of what the stage currently spends per
    /// link. Projects with a source tree far past the budget are where this
    /// number should be revisited.
    #[arg(long = "gen-specs-resident-source-window-ratio", default_value_t = 0.1)]
    pub gen_specs_resident_source_window_ratio: f64,

    /// Minimum global-linker-emitted link strength on the
    /// `(historical, finding)` edge. Independent axis from
    /// `--gen-specs-min-strength`: this rejects findings the linker
    /// considered only weakly tied to the historical semantic, even
    /// when the mapper found a strong extract↔historical match.
    /// Default `medium` skips `Low` link edges.
    #[arg(long = "gen-specs-min-link-strength", value_enum, default_value_t = MinStrengthArg::Medium)]
    pub gen_specs_min_link_strength: MinStrengthArg,

    /// Mine the target project's own invariants (accounting, cross-contract,
    /// economic) and hand them to the specification agent as a hypothesis
    /// source that does not depend on the knowledge graph.
    ///
    /// This is the ablation switch for measuring what the invariant layer
    /// contributes: `--gen-specs-invariants false` runs the same links with the
    /// historical-pattern source only.
    #[arg(long = "gen-specs-invariants", default_value_t = true, action = clap::ArgAction::Set)]
    pub gen_specs_invariants: bool,

    /// Order the link queue by how uncommon a finding's mechanism is in the
    /// corpus, gated by whether this project can host that mechanism and
    /// whether the engagement pays for an impact it can produce.
    #[arg(long = "gen-specs-rank-by-rarity", default_value_t = true, action = clap::ArgAction::Set)]
    pub gen_specs_rank_by_rarity: bool,

    /// Order the link queue breadth-first before `--gen-specs-max-links` cuts
    /// it: every extract gets a link before any gets a second, extracts
    /// reaching contracts no earlier link touched go first, and links repeating
    /// a historical finding or mechanism wait behind fresh ones. `false` keeps
    /// the priority order, which on a small budget spends it on one extract.
    #[arg(long = "gen-specs-coverage-first", default_value_t = true, action = clap::ArgAction::Set)]
    pub gen_specs_coverage_first: bool,

    /// Run only the links whose extract reaches this location, `Contract` or
    /// `Contract.function`, kept in the scheduler's order and still cut by
    /// `--gen-specs-max-links`. Repeatable. For aiming at code already under
    /// suspicion (an upgrade, a newly listed asset) and for measuring
    /// reasoning recall on the links that aim at a known defect.
    #[arg(long = "gen-specs-focus")]
    pub gen_specs_focus: Vec<LinkFocus>,

    /// Mine the project's own test assertions and show them to the
    /// specification agent (ablation switch for the oracle contribution).
    #[arg(long = "gen-specs-test-oracle", default_value_t = true, action = clap::ArgAction::Set)]
    pub gen_specs_test_oracle: bool,

    /// Debug prefix passed to llmy.
    #[arg(long = "gen-specs-debug-prefix", default_value = "spec")]
    pub gen_specs_debug_prefix: Option<String>,

    /// The engagement's audit-program scope, as a JSON (or TOML) file: which
    /// assets the program insures, which impacts it pays for, whether it pays
    /// for privileged actors, its value floor, its known issues.
    ///
    /// When set — or when the repository carries a `knowdit-scope.json` /
    /// `knowdit-scope.toml` — the pipeline holds every candidate to it. Links
    /// whose project semantic touches none of the insured assets are dropped
    /// before any agent runs, and `commit_specification` refuses a
    /// specification the program's own rules exclude, so an unpaid finding
    /// costs one turn instead of a harness agent, a fuzzer, three graders and
    /// a line in the report. The same file is rendered into the review agent's
    /// prompt, so what a candidate is aimed at and what it is judged against
    /// stay the same rules.
    ///
    /// Unset and absent means the engagement declared no scope: the run keeps
    /// its previous behaviour.
    #[arg(long = "scope-config")]
    pub scope_config: Option<PathBuf>,

    /// Hold every specification's economic claim to deterministic checks at
    /// commit time: a value-flow table with a loser other than the attacker,
    /// the authority the setup state needs (checked against the project's
    /// access guards), a live instance read from the fork or an inevitability
    /// argument, a magnitude above the loss floor, fidelity for any stand-in of
    /// an external protocol, and a verbatim intent quote. Off only to ablate.
    #[arg(long = "gen-specs-economic-gate", default_value_t = true, action = clap::ArgAction::Set)]
    pub gen_specs_economic_gate: bool,

    /// Loss floor in USD for the economic gate: a declared magnitude below it
    /// is refused as informational. 0 turns the floor off.
    #[arg(long = "gen-specs-min-loss-usd", default_value_t = 50.0)]
    pub gen_specs_min_loss_usd: f64,

    /// Require a verbatim quote of the promise a specification says is broken.
    #[arg(long = "gen-specs-require-intent", default_value_t = true, action = clap::ArgAction::Set)]
    pub gen_specs_require_intent: bool,

    /// Schedule one work item per invariant mined from the target's own code
    /// (accounting, cross-contract mirror, economic identities), anchored on
    /// the project semantic that moves the state it binds. These do not depend
    /// on a historical resemblance, which is where a failure mode no corpus
    /// holds can come from. Needs `--gen-specs-invariants`.
    #[arg(long = "gen-specs-target-hypotheses", default_value_t = true, action = clap::ArgAction::Set)]
    pub gen_specs_target_hypotheses: bool,

    /// Most target hypotheses one pass schedules (economic identities first).
    #[arg(long = "gen-specs-max-target-hypotheses", default_value_t = 48)]
    pub gen_specs_max_target_hypotheses: usize,

    /// Schedule the target hypotheses a previous pass already wrote to the repo
    /// DB, skipping the invariant mining and the LLM invariant stage. For
    /// resuming a run without re-deriving (and so replacing) its hypotheses.
    #[arg(long = "gen-specs-reuse-target-hypotheses", default_value_t = false)]
    pub gen_specs_reuse_target_hypotheses: bool,

    /// Judge every specification that passes the deterministic gates with an
    /// LLM reading the program's own rules (exclusions, impacts, assumptions,
    /// known issues), and refuse what the program would not pay for —
    /// economically impractical, bypassable DoS, the victim's own mistake,
    /// behaviour the program calls trusted by design. Needs a scope config.
    /// One LLM call per commit attempt.
    #[arg(long = "gen-specs-scope-judge", default_value_t = false)]
    pub gen_specs_scope_judge: bool,

    /// Also derive protocol-specific invariants with an LLM, from the target's
    /// own accounting code — the code-grounded complement to the name-heuristic
    /// static miner, and the hypothesis source for structurally novel bugs that
    /// resemble no historical finding. Feeds the same target-hypothesis pipeline.
    /// Needs `--gen-specs-target-hypotheses`. Use a schema-following model, NOT one
    /// that over-generates past the provider output-length limit.
    #[arg(long = "gen-specs-invariant-llm", default_value_t = true, action = clap::ArgAction::Set)]
    pub gen_specs_invariant_llm: bool,

    /// Cap on how many contracts the LLM invariant stage analyses (insured
    /// assets first, then the widest accounting surface).
    #[arg(long = "gen-specs-invariant-llm-max-contracts", default_value_t = 8)]
    pub gen_specs_invariant_llm_max_contracts: usize,

    /// Agent-step budget per contract in the LLM invariant stage.
    #[arg(long = "gen-specs-invariant-llm-max-agent-steps", default_value_t = 20)]
    pub gen_specs_invariant_llm_max_agent_steps: usize,

    /// Source of an external contract the project calls but does not contain
    /// (a `.sol` file, or a directory searched recursively). Repeatable. The
    /// source is shown to the specification and invariant agents as ground
    /// truth, so they reason from the dependency's real body instead of guessing
    /// from its interface — e.g. the lending core or accounting engine a vault
    /// integrates, whose semantics an agent otherwise guesses from the interface
    /// alone. Keep it to the few cores the
    /// project integrates; the block is re-billed (cached) on every call.
    #[arg(long = "gen-specs-dependency-source")]
    pub gen_specs_dependency_source: Vec<std::path::PathBuf>,

    /// Refuse an `abandoned` finalize on a target-derived (invariant) link
    /// until the agent lists at least one multi-call sequence it checked
    /// (`compositions_checked`). Per-writer "each writer preserves it"
    /// arguments miss orderings, interleavings, rounding accumulation and
    /// callbacks. Capped at two refusals per link. `false` is the ablation.
    #[arg(long = "gen-specs-require-composition", default_value_t = true, action = clap::ArgAction::Set)]
    pub gen_specs_require_composition: bool,

    /// Run target hypotheses ahead of the mapper's links. `false` appends them
    /// after the mapper queue instead.
    #[arg(long = "gen-specs-target-first", default_value_t = true, action = clap::ArgAction::Set)]
    pub gen_specs_target_first: bool,
}

impl GenSpecsSharedArgs {
    /// Build the [`SpecGenOptions`] for one Specification Generator pass from
    /// these CLI knobs. `link_source` frames the gen-spec agent (mapper
    /// topic-hint vs external reported finding).
    ///
    /// `include_lib_sources` comes from [`crate::cli::ProjectArgs`] and has to
    /// be the same value the project was loaded with — it decides whether the
    /// resident source block treats `lib/` as the project's implementation or
    /// as its dependencies.
    pub(crate) fn build_spec_options(
        &self,
        language_prompt_prefix: String,
        link_source: LinkSource,
        project_profile: Option<knowdit_repo_model::ProjectProfile>,
        include_lib_sources: bool,
        repo_root: &Path,
    ) -> Result<SpecGenOptions> {
        let scope_gate = self.scope_gate(repo_root)?;
        Ok(SpecGenOptions {
            max_agent_steps: self.gen_specs_max_agent_steps,
            max_specs_per_link: self.gen_specs_max_specs_per_link,
            debug_prefix: self.gen_specs_debug_prefix.clone(),
            llm_settings: None,
            max_links: (self.gen_specs_max_links > 0).then_some(self.gen_specs_max_links),
            max_findings_per_historical: (self.gen_specs_max_findings_per_historical > 0)
                .then_some(self.gen_specs_max_findings_per_historical),
            max_links_per_extract: (self.gen_specs_max_links_per_extract > 0)
                .then_some(self.gen_specs_max_links_per_extract),
            concurrency: 1,
            regenerate: self.gen_specs_regenerate,
            min_strength: self.gen_specs_min_strength.into(),
            min_link_strength: self.gen_specs_min_link_strength.into(),
            language_prompt_prefix,
            link_source,
            project_profile,
            resident_source_window_ratio: self.gen_specs_resident_source_window_ratio,
            include_lib_sources,
            source_access: std::sync::Arc::new(std::sync::OnceLock::new()),
            scope_gate: scope_gate.clone(),
            economic_gate: self.gen_specs_economic_gate.then(|| {
                Arc::new(EconomicGate::new(
                    EconomicGateOptions {
                        min_loss_usd: self.gen_specs_min_loss_usd,
                        require_intent: self.gen_specs_require_intent,
                    },
                    scope_gate.as_ref().map(|gate| gate.scope()),
                ))
            }),
            chain: None,
            target_hypotheses: self.gen_specs_target_hypotheses,
            max_target_hypotheses: self.gen_specs_max_target_hypotheses,
            reuse_target_hypotheses: self.gen_specs_reuse_target_hypotheses,
            target_first: self.gen_specs_target_first,
            pattern_library: None,
            invariant_llm: knowdit_audit::spec::LlmInvariantOptions {
                enabled: self.gen_specs_invariant_llm && self.gen_specs_target_hypotheses,
                max_contracts: self.gen_specs_invariant_llm_max_contracts,
                max_agent_steps: self.gen_specs_invariant_llm_max_agent_steps,
                concurrency: self.gen_specs_concurrency.unwrap_or(1).max(1),
                // Filled from the focus list when the options are applied.
                focus_contracts: Vec::new(),
                debug_prefix: None,
                llm_settings: None,
            },
            require_composition: self.gen_specs_require_composition,
            dependency_sources: {
                let deps = knowdit_audit::spec::DependencySources::load(
                    &self.gen_specs_dependency_source,
                )?;
                if !deps.is_empty() {
                    tracing::info!(
                        "dependency sources: {} file(s) shown to the agents as ground truth",
                        deps.files.len()
                    );
                }
                std::sync::Arc::new(deps)
            },
            invariants: self.gen_specs_invariants,
            mined_invariants: std::sync::Arc::new(Vec::new()),
            rank_by_rarity: self.gen_specs_rank_by_rarity,
            coverage_first: self.gen_specs_coverage_first,
            focus: self.gen_specs_focus.clone(),
            test_oracle: self.gen_specs_test_oracle,
            repo_root: Some(repo_root.to_path_buf()),
            mined_test_assertions: std::sync::Arc::new(Vec::new()),
        })
    }

    /// Resolve the engagement's audit-program scope for this run, or `None`
    /// when it declared none.
    ///
    /// An unreadable or malformed scope file is an error rather than a
    /// warning when the caller named it explicitly: running an audit that
    /// silently drops its acceptance criteria is worse than not running it —
    /// the whole point of the file is that candidates are held to it.
    fn scope_gate(&self, repo_root: &Path) -> Result<Option<Arc<ScopeGate>>> {
        let scope = ProgramScope::resolve(repo_root, self.scope_config.as_deref())?;
        Ok(scope.map(|scope| Arc::new(ScopeGate::new(scope, repo_root.to_path_buf()))))
    }
}

#[derive(Args)]
pub struct GenSpecsArgs {
    #[command(flatten)]
    pub project: ProjectArgs,

    #[command(flatten)]
    pub db: DatabaseArgs,

    #[command(flatten)]
    pub shared: GenSpecsSharedArgs,

    /// Optional JSON output path summarizing all link outcomes.
    /// CLI-only; the autoloop manages its own dumps.
    #[arg(long)]
    pub out: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct RunSummary {
    project: String,
    link_count: usize,
    built_link_count: usize,
    abandoned_link_count: usize,
    total_specs: usize,
    by_link: Vec<LinkSpecSummary>,
}

impl GenSpecsArgs {
    /// CLI entry: open the project DB, delegate to
    /// [`Self::gen_specs`], pretty-print + optionally dump `--out`.
    ///
    /// Standalone CLI is Solidity-only — same rationale as
    /// `map-semantics`. Move projects go through `workflow
    /// streamloop`, which passes its harness backend's prefix.
    pub async fn run(self, llm: &LLM) -> Result<()> {
        let LoadedRepoDatabase { spec, repo, .. } = self
            .project
            .to_repo_database(self.db.database_path.clone(), self.db.variant_render_cap)
            .await?;
        let outcome = Self::gen_specs(
            &repo,
            llm,
            &spec.name,
            SolidityHarness::PROMPT_PREFIX.to_string(),
            &self.shared,
            self.project.include_lib_sources,
            &spec.root,
        )
        .await?;
        println!(
            "Specification Generator finished: {} link(s), {} built, {} abandoned, {} spec(s) committed",
            outcome.link_count,
            outcome.built_link_count,
            outcome.abandoned_link_count,
            outcome.total_specs,
        );
        if let Some(out_path) = self.out.as_ref() {
            if let Some(parent) = out_path.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)?;
            }
            let summary = RunSummary {
                project: spec.name.clone(),
                link_count: outcome.link_count,
                built_link_count: outcome.built_link_count,
                abandoned_link_count: outcome.abandoned_link_count,
                total_specs: outcome.total_specs,
                by_link: outcome.by_link.iter().map(LinkSpecSummary::from).collect(),
            };
            std::fs::write(out_path, serde_json::to_string_pretty(&summary)?)?;
            tracing::info!("Specification summary written to {}", out_path.display());
        }
        Ok(())
    }

    /// In-process entry: run the Specification Generator against an
    /// already-opened repo DB + LLM. No `&self` — every knob comes
    /// from `shared`.
    ///
    /// `language_prompt_prefix` is the language-context block to
    /// prepend to the spec-gen agent's system prompt. Sourced from
    /// `harness_backend.prompt_prefix()` by streamloop, or
    /// hardcoded to `SolidityHarness::PROMPT_PREFIX` by the
    /// standalone Solidity-only CLI.
    pub async fn gen_specs(
        repo: &RepoDatabase,
        llm: &LLM,
        project_name: &str,
        language_prompt_prefix: String,
        shared: &GenSpecsSharedArgs,
        include_lib_sources: bool,
        repo_root: &Path,
    ) -> Result<SpecGenOutcome> {
        // The profile is both a precondition (the upstream pipeline must have
        // produced one) and an input: it goes into every link's system prompt.
        let profile = repo.get_project_profile().await?.ok_or_else(|| {
            color_eyre::eyre::eyre!(
                "Specification Generator requires a ProjectProfile; run `knowdit agentic \
                 profile` for project '{project_name}' first (or let `workflow streamloop` \
                 run the profile phase)."
            )
        })?;
        let mut options = shared.build_spec_options(
            language_prompt_prefix,
            LinkSource::Mapper,
            Some(profile),
            include_lib_sources,
            repo_root,
        )?;
        // The only knob this standalone path reads that the shared builder
        // does not: streamloop drives concurrency from its own scheduler.
        options.concurrency = shared.gen_specs_concurrency.unwrap_or(1).max(1);
        // Prime the static invariants first, then append the LLM-generated
        // ones, so both reach `run` (whose internal `primed()` then no-ops).
        // streamloop wires the same two steps in its pipeline runner.
        let repo_root = options.repo_root.clone();
        let options = options
            .primed(repo)
            .await
            .with_llm_invariants(repo, repo_root.as_deref(), llm)
            .await;
        SpecificationGenerator::new().run(repo, llm, &options).await
    }
}
