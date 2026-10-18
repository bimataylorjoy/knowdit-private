//! Foundry / fuzz-agent runtime knobs shared by `agentic fuzz`,
//! `agentic regen`, and `workflow autoloop` (which drives both).
//! The same `SolidityFuzzGenerator` underpins all three, so the
//! forge-backend config (binary / docker image / memory cap), the
//! per-spec LLM-agent budget, and the Gate 2 inline-fidelity
//! threshold are a single struct flattened wherever needed.
//!
//! Long-flag names carry a phase-disambiguating prefix
//! (`--harness-*` / `--forge-*` / `--gate2-*`) so flattening
//! [`HarnessSharedArgs`] into the autoloop alongside other phase
//! `SharedArgs` doesn't trip clap's duplicate-arg check. The CLI
//! standalone commands inherit the same names — there is no
//! pre-rename compatibility layer.

use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Args;
use color_eyre::eyre::Result;
use knowdit_audit::harness::forge::{ForgeBackend, ForgeRunner, ForgeTimeouts};
use knowdit_audit::harness::solidity::{FuzzOptions, HarnessKind, SolidityHarness};

/// CLI mirror of [`HarnessKind`] (kebab-cased on the command line).
/// Two-variant `clap::ValueEnum` that we map to the audit-crate enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum HarnessModeCli {
    /// Handler+invariant: Foundry handler contract + invariant test;
    /// fuzzer drives input sequences (existing default).
    Handler,
    /// Deterministic PoC: agent writes a `test_…()` that reproduces
    /// the attack deterministically and asserts post-attack with
    /// `KNOWDIT_POST_ATTACK_REACHED:`.
    Poc,
    /// Per specification, from its declared oracle: a path that must be
    /// searched (a `fuzz` oracle, differential, bounds) runs the handler +
    /// invariant fuzzer; a known path runs the deterministic PoC.
    Auto,
}

impl From<HarnessModeCli> for HarnessKind {
    fn from(v: HarnessModeCli) -> Self {
        match v {
            HarnessModeCli::Handler => Self::Handler,
            HarnessModeCli::Poc => Self::Poc,
            HarnessModeCli::Auto => Self::Auto,
        }
    }
}

/// Forge backend + harness-codegen-agent knobs. Flattened by
/// `agentic fuzz`, `agentic regen`, and `workflow autoloop`. All
/// fields use phase-disambiguated long names so multiple consumers
/// can co-exist in one `clap::Args` parse tree.
#[derive(Args, Clone, Debug)]
pub struct HarnessSharedArgs {
    /// Debug-prefix forwarded to llmy for LLM-call dumps.
    #[arg(long = "harness-debug-prefix", default_value = "harness")]
    pub harness_debug_prefix: Option<String>,

    /// Path to a local `forge` binary. When set, knowdit runs forge
    /// directly on the host (optionally wrapped in a cgroup v2 memory
    /// cap — see `--forge-mem-cap-gb`). When unset (default), knowdit
    /// uses `docker run --rm <docker-image> forge ...` so a
    /// memory-runaway forge can't OOM the host.
    #[arg(short = 'b', long = "forge-bin")]
    pub forge_bin: Option<PathBuf>,

    /// Docker image used when `--forge-bin` is not provided. The image
    /// must have `forge` installed at `/usr/local/bin/forge` and must
    /// run as a uid that matches the host (the default
    /// `lazymio/knowdit:foundry` runs uid 1000).
    #[arg(long = "forge-docker-image", default_value = "lazymio/knowdit:foundry")]
    pub docker_image: String,

    /// Per-invocation memory ceiling, in GiB. In docker mode this is
    /// plumbed through `--memory` / `--memory-swap`. In local mode
    /// this is enforced via cgroup v2 (when a writable delegated
    /// parent is available); if no usable cgroup is found we log a
    /// warning and fall back to direct exec. Set to 0 to disable the
    /// cap entirely.
    #[arg(long = "forge-mem-cap-gb", default_value_t = 12)]
    pub forge_mem_cap_gb: u64,

    /// Where to write `.sol` harness files. If omitted: auto-detect —
    /// if the repo has `foundry.toml` + `test/`, write to
    /// `<repo>/test/knowdit_harness/`, otherwise `<repo>/knowdit_harness/`.
    #[arg(long = "harness-dir")]
    pub harness_dir: Option<PathBuf>,

    /// Hard timeout per `forge test` subprocess (seconds).
    #[arg(long = "forge-timeout-secs", default_value_t = 600)]
    pub forge_timeout_secs: u64,

    /// Hard timeout per `forge coverage` subprocess (seconds). Coverage runs
    /// after every test attempt (it is the reflect grader's evidence and the
    /// link scheduler's breadth signal), so this bounds what a hung or
    /// hopeless attempt costs. Kept below `--forge-timeout-secs`.
    #[arg(long = "forge-coverage-timeout-secs", default_value_t = 300)]
    pub forge_coverage_timeout_secs: u64,

    /// Hard timeout for the one-off preflight `forge test` (seconds).
    /// Independent of `--forge-timeout-secs` because preflight runs a
    /// single trivial test — but the compile it triggers is not always
    /// trivial: on a `via_ir = true` project the preflight file is a
    /// fresh compilation unit that drags forge-std through the IR
    /// pipeline every time, which no artifact cache can shortcut
    /// (measured: ~61s per invocation on solc 0.8.13). Raise this for
    /// such projects instead of losing them to a false "environment is
    /// broken" verdict.
    #[arg(long = "forge-preflight-timeout-secs", default_value_t = 60)]
    pub forge_preflight_timeout_secs: u64,

    /// `--fuzz-runs` for the `forge test` invocation.
    #[arg(long = "forge-test-runs", default_value_t = 100_000)]
    pub forge_test_runs: u64,

    /// `--fuzz-runs` for the `forge coverage` invocation. Coverage is
    /// much slower per sample, so this is usually a smaller value.
    #[arg(long = "forge-coverage-runs", default_value_t = 5_000)]
    pub forge_coverage_runs: u64,

    /// Maximum agent steps per spec across all restarts. Cumulative
    /// budget — restarts reuse the same step counter so a model that
    /// thrashes can't reset its budget by re-opening.
    #[arg(long = "harness-max-agent-steps", default_value_t = 120)]
    pub harness_max_agent_steps: usize,

    /// Approximate token count at which the orchestrator restarts the
    /// agent. Defaults to None (no restart; rely on step budget). Set
    /// this to ~80% of the model's input window to avoid hitting
    /// hard context limits during long iterations.
    #[arg(long = "harness-window-restart-threshold-tokens")]
    pub harness_window_restart_threshold_tokens: Option<usize>,

    /// Maximum number of agent restarts per spec before the spec is
    /// abandoned. Each restart drops conversation history but keeps
    /// the harness file on disk plus a short "previous attempt" note
    /// in the new system prompt so the new agent doesn't restart from
    /// scratch.
    #[arg(long = "harness-max-restarts", default_value_t = 3)]
    pub harness_max_restarts: usize,

    /// Maximum number of *content-filter* restarts per spec, budgeted
    /// separately from `--harness-max-restarts`: a provider refusal never
    /// reached the model, so it shouldn't spend the agent's ordinary
    /// restart allowance. The fresh conversation re-reads the on-disk
    /// harness file, dropping the accumulated narration the filter
    /// objected to.
    #[arg(long = "harness-max-filter-restarts", default_value_t = 3)]
    pub harness_max_filter_restarts: usize,

    /// Gate 2 fidelity threshold for the inline coverage gate. The
    /// agent gets `[GATE 2 FAILED — coverage]` in its `run_forge`
    /// tool result when `hit_fns / expected_fns < this`. Same default
    /// as `agentic reflect`.
    #[arg(long = "gate2-fidelity-threshold", default_value_t = 0.5)]
    pub gate2_fidelity_threshold: f64,

    /// Tell the harness pipeline that the project's solc build uses
    /// viaIR. Plumbed into the per-spec prompt (so the agent knows
    /// coverage might fail) and into `forge coverage --force-via-ir`
    /// when the resolved forge backend supports the flag. The
    /// autoloop orchestrator overrides this from its top-level
    /// `--via-ir` so the user doesn't have to pass it twice.
    #[arg(long = "harness-via-ir", default_value_t = false)]
    pub harness_via_ir: bool,

    /// Pick which test-authoring mode the harness agent runs.
    /// `handler` (default) is the Foundry handler + invariant pipeline;
    /// `poc` writes a deterministic `test_…()` that asserts with
    /// `KNOWDIT_POST_ATTACK_REACHED:`; `auto` picks per specification from
    /// its declared oracle — a path that must be searched gets the handler
    /// fuzzer, a known path gets the PoC.
    #[arg(short = 'm', long = "harness-mode", value_enum, default_value_t = HarnessModeCli::Handler)]
    pub harness_mode: HarnessModeCli,

    /// Recursion budget for the `list_test_files` agent tool when it
    /// scans the project's `test/` / `script/` / etc. directories.
    /// 0 disables the scan entirely.
    #[arg(long = "harness-list-test-files-max-depth", default_value_t = 6)]
    pub harness_list_test_files_max_depth: usize,

    /// JSON-RPC endpoint a harness may fork for deployed state
    /// (e.g. `https://eth-mainnet.g.alchemy.com/v2/<key>`). Every spawned
    /// `forge` receives it as `KNOWDIT_FORK_RPC_URL`, and both mode prompts
    /// are told to fork with it and resolve deployed addresses instead of
    /// constructing the project's contracts.
    ///
    /// Repeatable: each endpoint is probed at startup to learn which chain it
    /// serves, and a specification forks the lowest-numbered configured chain
    /// that deploys every contract its call sequence names. Give it every
    /// chain your specs touch — a protocol spread over several chains does
    /// not deploy the same contracts everywhere, and a spec forked on the
    /// wrong chain can only produce a synthetic proof.
    ///
    /// Unset (the default) confines the run to synthetic deployments: a
    /// harness can still demonstrate a defect, but nothing about the deployed
    /// protocol, so its findings are recorded as investigated candidates and
    /// never as reportable. Not usable with the docker forge backend, which
    /// runs with `--network none`.
    #[arg(long = "fork-rpc-url")]
    pub fork_rpc_url: Vec<String>,

    /// Block to pin every forked harness to, overriding the block the endpoint
    /// reports as its head.
    ///
    /// Unset (the normal case) uses the endpoint's head, resolved once at
    /// startup and given to every harness in the run, so the whole run stands
    /// on one chain state and a proof can be re-run by whoever triages it. Set
    /// it when the deployment you are auditing is not on head: an engagement
    /// snapshotted at an earlier block, a proxy upgraded since, or a state the
    /// public endpoint no longer serves at head.
    ///
    /// A run that resolved a block requires harnesses to fork at it — a
    /// violation found on a fork at head is not accepted as the reproduction,
    /// because the same harness can violate today and pass tomorrow without
    /// anything about the code having changed.
    #[arg(long = "fork-block")]
    pub fork_block: Option<u64>,

    /// Skip specs whose call sequence names a contract the forked chain
    /// does not deploy. Off by default, and the run warns about them either
    /// way. A specification often names an abstract base while the deployed
    /// instance is a concrete subclass under another name, so absence from
    /// the deployment tree does not prove the target is unreachable — this
    /// only decides whether to spend a harness agent on the doubt.
    #[arg(long = "fuzz-skip-undeployed-specs")]
    pub fuzz_skip_undeployed_specs: bool,

    /// Offer the specification agent read-only tools over the fork endpoints
    /// (`chain_call`, `chain_storage`, `chain_logs`, `chain_account`,
    /// `resolve_address`) and let the economic gate verify live instances
    /// against them. Needs `--fork-rpc-url`; reads are pinned to the same block
    /// the harnesses fork at.
    #[arg(long = "chain-tools", default_value_t = true, action = clap::ArgAction::Set)]
    pub chain_tools: bool,

    /// The `cast` binary for the chain tools. Default: next to `--forge-bin`,
    /// then `PATH`.
    #[arg(long = "chain-cast-bin")]
    pub chain_cast_bin: Option<PathBuf>,

    /// Timeout for one chain read, in seconds.
    #[arg(long = "chain-read-timeout-secs", default_value_t = 30)]
    pub chain_read_timeout_secs: u64,

    /// Widest block range one `chain_logs` query may span.
    #[arg(long = "chain-max-log-span", default_value_t = 50_000)]
    pub chain_max_log_span: u64,
}

/// Per-subcommand fields [`HarnessSharedArgs`] doesn't know about
/// (project root, fuzz-only caps, project viaIR state). The caller
/// fills this in;
/// [`HarnessSharedArgs::to_fuzz_options`] consumes it.
#[derive(Debug, Clone)]
pub struct FuzzOptionsBuild {
    pub repo_root: PathBuf,
    pub max_specs: usize,
    pub concurrency: usize,
    pub regenerate: bool,
    /// Whether the project's solc settings use viaIR. Plumbs into
    /// `forge coverage --force-via-ir` when the backend supports it
    /// and into the per-spec prompt note when it doesn't. Standalone
    /// `agentic fuzz` / `agentic regen` get this from the project
    /// foundry profile via `--harness-via-ir`; the autoloop
    /// orchestrator overrides it from `--via-ir` on the foundry
    /// flatten group.
    pub via_ir: bool,
}

impl HarnessSharedArgs {
    /// Connect the read-only chain access the specification agent and the
    /// economic gate use, or `None` when the run has no fork endpoint or the
    /// tools are turned off.
    ///
    /// When the run did not pin a block and there is exactly one endpoint, the
    /// block this resolves is written back into `fork_block`, so the harnesses
    /// built afterwards fork at the state the agent read — otherwise a
    /// precondition checked at one head could be proved at a later one.
    pub async fn connect_chain(
        &mut self,
        repo_root: &Path,
        scope: Option<&knowdit_audit::scope::ProgramScope>,
    ) -> Result<Option<knowdit_audit::chain::ChainReader>> {
        if !self.chain_tools || self.fork_rpc_url.is_empty() {
            return Ok(None);
        }
        let options = knowdit_audit::chain::ChainReaderOptions {
            cast_bin: self.chain_cast_bin.clone(),
            forge_bin: self.forge_bin.clone(),
            rpc_urls: self.fork_rpc_url.clone(),
            fork_block: self.fork_block,
            timeout: Duration::from_secs(self.chain_read_timeout_secs.max(1)),
            max_log_span: self.chain_max_log_span.max(1),
            ..Default::default()
        };
        let reader = knowdit_audit::chain::ChainReader::connect(options, repo_root, scope).await?;
        if let Some(reader) = &reader
            && self.fork_block.is_none()
            && self.fork_rpc_url.len() == 1
            && let Some(block) = reader
                .chain_or_default(None)
                .ok()
                .and_then(|chain_id| reader.block_for(chain_id))
        {
            tracing::info!("[chain] pinning every harness to block {block}, the block the agent reads");
            self.fork_block = Some(block);
        }
        Ok(reader)
    }

    /// Refuse a run whose engagement expects reportable findings but has no way
    /// to produce one.
    ///
    /// A scope that insures named assets is a statement that the findings are
    /// meant to be submittable, and a submittable finding needs a proof against
    /// the deployed system. With no fork endpoint every proof this run can
    /// produce stands on a synthetic deployment, which the review stage records
    /// as an investigated candidate — so the run would spend its whole budget on
    /// something the engagement cannot use. Measured on two real engagements:
    /// 48 of 48 findings were synthetic-deployment proofs and not one of them was
    /// reportable.
    ///
    /// `accept_synthetic` is the caller's `--allow-scaffolded-poc`: it says "I
    /// know, run anyway", which is the right answer for a reconnaissance pass.
    pub fn ensure_fork_for_engagement(
        &self,
        repo_root: &Path,
        scope_config: Option<&Path>,
        accept_synthetic: bool,
    ) -> Result<()> {
        if !self.fork_rpc_url.is_empty() || accept_synthetic {
            return Ok(());
        }
        let Some(scope) = knowdit_audit::scope::ProgramScope::resolve(repo_root, scope_config)?
        else {
            return Ok(());
        };
        if scope.assets.is_empty() {
            return Ok(());
        }
        Err(color_eyre::eyre::eyre!(
            "the engagement scope for `{}` insures {} asset(s), so its findings are meant to be \
             submittable — and that needs a proof against the deployed system. No \
             `--fork-rpc-url` was given: every proof this run can produce would stand on a \
             synthetic deployment, and the review stage would record all of them as investigated \
             candidates. Pass `--fork-rpc-url <endpoint>` for the chain this scope names, or pass \
             `--allow-scaffolded-poc` to say the synthetic proofs are what you want.",
            scope.program,
            scope.assets.len(),
        ))
    }

    /// Resolve a [`ForgeBackend`] (preflight + feature probe) from
    /// these args. Callers do this exactly once per pipeline
    /// invocation and thread the result through every phase that
    /// needs a forge runtime — the per-spec [`ForgeRunner`]
    /// constructor takes `ForgeBackend` by value, so the
    /// `forge coverage --help` probe runs only once even when fuzz
    /// + regen both run inside the same autoloop cycle.
    pub fn to_forge_backend(&self) -> Result<ForgeBackend> {
        let mem_cap_bytes =
            (self.forge_mem_cap_gb > 0).then(|| self.forge_mem_cap_gb * 1024 * 1024 * 1024);
        ForgeBackend::new(
            self.forge_bin.clone(),
            self.docker_image.clone(),
            mem_cap_bytes,
        )
    }

    /// Smoke-test the forge environment before any LLM-driven phase
    /// touches the project. Build a one-off [`ForgeRunner`] off
    /// `backend` (so the `forge coverage --help` probe doesn't
    /// re-fire) with a short 60s timeout, point it at the same
    /// harness dir the fuzz phase would use, and run
    /// [`ForgeRunner::preflight`] — a trivial compile + execute of
    /// `import "forge-std/Test.sol"`. Returns `Err` with forge's
    /// verbatim stdout/stderr on failure so the user sees the actual
    /// compile error rather than burning extract/map/spec-gen tokens
    /// on a project that was always going to fail.
    ///
    /// Cheap: one short forge subprocess. Designed to be called once
    /// per pipeline invocation at the top of streamloop / autoloop /
    /// standalone fuzz / standalone regen, before any DB-heavy or
    /// LLM-heavy stage runs.
    pub async fn preflight(&self, backend: &ForgeBackend, repo_root: &Path) -> Result<()> {
        // Reuse `to_fuzz_options` purely for its harness_dir resolution
        // logic (foundry.toml → test/knowdit_harness; else
        // knowdit_harness). Other field values are placeholders since
        // `resolve_harness_dir` only reads `harness_dir` + `repo_root`.
        let opts = self.to_fuzz_options(FuzzOptionsBuild {
            repo_root: repo_root.to_path_buf(),
            max_specs: 0,
            concurrency: 1,
            regenerate: false,
            via_ir: false,
        });
        let harness_dir = opts.resolve_harness_dir();
        tokio::fs::create_dir_all(&harness_dir).await.map_err(|e| {
            color_eyre::eyre::eyre!(
                "preflight: failed to create harness_dir {}: {e}",
                harness_dir.display()
            )
        })?;
        let harness_dir = std::fs::canonicalize(&harness_dir).unwrap_or(harness_dir);
        let forge_work_dir = if repo_root.join("foundry.toml").exists() {
            repo_root.to_path_buf()
        } else {
            harness_dir.clone()
        };
        // Preflight harness is one trivial test, so the default 60s is
        // generous even for slow docker pulls — but a `via_ir = true`
        // project recompiles forge-std through the IR pipeline for the
        // fresh preflight compilation unit on every invocation, which
        // can exceed it on its own. `--forge-preflight-timeout-secs`
        // exists for those projects. Independent of
        // `forge_timeout_secs` (which the agent uses for real fuzz
        // runs). Preflight never invokes coverage, so both deadlines
        // are the same here.
        let preflight_timeout = Duration::from_secs(self.forge_preflight_timeout_secs);
        let runner = ForgeRunner::new(
            backend.clone(),
            forge_work_dir,
            ForgeTimeouts {
                test: preflight_timeout,
                coverage: preflight_timeout,
            },
        );
        runner.preflight(&harness_dir).await
    }

    /// Compose a [`SolidityHarness`] in one step: resolve the
    /// [`ForgeBackend`] (one-time `forge coverage --help` probe),
    /// fold these args into [`FuzzOptions`], bundle both with the
    /// [`HarnessKind`] mode. Single seam for "CLI args → backend
    /// handle" so orchestrators don't re-thread the underlying
    /// `(backend, options, kind)` tuple manually.
    pub fn to_solidity_harness(&self, build: FuzzOptionsBuild) -> Result<SolidityHarness> {
        let forge_backend = self.to_forge_backend()?;
        let options = self.to_fuzz_options(build);
        Ok(SolidityHarness::new(
            self.harness_mode.into(),
            options,
            forge_backend,
        ))
    }

    /// Compose [`FuzzOptions`] from `self` plus the per-subcommand
    /// pieces in `build`. Single source of truth for forge-runtime
    /// config — no caller spells `FuzzOptions { ... }` directly.
    pub fn to_fuzz_options(&self, build: FuzzOptionsBuild) -> FuzzOptions {
        let forge_mem_cap_bytes =
            (self.forge_mem_cap_gb > 0).then(|| self.forge_mem_cap_gb * 1024 * 1024 * 1024);
        FuzzOptions {
            repo_root: build.repo_root,
            harness_dir: self.harness_dir.clone(),
            forge_bin: self.forge_bin.clone(),
            docker_image: self.docker_image.clone(),
            forge_mem_cap_bytes,
            forge_timeout_secs: self.forge_timeout_secs,
            forge_preflight_timeout_secs: self.forge_preflight_timeout_secs,
            forge_coverage_timeout_secs: self.forge_coverage_timeout_secs,
            forge_test_runs: self.forge_test_runs,
            forge_coverage_runs: self.forge_coverage_runs,
            max_agent_steps: self.harness_max_agent_steps,
            max_specs: build.max_specs,
            concurrency: build.concurrency.max(1),
            regenerate: build.regenerate,
            debug_prefix: self.harness_debug_prefix.clone(),
            llm_settings: None,
            window_restart_threshold_tokens: self.harness_window_restart_threshold_tokens,
            max_restarts: self.harness_max_restarts,
            max_filter_restarts: self.harness_max_filter_restarts,
            gate2_fidelity_threshold: self.gate2_fidelity_threshold,
            via_ir: build.via_ir,
            list_test_files_max_depth: self.harness_list_test_files_max_depth,
            fork_rpc_urls: self.fork_rpc_url.clone(),
            fork_block: self.fork_block,
            skip_undeployed_specs: self.fuzz_skip_undeployed_specs,
        }
    }
}

#[cfg(test)]
mod engagement_fork_tests {
    use super::*;
    use clap::Parser;

    /// The flattened args need a `Parser` to be built from a command line, and
    /// going through the real parser also keeps the flag spelling honest.
    #[derive(Parser)]
    struct Wrap {
        #[command(flatten)]
        harness: HarnessSharedArgs,
    }

    /// A `HarnessSharedArgs` carrying only the fork endpoints under test.
    fn harness(fork_rpc_url: &[&str]) -> HarnessSharedArgs {
        let mut argv: Vec<String> = vec!["test".to_string()];
        for url in fork_rpc_url {
            argv.push("--fork-rpc-url".to_string());
            argv.push((*url).to_string());
        }
        Wrap::parse_from(argv).harness
    }

    /// A repository carrying a conventional scope file, and one that carries
    /// none.
    fn repo_with_scope(scope_json: Option<&str>) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "knowdit-engagement-fork-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&root).expect("create scratch dir");
        if let Some(json) = scope_json {
            std::fs::write(root.join("knowdit-scope.json"), json).expect("write scope");
        }
        root
    }

    const SCOPED: &str = r#"{
        "program": "Immunefi: Fixture",
        "assets": [{ "name": "VaultCore", "chain_id": 1, "value_usd": 5000000 }]
    }"#;
    const NO_ASSETS: &str = r#"{ "program": "Immunefi: Fixture" }"#;

    #[test]
    fn a_scoped_engagement_without_a_fork_is_refused() {
        let root = repo_with_scope(Some(SCOPED));
        let err = harness(&[])
            .ensure_fork_for_engagement(&root, None, false)
            .expect_err("a scope that insures assets needs deployed state");
        let message = err.to_string();
        assert!(message.contains("--fork-rpc-url"), "{message}");
        assert!(
            message.contains("--allow-scaffolded-poc"),
            "the message must name the way to say 'run anyway': {message}"
        );
        assert!(message.contains("Immunefi: Fixture"), "{message}");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_fork_endpoint_or_an_explicit_relaxation_is_enough() {
        let root = repo_with_scope(Some(SCOPED));
        assert!(
            harness(&["https://rpc.example.invalid"])
                .ensure_fork_for_engagement(&root, None, false)
                .is_ok()
        );
        assert!(
            harness(&[])
                .ensure_fork_for_engagement(&root, None, true)
                .is_ok(),
            "--allow-scaffolded-poc is the operator saying the synthetic proofs are wanted"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn an_unscoped_run_or_a_scope_without_assets_is_left_alone() {
        let unscoped = repo_with_scope(None);
        assert!(
            harness(&[])
                .ensure_fork_for_engagement(&unscoped, None, false)
                .is_ok()
        );
        std::fs::remove_dir_all(&unscoped).ok();

        // A scope that insures nothing yet restricts nothing: reconnaissance
        // runs stay possible.
        let bare = repo_with_scope(Some(NO_ASSETS));
        assert!(
            harness(&[])
                .ensure_fork_for_engagement(&bare, None, false)
                .is_ok()
        );
        std::fs::remove_dir_all(&bare).ok();
    }
}
