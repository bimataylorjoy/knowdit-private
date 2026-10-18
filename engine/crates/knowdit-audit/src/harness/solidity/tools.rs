//! Solidity-layer agent tools, both shared between the two harness
//! modes. Each tool struct is a `llmy::agent::tool!`-generated type
//! plumbed through [`super::runtime`]'s `AgentLoopState::build_toolbox`.
//!
//! Why one file: the tool types are small data holders + an `invoke`
//! async fn; keeping them together (away from the agent loop and the
//! mode-specific prompts) makes it easy to find every place the agent
//! is allowed to touch.
//!
//! Three tools live here today:
//!
//! * [`WriteHarnessFileTool`] — writes (or overwrites) the per-spec
//!   `.sol` harness file and stamps the new source/filename onto the
//!   shared [`AttemptHandle`] so the orchestrator can persist it as
//!   `CodeGenRecord.harness_source` and reference it from the restart
//!   bootstrap.
//! * [`RunForgeTool`] — runs `forge test --json`, parses the digest,
//!   auto-chases coverage when the test ran cleanly, and emits Gate 1
//!   / Gate 2 verdicts back to the agent in one combined tool result.
//! * [`ListTestFilesTool`] — depth-limited recursive scan of the
//!   project's `test/` / `script/` / `scripts/` / `deploy/` /
//!   `deployments/` directories so the agent can imitate existing
//!   setUp patterns.

use std::path::PathBuf;
use std::sync::Arc;

use knowdit_repo_model::{CoverageEntry, HarnessRunRecord, RunKind};
use schemars::JsonSchema;
use serde::Deserialize;

use super::oracle::OracleGuidance;
use super::utils::{
    AttemptHandle, ForkGuidance, ForgeTestSummary, RecordedRun, parse_lcov, tail_str, truncate_str,
};
use crate::harness::forge::ForgeRunner;
use crate::types::AuditSpecification;

// =============================================================================
// write_harness_file
// =============================================================================

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub(super) struct WriteHarnessFileArgs {
    /// Filename, just the base (no directory). e.g. `"Test_42.t.sol"`.
    /// Re-using the same name overwrites the file on disk.
    filename: String,
    /// Full Solidity source. The required file shape is described by
    /// the system prompt.
    content: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = WriteHarnessFileArgs,
    invoke = invoke,
    description = "Write (or overwrite) the Solidity test file under the per-spec harness directory. The required file shape is dictated entirely by the system prompt — this tool just writes the bytes you provide and remembers the filename for restart bookkeeping. Re-call to update the file after a forge compile error.",
    name = "write_harness_file",
)]
pub(super) struct WriteHarnessFileTool {
    pub(super) attempt: AttemptHandle,
    pub(super) harness_dir: PathBuf,
}

impl WriteHarnessFileTool {
    async fn invoke(
        &self,
        args: WriteHarnessFileArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        self.attempt
            .write_agent_file(&self.harness_dir, &args.filename, &args.content)
            .await
    }
}

// =============================================================================
// run_forge
// =============================================================================

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub(super) struct RunForgeArgs {
    pub(super) match_contract: String,
    #[serde(default)]
    pub(super) match_test: Option<String>,
    #[serde(default)]
    pub(super) seed: Option<i64>,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = RunForgeArgs,
    invoke = invoke,
    description = "Run `forge test --json -vvvvvv` against the harness; if the test exits cleanly and actually exercises the harness (calls > 0, or a deterministic test_… function ran) the orchestrator immediately follows up with `forge coverage --report lcov` server-side and the result includes the coverage gate verdict. Use this whenever you finish writing the harness file. Returns gate verdicts plus the captured exit codes and tails of stdout/stderr.",
    name = "run_forge",
)]
pub(super) struct RunForgeTool {
    pub(super) attempt: AttemptHandle,
    pub(super) runner: ForgeRunner,
    pub(super) test_runs: u64,
    pub(super) coverage_runs: u64,
    pub(super) match_path_glob: String,
    pub(super) spec: Arc<AuditSpecification>,
    pub(super) callgraph: Arc<knowdit_repo_model::cg::CallGraph>,
    pub(super) gate2_fidelity_threshold: f64,
    pub(super) via_ir: bool,
    /// Every contract this spec's sequence calls is deployed on the chain the
    /// run forked, so a production-path proof is available. A violation from
    /// a harness that stayed synthetic is then a rejected reproduction, not a
    /// success — see the Gate 4 branch in `invoke`.
    pub(super) require_production_path: bool,
    /// This run resolved a block for the spec's chain, so a forked harness must
    /// pin its fork to it: a proof standing on chain state that keeps moving
    /// cannot be re-run by whoever triages it. Enforced in the same Gate 4
    /// branch, so the agent still has turns to fix it.
    pub(super) require_pinned_fork: bool,
}

impl RunForgeTool {
    async fn invoke(
        &self,
        args: RunForgeArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let test_argv = self.build_test_argv(&args);
        let test_out = match self.runner.run(RunKind::Test, &test_argv).await {
            Ok(o) => o,
            Err(e) => return Ok(format!("error: forge test spawn failed: {e:#}")),
        };
        let test_summary = ForgeTestSummary::parse(&test_out.stdout);
        let test_record = HarnessRunRecord {
            kind: RunKind::Test,
            seed: args.seed,
            runs: self.test_runs as i64,
            forge_args: test_argv.clone(),
            exit_code: test_out.exit_code,
            stdout: truncate_str(&test_out.stdout, 200_000),
            stderr: truncate_str(&test_out.stderr, 200_000),
            duration_ms: test_out.duration_ms,
            violated: test_summary.violated,
            sequence: test_summary.counterexample_json.clone(),
        };
        let test_gate_verdict = match crate::reflect::runtime_audit::run(&test_summary, &self.spec)
        {
            crate::reflect::AuditOutcome::Fail { reason, .. } => {
                Some(format!("[GATE 1 FAILED — runtime] {reason}"))
            }
            _ => None,
        };

        // Determinism, before anything downstream is spent on this violation: a
        // violation that does not survive a second run of the same harness says
        // something about the state the harness stood on, not about the
        // protocol. Re-running is cheap next to a grading pass — and next to a
        // triager who cannot reproduce the finding.
        let replay = match test_summary.violated {
            true => Some(self.replay(&args, &test_summary).await),
            false => None,
        };
        let replay_disagreed = matches!(replay, Some((ReplayVerdict::Disagreed { .. }, _)));

        // Coverage runs unconditionally. Both modes need it: handler-invariant
        // for Gate 2, PoC mode as evidence that the deterministic test actually
        // exercised project lines (helps the reflect grader judge ValidFinding
        // vs hallucination). It is also the audit-wide breadth signal the link
        // scheduler feeds on, so a run that skips it leaves a permanent hole.
        //
        // The previous `exit_code == 0 && (total_calls > 0 || deterministic)`
        // guard made this dead code in PoC mode: a reproduced PoC fails its
        // `KNOWDIT_POST_ATTACK_REACHED:` assertion, so exit_code != 0, while a
        // clean deterministic run has total_calls == 0 and is not flagged
        // deterministic — the two arms were mutually exclusive and no PoC-mode
        // project ever recorded a single coverage row. `ForgeTimeouts::coverage`
        // bounds the cost of running it on hopeless attempts.
        let mut cov_summary: Option<CoverageBlockBundle> = None;
        // No coverage on a violation that did not reproduce: it would measure
        // how much of the spec a harness walks, and this harness's verdict is
        // already in question.
        if !replay_disagreed {
            let cov_argv = self.build_coverage_argv(&args);
            // Clear the report the previous invocation left behind. `forge
            // coverage` writes `lcov.info` under the project root, and a run
            // that fails before writing (a stack-too-deep compile, a timeout)
            // leaves the last one in place — which would then be read as *this*
            // run's evidence, attributing another harness's lines to this one
            // and letting Gate 2 score a measurement it never took.
            let lcov_path = self.runner.work_dir().join("lcov.info");
            if let Err(error) = tokio::fs::remove_file(&lcov_path).await
                && error.kind() != std::io::ErrorKind::NotFound
            {
                tracing::warn!(
                    path = %lcov_path.display(),
                    "failed to clear the previous coverage report: {error}"
                );
            }
            match self.runner.run(RunKind::Coverage, &cov_argv).await {
                Ok(out) => {
                    let coverage = match tokio::fs::read_to_string(&lcov_path).await {
                        Ok(text) => parse_lcov(&text),
                        Err(_) => Vec::new(),
                    };
                    let gate = if coverage.is_empty() {
                        Some(
                            "[GATE 2 FAILED — coverage] no lcov rows produced (forge coverage ran \
                             but emitted nothing)"
                                .to_string(),
                        )
                    } else if test_summary.deterministic_test_failure {
                        // PoC mode: coverage is informational, not gated.
                        None
                    } else {
                        let thresholds = crate::reflect::AuditThresholds {
                            gate2_fidelity_threshold: self.gate2_fidelity_threshold,
                        };
                        match crate::reflect::coverage_audit::run(
                            &self.spec,
                            &coverage,
                            &self.callgraph,
                            &thresholds,
                        ) {
                            crate::reflect::AuditOutcome::Fail { reason, .. } => {
                                Some(format!("[GATE 2 FAILED — coverage] {reason}"))
                            }
                            _ => None,
                        }
                    };
                    let cov_record = HarnessRunRecord {
                        kind: RunKind::Coverage,
                        seed: args.seed,
                        runs: self.coverage_runs as i64,
                        forge_args: cov_argv.clone(),
                        exit_code: out.exit_code,
                        stdout: truncate_str(&out.stdout, 200_000),
                        stderr: truncate_str(&out.stderr, 200_000),
                        duration_ms: out.duration_ms,
                        violated: false,
                        sequence: None,
                    };
                    cov_summary = Some(CoverageBlockBundle {
                        record: cov_record,
                        coverage,
                        gate_verdict: gate,
                        timed_out: out.timed_out,
                    });
                }
                Err(e) => {
                    tracing::warn!("auto-coverage spawn failed: {e:#}");
                }
            }
        }

        let (production_path, rejected_as_synthetic, oracle_gap) = {
            let mut state = self.attempt.0.lock().await;
            state.runs.push(RecordedRun {
                record: test_record.clone(),
                coverage_idx: None,
            });
            // The replay is a run of its own: it is the evidence a finding's
            // "reproducible" claim rests on, so it is persisted next to the run
            // it re-ran rather than folded into it.
            if let Some((_, replay_record)) = &replay {
                state.runs.push(RecordedRun {
                    record: replay_record.clone(),
                    coverage_idx: None,
                });
            }
            state.last_test_calls = test_summary.total_calls.max(0) as u64;
            if let Some(block) = &cov_summary {
                let coverage_idx = if !block.coverage.is_empty() {
                    state.coverage.push(block.coverage.clone());
                    Some(state.coverage.len() - 1)
                } else {
                    None
                };
                state.runs.push(RecordedRun {
                    record: block.record.clone(),
                    coverage_idx,
                });
                state.last_gate_passed = block.gate_verdict.is_none() && !block.coverage.is_empty();
            } else {
                state.last_gate_passed = false;
            }
            // Reported to the agent below, and separately the input the
            // review stage uses to decide reportability. The agent cannot
            // influence the latter, but a harness that exercises deployed
            // state is the only kind a program pays for, so saying so here
            // is the cheapest place to steer it. Nothing to say before the
            // agent has written a harness.
            let production_path = (!state.harness_source.is_empty()).then(|| {
                crate::reflect::admissibility::PocAdmissibility::of_harness_with(
                    &state.harness_source,
                    // Contracts and interfaces alike: the review stage asks the
                    // same question through `ProjectIndex::has_contract`, and two
                    // answers to "is this one of the project's own names" would
                    // mean a harness this gate accepts and the review demotes.
                    |name| {
                        self.callgraph.contracts.values().any(|c| c.name == name)
                            || self.callgraph.interfaces.values().any(|i| i.name == name)
                    },
                    crate::reflect::admissibility::PocAdmissibilityPolicy {
                        require_pinned_fork: self.require_pinned_fork,
                    },
                )
            });
            // A violation from a synthetic deployment is not a reproduction
            // of the protocol, and this spec could have reached deployed
            // state. Refusing it here is the only way the agent still gets a
            // turn to fork: reporting it as success ends the loop, and the
            // review stage would then spend a full grading pass to demote it.
            let rejected = self.require_production_path
                && test_summary.violated
                && production_path
                    .as_ref()
                    .is_some_and(|p| !p.is_admissible());
            // The oracle the specification declared is what the violation has
            // to be about. A harness that trips its assertion without measuring
            // what the oracle names (no balances for a value claim, no varied
            // input for a bound) has shown something else, and accepting it
            // would end the loop on a proof of the wrong claim.
            let oracle_gap = if test_summary.violated && !rejected {
                OracleGuidance::of(&self.spec).gap(&state.harness_source)
            } else {
                None
            };
            if test_summary.violated && !rejected && !replay_disagreed && oracle_gap.is_none() {
                state.violation_observed = true;
            }
            (production_path, rejected, oracle_gap)
        };
        if let Some(gap) = &oracle_gap {
            tracing::warn!("run_forge: rejected a violation that does not encode the oracle: {gap}");
        }
        if rejected_as_synthetic {
            tracing::warn!(
                "run_forge: rejected a synthetic reproduction — this spec's contracts are deployed \
                 on the forked chain, so the harness must fork them"
            );
        }

        let mut sections: Vec<String> = Vec::new();
        if let Some(v) = &test_gate_verdict {
            sections.push(v.clone());
        }
        if rejected_as_synthetic {
            let labels = production_path
                .as_ref()
                .map(|p| p.signal_labels().join("; "))
                .unwrap_or_else(|| "never forked the deployed system".to_string());
            let remedy = ForkGuidance::new(self.runner.fork_rpc_url(), self.runner.fork_block()).remedy();
            sections.push(format!(
                "[GATE 4 FAILED — production path] a violation was observed, but this harness \
                 {labels}, and every contract this specification calls IS deployed on the forked \
                 chain — a proof against deployed state is available, so this run is NOT accepted \
                 as the reproduction. {remedy} Rewrite `setUp()` to fork and call the deployed \
                 instances, then run forge again."
            ));
        } else if let Some(gap) = &oracle_gap {
            sections.push(format!(
                "[GATE 5 FAILED — oracle] a violation was observed, but {gap}. It is NOT accepted \
                 as the reproduction. Re-read the oracle section of your instructions, measure \
                 what it names, assert on that, then run forge again."
            ));
        } else if replay_disagreed {
            let (first, second) = match replay.as_ref().map(|(verdict, _)| verdict) {
                Some(ReplayVerdict::Disagreed { first, replay }) => {
                    (first.clone(), replay.clone())
                }
                _ => (String::new(), String::new()),
            };
            sections.push(format!(
                "[REPLAY DISAGREED — not accepted as the reproduction] this harness reproduced a \
                 violation on the first run, but a second run of the *same* command did not \
                 reproduce it (first: {first}; replay: {second}). The verdict therefore depends on \
                 state that moved between the two runs — a fork at head, a live RPC read outside \
                 the pinned block, `block.timestamp`-driven setup, or ambient state the harness \
                 does not control. Fix the harness (fork at the run's block and derive all state \
                 from it), then run forge again; nothing downstream has been spent on this result."
            ));
        } else if test_summary.violated {
            let kind = if test_summary.deterministic_test_failure {
                "deterministic test reverted with KNOWDIT_POST_ATTACK_REACHED"
            } else {
                "invariant counter-example produced"
            };
            let replay_note = match replay.as_ref().map(|(verdict, _)| verdict) {
                Some(ReplayVerdict::Agreed) => {
                    " A second run of the same command reproduced it, so the verdict does not depend \
                     on state that moved."
                }
                Some(ReplayVerdict::Unavailable { .. }) => {
                    " The second run could not be executed (see the note below), so reproducibility \
                     is unconfirmed."
                }
                _ => "",
            };
            sections.push(format!(
                "[SUCCESS — violation observed] {kind}.{replay_note} The orchestrator will \
                 terminate after this turn."
            ));
        } else if test_summary.total_calls == 0 && !test_summary.deterministic_test_failure {
            sections.push(
                "[NO-OP] forge test ran but produced 0 calls AND no deterministic test_… revert. \
                 Common causes: (a) `No tests found in project` — your harness is outside the \
                 project's configured test path; (b) `setUp()` reverted before any sequence call \
                 ran; (c) `--match-test` filter excluded every test. Fix the discovery / setUp \
                 issue and call run_forge again."
                    .to_string(),
            );
        }
        if let Some(advisory) = &production_path
            && !advisory.is_admissible()
        {
            let remedy = ForkGuidance::new(self.runner.fork_rpc_url(), self.runner.fork_block()).remedy();
            sections.push(format!(
                "[ADVISORY — production path] this harness {}. {remedy}",
                advisory.signal_labels().join("; ")
            ));
        }
        sections.push(format!(
            "ok: forge test exit={}{} duration_ms={}. runtime: setup_failed={} calls={} \
             reverts={} revert_rate={:.3} violated={} deterministic_violation={}",
            test_record.exit_code,
            if test_out.timed_out { " [TIMEOUT]" } else { "" },
            test_record.duration_ms,
            test_summary.setup_failed,
            test_summary.total_calls,
            test_summary.total_reverts,
            test_summary.revert_rate(),
            test_summary.violated,
            test_summary.deterministic_test_failure,
        ));
        sections.push(format!(
            "--- forge test stdout (last 4K) ---\n{}",
            tail_str(&test_record.stdout, 4_000)
        ));
        sections.push(format!(
            "--- forge test stderr (last 4K) ---\n{}",
            tail_str(&test_record.stderr, 4_000)
        ));

        if let Some(block) = &cov_summary {
            if let Some(v) = &block.gate_verdict {
                sections.push(v.clone());
            }
            sections.push(format!(
                "ok: forge coverage exit={}{} duration_ms={}. runtime: {} lcov rows",
                block.record.exit_code,
                if block.timed_out { " [TIMEOUT]" } else { "" },
                block.record.duration_ms,
                block.coverage.len(),
            ));
            sections.push(format!(
                "--- forge coverage stdout (last 4K) ---\n{}",
                tail_str(&block.record.stdout, 4_000)
            ));
        } else if replay_disagreed {
            sections.push(
                "note: coverage skipped — this violation did not reproduce, so there is nothing to \
                 measure until the harness is fixed."
                    .to_string(),
            );
        } else {
            // Coverage is attempted after every test run now, so reaching here
            // means the spawn itself failed (see the `Err` arm above).
            sections.push(
                "note: orchestrator attempted to run forge coverage but the spawn failed; \
                 coverage gate skipped this turn."
                    .to_string(),
            );
        }
        if let Some((ReplayVerdict::Unavailable { error }, _)) = &replay {
            sections.push(format!(
                "[REPLAY UNAVAILABLE] the second run could not be executed ({error}); the \
                 violation stands as observed but its reproducibility is unconfirmed."
            ));
        }

        Ok(sections.join("\n"))
    }

    /// Re-run the identical forge command once and compare the verdict.
    ///
    /// The second run is what makes "reproduced" mean anything: the harness is
    /// the same, the command is the same, and any difference between the two
    /// results comes from state the harness did not control. When the agent gave
    /// no seed the replay uses the one forge reported for the counter-example,
    /// so the second run is the same experiment rather than a fresh fuzz
    /// campaign.
    async fn replay(
        &self,
        args: &RunForgeArgs,
        first: &ForgeTestSummary,
    ) -> (ReplayVerdict, HarnessRunRecord) {
        let mut argv = self.build_test_argv(args);
        let seed = args.seed.or_else(|| first.counterexample_seed());
        if args.seed.is_none()
            && let Some(seed) = seed
        {
            argv.push("--fuzz-seed".to_string());
            argv.push(seed.to_string());
        }
        let out = match self.runner.run(RunKind::Test, &argv).await {
            Ok(out) => out,
            Err(error) => {
                let record = HarnessRunRecord {
                    kind: RunKind::Replay,
                    seed,
                    runs: self.test_runs as i64,
                    forge_args: argv,
                    exit_code: -1,
                    stdout: String::new(),
                    stderr: format!("{error:#}"),
                    duration_ms: 0,
                    violated: false,
                    sequence: None,
                };
                return (
                    ReplayVerdict::Unavailable {
                        error: format!("{error:#}"),
                    },
                    record,
                );
            }
        };
        let summary = ForgeTestSummary::parse(&out.stdout);
        let record = HarnessRunRecord {
            kind: RunKind::Replay,
            seed,
            runs: self.test_runs as i64,
            forge_args: argv,
            exit_code: out.exit_code,
            stdout: truncate_str(&out.stdout, 200_000),
            stderr: truncate_str(&out.stderr, 200_000),
            duration_ms: out.duration_ms,
            violated: summary.violated,
            sequence: summary.counterexample_json.clone(),
        };
        let verdict = if summary.violated && Self::same_verdict(first, &summary) {
            ReplayVerdict::Agreed
        } else {
            ReplayVerdict::Disagreed {
                first: Self::verdict_label(first),
                replay: Self::verdict_label(&summary),
            }
        };
        (verdict, record)
    }

    /// Whether two runs of one harness reached the same result: the same kind
    /// of violation, and — for a fuzzer-found counter-example — the same call
    /// sequence. A different sequence is a different experiment and is treated
    /// as unreproduced rather than as confirmation.
    fn same_verdict(first: &ForgeTestSummary, replay: &ForgeTestSummary) -> bool {
        first.deterministic_test_failure == replay.deterministic_test_failure
            && (replay.deterministic_test_failure
                || first.counterexample_signature() == replay.counterexample_signature())
    }

    /// One line naming what a run produced, for the disagreement message.
    fn verdict_label(summary: &ForgeTestSummary) -> String {
        let signature = summary.counterexample_signature();
        match (summary.violated, summary.deterministic_test_failure) {
            (false, _) => "no violation".to_string(),
            (true, true) => "deterministic test violation".to_string(),
            (true, false) if signature.is_empty() => {
                "violation with no decoded counter-example".to_string()
            }
            (true, false) => format!(
                "counter-example {}",
                signature
                    .iter()
                    .map(|(contract, function)| format!("{contract}::{function}"))
                    .collect::<Vec<_>>()
                    .join(" → ")
            ),
        }
    }

    fn build_test_argv(&self, args: &RunForgeArgs) -> Vec<String> {
        let mut argv = vec![
            "test".to_string(),
            "--json".to_string(),
            "-vvvvvv".to_string(),
            "--fuzz-runs".to_string(),
            self.test_runs.to_string(),
            "--match-contract".to_string(),
            args.match_contract.clone(),
        ];
        if let Some(t) = &args.match_test {
            argv.push("--match-test".to_string());
            argv.push(t.clone());
        }
        argv.push("--match-path".to_string());
        argv.push(self.match_path_glob.clone());
        if let Some(seed) = args.seed {
            argv.push("--fuzz-seed".to_string());
            argv.push(seed.to_string());
        }
        argv
    }

    fn build_coverage_argv(&self, args: &RunForgeArgs) -> Vec<String> {
        let mut argv = vec![
            "coverage".to_string(),
            "--report".to_string(),
            "lcov".to_string(),
            "--fuzz-runs".to_string(),
            self.coverage_runs.to_string(),
        ];
        if self.via_ir && self.runner.features().coverage_force_via_ir {
            argv.push("--force-via-ir".to_string());
        }
        argv.push("--match-contract".to_string());
        argv.push(args.match_contract.clone());
        if let Some(t) = &args.match_test {
            argv.push("--match-test".to_string());
            argv.push(t.clone());
        }
        argv.push("--match-path".to_string());
        argv.push(self.match_path_glob.clone());
        if let Some(seed) = args.seed {
            argv.push("--fuzz-seed".to_string());
            argv.push(seed.to_string());
        }
        argv
    }
}

/// Intermediate bundle held inside `RunForgeTool::invoke` between the
/// coverage spawn and the eventual feedback assembly.
struct CoverageBlockBundle {
    record: HarnessRunRecord,
    coverage: Vec<CoverageEntry>,
    gate_verdict: Option<String>,
    timed_out: bool,
}

/// What a second run of the same harness said about the first run's violation.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ReplayVerdict {
    /// The same command produced the same violation, so the result does not
    /// depend on state that moved between the runs.
    Agreed,
    /// It did not. `first` / `replay` name what each run produced.
    Disagreed { first: String, replay: String },
    /// The second spawn itself failed — an environment problem, not evidence
    /// about the harness. The violation stands, marked unconfirmed.
    Unavailable { error: String },
}

// =============================================================================
// list_test_files
// =============================================================================

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub(super) struct ListTestFilesArgs {
    #[serde(default)]
    pub(super) filter: Option<String>,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = ListTestFilesArgs,
    invoke = invoke,
    description = "List existing project test/script files (Foundry .t.sol, Hardhat .ts/.js, etc.) under the repo's `test/`, `script/`, `scripts/`, and `deploy/` directories. Use this to find existing setUp() / deployment patterns to imitate before writing your harness's setUp(). Returns relative paths.",
    name = "list_test_files",
)]
pub(super) struct ListTestFilesTool {
    pub(super) repo_root: PathBuf,
    /// Recursion budget passed to [`walk_collect`]. Wired from
    /// `FuzzOptions::list_test_files_max_depth`.
    pub(super) max_depth: usize,
}

/// Vendored / build-output directory names skipped during the
/// `list_test_files` scan. Hardcoded because they're project-tooling
/// conventions (npm, foundry, hardhat), not knowdit knobs.
const LIST_TEST_FILES_SKIP_DIRS: &[&str] =
    &["node_modules", "lib", "out", "cache", "artifacts", ".git"];

/// File extensions the `list_test_files` scan collects. Anything the
/// agent might want to imitate when authoring a harness setUp(): solidity
/// test sources, hardhat ts/js tests, deploy-script json metadata.
const LIST_TEST_FILES_COLLECT_EXTS: &[&str] = &["sol", "ts", "js", "json"];

impl ListTestFilesTool {
    async fn invoke(
        &self,
        args: ListTestFilesArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let accept_dir = |dir: &std::path::Path| -> bool {
            dir.file_name()
                .and_then(|n| n.to_str())
                .map(|name| !LIST_TEST_FILES_SKIP_DIRS.contains(&name))
                .unwrap_or(true)
        };
        let accept_file = |file: &std::path::Path| -> bool {
            file.extension()
                .and_then(|e| e.to_str())
                .map(|ext| LIST_TEST_FILES_COLLECT_EXTS.contains(&ext))
                .unwrap_or(false)
        };
        let mut found: Vec<PathBuf> = Vec::new();
        for sub in ["test", "script", "scripts", "deploy", "deployments"] {
            let dir = self.repo_root.join(sub);
            if !dir.is_dir() {
                continue;
            }
            match knowdit_project::scope::walk_files(
                &dir,
                &accept_dir,
                &accept_file,
                Some(self.max_depth),
            ) {
                Ok(set) => found.extend(set),
                Err(e) => {
                    tracing::warn!(dir = %dir.display(), "list_test_files: walk failed: {e:#}");
                }
            }
        }
        let filter = args.filter.unwrap_or_default();
        let mut out = String::new();
        for p in &found {
            let rel = p
                .strip_prefix(&self.repo_root)
                .unwrap_or(p)
                .to_string_lossy()
                .into_owned();
            if !filter.is_empty() && !rel.contains(&filter) {
                continue;
            }
            out.push_str("- ");
            out.push_str(&rel);
            out.push('\n');
        }
        if out.is_empty() {
            return Ok(format!(
                "no test files found{}",
                if filter.is_empty() {
                    String::new()
                } else {
                    format!(" matching `{filter}`")
                }
            ));
        }
        Ok(format!("ok: {} file(s)\n{out}", found.len()))
    }
}
