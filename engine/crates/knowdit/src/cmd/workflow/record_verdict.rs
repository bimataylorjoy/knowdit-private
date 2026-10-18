//! `workflow record-verdict` — record what a bounty program said about a
//! finding that was submitted to it.
//!
//! Every other number this pipeline produces is its own opinion. This command
//! is the entry point for the one number that is not: whether the program paid
//! or declined. A rejection is stored as a settled claim
//! (`reflection.ruled_out_claim`, placed into the conclusion set) so later runs
//! stop re-deriving a class the program has already refused; an acceptance is
//! stored for the record, since a finding the program paid for is not ground
//! to avoid.
//!
//! The name lives under `workflow` because that is where the report it answers
//! comes from (`review-findings`). It makes no model call, and the dispatch in
//! `lib.rs` therefore does not build an LLM for it — a command that records a
//! human's answer should not require a model to be configured.

use std::path::PathBuf;

use clap::{Args, ValueEnum};
use color_eyre::eyre::{Result, WrapErr, eyre};
use knowdit_audit::scope::{KnownIssue, ProgramScope};
use knowdit_repo_model::ProgramVerdict;

use crate::cli::{DatabaseArgs, LoadedRepoDatabase, ProjectArgs};

/// What the program said, as a CLI value.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum VerdictArg {
    /// The program paid for it, or resolved it as a valid issue.
    Accepted,
    /// The program declined it.
    Rejected,
}

impl From<VerdictArg> for ProgramVerdict {
    fn from(verdict: VerdictArg) -> Self {
        match verdict {
            VerdictArg::Accepted => Self::Accepted,
            VerdictArg::Rejected => Self::Rejected,
        }
    }
}

#[derive(Args)]
pub struct RecordVerdictArgs {
    #[command(flatten)]
    pub project: ProjectArgs,

    #[command(flatten)]
    pub db: DatabaseArgs,

    /// Canonical finding id this submission is about — the id
    /// `audit_report.md` and `unique_findings/` use.
    #[arg(long)]
    pub finding: i32,

    /// The program the finding was submitted to, e.g.
    /// `"Immunefi: Example Protocol"`.
    #[arg(long)]
    pub program: String,

    #[arg(long, value_enum)]
    pub verdict: VerdictArg,

    /// The program's answer, in its own words where possible: what it said and
    /// why. This is what a later agent will read instead of re-deriving it.
    #[arg(long)]
    pub reason: String,

    /// Where the answer can be re-checked: report URL, ticket id, or a quoted
    /// line from the response.
    #[arg(long)]
    pub evidence: Option<String>,

    /// Also write the rejection into this engagement scope file
    /// (`knowdit-scope.json`), as a known issue.
    ///
    /// The claim above only reaches an agent that reads it, and only if the run
    /// passes `--inject-ruled-out`; a `known_issues` entry is enforced by the
    /// gate: the next run refuses a specification that reproduces this same
    /// defect, before a harness agent is paid for it. Off by default because it
    /// edits a file you own — the entry is written from the finding's own
    /// contract and function, so review the diff after recording.
    #[arg(long)]
    pub into_scope: Option<PathBuf>,

    /// Replace a claim this finding already carries.
    #[arg(long)]
    pub force: bool,
}

impl RecordVerdictArgs {
    pub async fn run(self) -> Result<()> {
        let LoadedRepoDatabase { repo, .. } = self
            .project
            .to_repo_database(self.db.database_path.clone(), self.db.variant_render_cap)
            .await?;

        // The claim is stored on the representative member's reflection, the
        // same row the report renders its PoC from.
        let members = repo
            .load_report_finding_members(self.finding)
            .await
            .wrap_err_with(|| format!("failed to load members of finding {}", self.finding))?;
        let representative = members.iter().max_by_key(|m| m.representative_key()).ok_or_else(
            || {
                eyre!(
                    "finding {} has no reviewed members — a verdict answers a finding that was \
                     reviewed and reported",
                    self.finding
                )
            },
        )?;
        let reflection_id = representative.review.reflection_id;

        let verdict: ProgramVerdict = self.verdict.into();
        repo.record_program_verdict(
            reflection_id,
            &self.program,
            verdict,
            &self.reason,
            self.evidence.as_deref(),
            self.force,
        )
        .await?;

        println!(
            "recorded: {} {} finding {} (reflection {})",
            self.program,
            verdict.as_str(),
            self.finding,
            reflection_id
        );
        match verdict {
            ProgramVerdict::Rejected => println!(
                "later runs inherit this as settled ground when they pass `--inject-ruled-out`"
            ),
            ProgramVerdict::Accepted => println!(
                "kept for the record — an accepted finding is not ground to avoid, so it is not \
                 replayed as a settled claim"
            ),
        }

        if let Some(path) = self.into_scope.as_deref() {
            self.write_known_issue(path, verdict, &representative.review)?;
        }
        Ok(())
    }

    /// Turn the program's answer into a `known_issues` entry the gate enforces.
    ///
    /// The entry is keyed to what the finding itself named — its primary
    /// contract and function — because that pair is what the specification gate
    /// can match against a later specification's own text. A rejection of a
    /// *different* defect on the same contract must not be swept up by it, and
    /// a defect that reproduces elsewhere on the same function is exactly the
    /// rediscovery this is meant to stop.
    fn write_known_issue(
        &self,
        path: &std::path::Path,
        verdict: ProgramVerdict,
        review: &knowdit_repo_model::ReviewedFinding,
    ) -> Result<()> {
        if verdict != ProgramVerdict::Rejected {
            println!(
                "--into-scope ignored: an accepted finding is not ground to exclude, so nothing \
                 was written to {}",
                path.display()
            );
            return Ok(());
        }
        if review.primary_function.trim().is_empty() {
            tracing::warn!(
                finding = self.finding,
                "the review recorded no primary function, so the known-issue entry matches on the \
                 contract alone and will refuse every specification naming it"
            );
        }
        let mut scope = ProgramScope::load(path).wrap_err_with(|| {
            format!(
                "failed to load the engagement scope at {} — point --into-scope at the file the \
                 run reads, or omit it",
                path.display()
            )
        })?;
        let issue = KnownIssue {
            id: format!("REJ-{}", self.finding),
            title: format!("{} declined this: {}", self.program, self.reason.trim()),
            contracts: vec![review.primary_contract.clone()],
            match_any: match review.primary_function.trim().is_empty() {
                true => Vec::new(),
                false => vec![review.primary_function.trim().to_string()],
            },
            // The program's own answer: enforced, as before this field existed.
            enforce: true,
            summary: self.reason.trim().to_string(),
        };
        let replaced = scope.add_known_issue(issue);
        scope.write(path)?;
        println!(
            "{} known issue REJ-{} in {} (contract `{}`, function `{}`)",
            if replaced { "updated" } else { "added" },
            self.finding,
            path.display(),
            review.primary_contract,
            if review.primary_function.trim().is_empty() {
                "(none recorded)"
            } else {
                review.primary_function.trim()
            },
        );
        Ok(())
    }
}
