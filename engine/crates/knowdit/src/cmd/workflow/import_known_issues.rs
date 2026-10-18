//! `workflow import-known-issues` — turn earlier audit reports into known
//! issues the run starts from.
//!
//! A program does not pay for what a previous audit reported, so every finding
//! in those reports is a duplicate waiting to be rediscovered. This reads the
//! reports (markdown, text, or PDF via `pdftotext`), anchors each finding in the
//! project's own contracts and functions, and merges the result into the
//! engagement scope file. The commit gate then refuses a specification that
//! reproduces an anchored issue, and every agent and the novelty judge see the
//! whole list. No model call is made.

use std::path::PathBuf;

use clap::Args;
use color_eyre::eyre::{Result, WrapErr, eyre};
use knowdit_audit::known_issues::{PriorReportParser, ProjectVocabulary, read_report};
use knowdit_audit::scope::ProgramScope;

use crate::cli::{DatabaseArgs, LoadedRepoDatabase, ProjectArgs};

#[derive(Args)]
pub struct ImportKnownIssuesArgs {
    #[command(flatten)]
    pub project: ProjectArgs,

    #[command(flatten)]
    pub db: DatabaseArgs,

    /// The engagement scope file to merge into (`knowdit-scope.json`). Created
    /// when absent, which requires `--program`.
    #[arg(long)]
    pub scope: PathBuf,

    /// Program name for a scope file this command creates.
    #[arg(long)]
    pub program: Option<String>,

    /// Report files to read. Repeatable. `.pdf` needs `pdftotext`.
    #[arg(long = "report", required = true)]
    pub reports: Vec<PathBuf>,

    /// Prefix for the imported ids (`<label>:H-01`), naming where they came
    /// from — e.g. `c4-2024-12` or `spearbit-v2`. Defaults to each report's
    /// file stem.
    #[arg(long)]
    pub label: Option<String>,

    /// Severities not imported. Gas notes are not defects a program could
    /// have paid for, so they are skipped by default.
    #[arg(long = "skip-severity", default_values_t = vec!["gas".to_string()])]
    pub skip_severity: Vec<String>,

    /// Treat each report as a single free-form note holding one finding (an
    /// operator's kill-ledger entry, a one-issue write-up) rather than a report
    /// listing many findings. The id is the file stem and the whole file is the
    /// body, so a note with no report-style `[H-01]` id lines is still imported.
    #[arg(long = "one-per-file")]
    pub one_per_file: bool,

    /// Import every issue as context only — shown to every agent and the novelty
    /// judge, never enforced at the commit gate. Use for ruled-out directions
    /// (a kill-ledger): a killed hypothesis about a function is not a confirmed
    /// bug there, so it must not hard-block every new finding on that function.
    #[arg(long = "context-only")]
    pub context_only: bool,

    /// Print what would be imported without writing the scope file.
    #[arg(long)]
    pub dry_run: bool,
}

impl ImportKnownIssuesArgs {
    pub async fn run(self) -> Result<()> {
        let LoadedRepoDatabase { repo, .. } = self
            .project
            .to_repo_database(self.db.database_path.clone(), self.db.variant_render_cap)
            .await?;
        let call_graph = repo
            .load_call_graph()
            .await
            .wrap_err("failed to load the project call graph")?;
        let vocabulary = ProjectVocabulary::from_call_graph(&call_graph);
        if vocabulary.is_empty() {
            return Err(eyre!(
                "the project database has no call graph yet, so no issue can be anchored to a \
                 contract — run the static call-graph phase (or a streamloop up to it) first"
            ));
        }
        let parser = PriorReportParser::new(vocabulary);

        let mut scope = if self.scope.is_file() {
            ProgramScope::load(&self.scope)?
        } else {
            let program = self.program.clone().ok_or_else(|| {
                eyre!(
                    "{} does not exist; pass --program to create it",
                    self.scope.display()
                )
            })?;
            ProgramScope::parse(
                &serde_json::to_string(&serde_json::Map::from_iter([(
                    "program".to_string(),
                    serde_json::Value::String(program),
                )]))
                .wrap_err("failed to build an empty scope")?,
                &self.scope,
            )?
        };

        let (mut added, mut replaced, mut enforced, mut context_only, mut skipped) =
            (0usize, 0usize, 0usize, 0usize, 0usize);
        for path in &self.reports {
            let label = self.label.clone().unwrap_or_else(|| {
                path.file_stem()
                    .map(|stem| stem.to_string_lossy().to_string())
                    .unwrap_or_else(|| "prior".to_string())
            });
            let text = read_report(path)?;
            let mut imported = if self.one_per_file {
                let id = path
                    .file_stem()
                    .map(|stem| stem.to_string_lossy().to_string())
                    .unwrap_or_else(|| "note".to_string());
                parser.import_note(&label, &id, &text)
            } else {
                parser.import(&label, &text, &self.skip_severity)
            };
            if self.context_only {
                for issue in &mut imported.issues {
                    issue.enforce = false;
                }
            }
            skipped += imported.skipped;
            println!(
                "{}: {} finding(s), {} skipped by severity",
                path.display(),
                imported.issues.len(),
                imported.skipped
            );
            for issue in imported.issues {
                let anchor = match (issue.contracts.is_empty(), issue.match_any.is_empty()) {
                    (false, false) => format!(
                        "enforced on {} × {}",
                        issue.contracts.join("/"),
                        issue.match_any.join("/")
                    ),
                    (false, true) => format!("context only (contract {}, no specific function)", issue.contracts.join("/")),
                    _ => "context only (names no project contract)".to_string(),
                };
                println!("  [{}] {} — {anchor}", issue.id, issue.title);
                if issue.enforce {
                    enforced += 1;
                } else {
                    context_only += 1;
                }
                if scope.add_known_issue(issue) {
                    replaced += 1;
                } else {
                    added += 1;
                }
            }
        }
        println!(
            "{added} added, {replaced} replaced; {enforced} enforced by the commit gate, \
             {context_only} context-only; {skipped} skipped by severity"
        );
        if self.dry_run {
            println!("--dry-run: {} not written", self.scope.display());
            return Ok(());
        }
        scope.write(&self.scope)?;
        println!("wrote {}", self.scope.display());
        Ok(())
    }
}
