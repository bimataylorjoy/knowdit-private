//! `workflow judge-specs` — run the scope judge over specifications already in
//! a project database: to measure a judge prompt against triaged specs, or to
//! re-judge a run made before the judge existed. Read-only: it prints verdicts
//! and writes nothing.

use std::path::PathBuf;
use std::sync::Arc;

use clap::Args;
use color_eyre::eyre::{Result, WrapErr};
use knowdit_audit::scope::ProgramScope;
use knowdit_audit::spec::scope_judge::ScopeJudge;
use knowdit_audit::types::AuditSpecification;
use llmy::client::client::LLM;

use crate::cli::{DatabaseArgs, LoadedRepoDatabase, ProjectArgs};

#[derive(Args)]
pub struct JudgeSpecsArgs {
    #[command(flatten)]
    pub project: ProjectArgs,

    #[command(flatten)]
    pub db: DatabaseArgs,

    /// The engagement scope file the judge reads the program's rules from.
    #[arg(long)]
    pub scope: PathBuf,

    /// Only these specification ids (repeatable). Default: every one.
    #[arg(long = "spec-id")]
    pub spec_ids: Vec<i32>,
}

impl JudgeSpecsArgs {
    pub async fn run(self, llm: &LLM) -> Result<()> {
        let LoadedRepoDatabase { repo, .. } = self
            .project
            .to_repo_database(self.db.database_path.clone(), self.db.variant_render_cap)
            .await?;
        let scope = ProgramScope::load(&self.scope)?;
        let judge = Arc::new(ScopeJudge::new(llm.scope(Some("scope-judge".to_string()), None), &scope));
        let specs = repo
            .load_specifications()
            .await
            .wrap_err("failed to load specifications")?;
        let (mut admitted, mut refused, mut failed) = (0usize, 0usize, 0usize);
        for loaded in specs {
            if !self.spec_ids.is_empty() && !self.spec_ids.contains(&loaded.id) {
                continue;
            }
            let spec: AuditSpecification = match serde_json::from_str(&loaded.specification_json) {
                Ok(spec) => spec,
                Err(err) => {
                    println!("#{}\tunreadable\t{err}", loaded.id);
                    failed += 1;
                    continue;
                }
            };
            match judge.judge(&spec).await {
                Ok(verdict) => {
                    if verdict.admit {
                        admitted += 1;
                    } else {
                        refused += 1;
                    }
                    println!(
                        "#{}\t{}\t{}\t{}",
                        loaded.id,
                        if verdict.admit { "ADMIT" } else { "REFUSE" },
                        verdict.rule.trim(),
                        verdict.reason.trim().replace('\n', " ")
                    );
                }
                Err(err) => {
                    failed += 1;
                    println!("#{}\tno-verdict\t{err:#}", loaded.id);
                }
            }
        }
        println!("admitted {admitted}, refused {refused}, no verdict {failed}");
        Ok(())
    }
}
