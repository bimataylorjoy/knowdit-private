//! `workflow eval-recall` — measure recall against publicly disclosed bounty
//! findings.
//!
//! `prepare` checks every case out at the commit before its fix; each checkout
//! is then run like any engagement (`workflow streamloop -p <slug>:<checkout>
//! -o <results>/<slug>`). `score` reads those outputs and reports, per case, how
//! far the run got toward the disclosed defect. `aim` plans each run's link
//! queue from its project database the way streamloop would and reports
//! whether the queue reaches the defect at all. No model call is made here.

use std::path::{Path, PathBuf};
use std::process::Command;

use clap::{Args, Subcommand};
use color_eyre::eyre::{Result, WrapErr, eyre};
use knowdit_audit::recall_eval::{AimReport, AimScorer, DisclosedCase, DisclosedSet, RecallReport};
use knowdit_audit::spec::{LinkSource, SpecificationGenerator};
use knowdit_repo_model::RepoDatabase;

use crate::cmd::audit::gen_specs::GenSpecsSharedArgs;

#[derive(Args)]
pub struct EvalRecallArgs {
    /// The case set (JSON).
    #[arg(long)]
    pub cases: PathBuf,

    #[command(subcommand)]
    pub command: EvalRecallCommand,
}

#[derive(Subcommand)]
pub enum EvalRecallCommand {
    /// Clone every case's repository and check out its vulnerable commit.
    Prepare(PrepareArgs),
    /// Score run outputs against the cases.
    Score(ScoreArgs),
    /// Plan each run's link queue and report whether it reaches the cases.
    Aim(AimArgs),
}

#[derive(Args)]
pub struct PrepareArgs {
    /// Directory the checkouts go under, one per case slug.
    #[arg(long)]
    pub workdir: PathBuf,

    /// Only these case ids. Repeatable.
    #[arg(long = "case")]
    pub only: Vec<String>,
}

#[derive(Args)]
pub struct ScoreArgs {
    /// Directory holding one streamloop output folder per case slug.
    #[arg(long)]
    pub results: PathBuf,

    /// Write the report as JSON here as well.
    #[arg(long)]
    pub out: Option<PathBuf>,
}

#[derive(Args)]
pub struct AimArgs {
    /// One project run: `<case-id>[,<case-id>...]=<repo database>@<project root>`.
    /// The database is copied before planning (planning writes to it).
    /// Repeatable.
    #[arg(long = "run", required = true)]
    pub runs: Vec<String>,

    /// Link budgets to plan at, each planned separately (the priority order
    /// is re-interleaved after its cut, so one uncut queue cannot stand in
    /// for every budget). `0` plans the whole pool.
    #[arg(long = "budget", value_delimiter = ',', default_value = "0,20,60")]
    pub budgets: Vec<usize>,

    /// Where the database copies go. Defaults to the system temp directory.
    #[arg(long)]
    pub scratch: Option<PathBuf>,

    /// Write the report as JSON here as well.
    #[arg(long)]
    pub out: Option<PathBuf>,

    /// The same spec-generation knobs streamloop takes; `--gen-specs-max-links`
    /// is overridden by `--budget`.
    #[command(flatten)]
    pub gen_specs: GenSpecsSharedArgs,
}

/// One `--run` entry.
struct AimRun {
    cases: Vec<String>,
    database: PathBuf,
    root: PathBuf,
}

impl AimRun {
    fn parse(raw: &str) -> Result<Self> {
        let (cases, rest) = raw
            .split_once('=')
            .ok_or_else(|| eyre!("--run {raw:?}: expected <cases>=<database>@<root>"))?;
        let (database, root) = rest
            .rsplit_once('@')
            .ok_or_else(|| eyre!("--run {raw:?}: expected <database>@<root>"))?;
        Ok(Self {
            cases: cases.split(',').map(|id| id.trim().to_string()).collect(),
            database: PathBuf::from(database),
            root: PathBuf::from(root),
        })
    }
}

impl EvalRecallArgs {
    pub async fn run(self) -> Result<()> {
        let set = DisclosedSet::load(&self.cases)?;
        match self.command {
            EvalRecallCommand::Prepare(args) => args.run(&set),
            EvalRecallCommand::Score(args) => args.run(&set),
            EvalRecallCommand::Aim(args) => args.run(&set).await,
        }
    }
}

impl PrepareArgs {
    fn run(&self, set: &DisclosedSet) -> Result<()> {
        std::fs::create_dir_all(&self.workdir)
            .wrap_err_with(|| format!("failed to create {}", self.workdir.display()))?;
        for case in &set.cases {
            if !self.only.is_empty() && !self.only.contains(&case.id) {
                continue;
            }
            let checkout = self.workdir.join(case.slug());
            match self.prepare(case, &checkout) {
                Ok(root) => println!(
                    "[{}] ready at {} — run: knowdit workflow streamloop -p {}:{} -o <results>/{}",
                    case.id,
                    root.display(),
                    case.slug(),
                    root.display(),
                    case.slug()
                ),
                Err(err) => println!("[{}] FAILED: {err:#}", case.id),
            }
        }
        Ok(())
    }

    /// Clone (once) and check out the vulnerable commit. Returns the project
    /// root, which is the checkout or its `subdir`.
    fn prepare(&self, case: &DisclosedCase, checkout: &Path) -> Result<PathBuf> {
        if !checkout.join(".git").exists() {
            Self::git(None, &["clone", "--quiet", &case.repo, &checkout.display().to_string()])?;
        }
        Self::git(Some(checkout), &["fetch", "--quiet", "origin", &case.vulnerable_commit])
            .or_else(|_| Self::git(Some(checkout), &["fetch", "--quiet", "--all"]))?;
        Self::git(Some(checkout), &["checkout", "--quiet", "--force", &case.vulnerable_commit])?;
        Self::git(Some(checkout), &["submodule", "update", "--quiet", "--init", "--recursive"])?;
        let root = match &case.subdir {
            Some(subdir) => checkout.join(subdir),
            None => checkout.to_path_buf(),
        };
        if !root.is_dir() {
            return Err(eyre!("subdir {} does not exist at the vulnerable commit", root.display()));
        }
        Ok(root)
    }

    fn git(dir: Option<&Path>, args: &[&str]) -> Result<()> {
        let mut command = Command::new("git");
        if let Some(dir) = dir {
            command.current_dir(dir);
        }
        let output = command
            .args(args)
            .output()
            .wrap_err("failed to start git")?;
        if output.status.success() {
            return Ok(());
        }
        Err(eyre!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

impl ScoreArgs {
    fn run(&self, set: &DisclosedSet) -> Result<()> {
        let report = RecallReport::build(set, &self.results)?;
        print!("{}", report.render());
        if let Some(out) = &self.out {
            std::fs::write(out, serde_json::to_string_pretty(&report)?)
                .wrap_err_with(|| format!("failed to write {}", out.display()))?;
            println!("wrote {}", out.display());
        }
        Ok(())
    }
}

impl AimArgs {
    async fn run(&self, set: &DisclosedSet) -> Result<()> {
        let scratch = self.scratch.clone().unwrap_or_else(std::env::temp_dir);
        std::fs::create_dir_all(&scratch)
            .wrap_err_with(|| format!("failed to create {}", scratch.display()))?;
        let mut report = AimReport::default();
        for (ordinal, raw) in self.runs.iter().enumerate() {
            let run = AimRun::parse(raw)?;
            let cases: Vec<&DisclosedCase> = run
                .cases
                .iter()
                .map(|id| {
                    set.cases
                        .iter()
                        .find(|case| case.id == *id)
                        .ok_or_else(|| eyre!("case {id:?} is not in the set"))
                })
                .collect::<Result<_>>()?;
            let copy = Self::copy_database(&run.database, &scratch, ordinal)?;
            let result = self.plan_run(&run, &copy, &cases, &mut report).await;
            Self::remove_database(&copy);
            result?;
        }
        print!("{}", report.render());
        if let Some(out) = &self.out {
            std::fs::write(out, serde_json::to_string_pretty(&report)?)
                .wrap_err_with(|| format!("failed to write {}", out.display()))?;
            println!("wrote {}", out.display());
        }
        Ok(())
    }

    async fn plan_run(
        &self,
        run: &AimRun,
        copy: &Path,
        cases: &[&DisclosedCase],
        report: &mut AimReport,
    ) -> Result<()> {
        let repo = RepoDatabase::open_sqlite(copy.to_path_buf())
            .await
            .wrap_err_with(|| format!("failed to open {}", copy.display()))?;
        let profile = repo.get_project_profile().await?;
        let mut options = self.gen_specs.build_spec_options(
            String::new(),
            LinkSource::Mapper,
            profile,
            false,
            &run.root,
        )?;
        // Planning on a copy: nothing has been generated for the queue that
        // matters here, so every link counts as not started.
        options.regenerate = true;
        let generator = SpecificationGenerator::new();
        for budget in &self.budgets {
            options.max_links = (*budget > 0).then_some(*budget);
            let primed = options.primed(&repo).await;
            let queue = generator
                .plan_link_inputs(&repo, &primed)
                .await?
                .map(|planned| planned.links)
                .unwrap_or_default();
            for case in cases {
                report.rows.push(AimScorer::score(case, options.max_links, &queue, |link| {
                    link.extract
                        .functions
                        .iter()
                        .map(|function| (function.contract.as_str(), function.name.as_str()))
                        .collect()
                }));
            }
        }
        Ok(())
    }

    /// Copy the database (and any WAL beside it) so planning never touches
    /// the run's own state.
    fn copy_database(database: &Path, scratch: &Path, ordinal: usize) -> Result<PathBuf> {
        let copy = scratch.join(format!("aim-{}-{ordinal}.sqlite3", std::process::id()));
        std::fs::copy(database, &copy)
            .wrap_err_with(|| format!("failed to copy {}", database.display()))?;
        for suffix in ["-wal", "-shm"] {
            let side = PathBuf::from(format!("{}{suffix}", database.display()));
            if side.exists() {
                let target = PathBuf::from(format!("{}{suffix}", copy.display()));
                std::fs::copy(&side, &target)
                    .wrap_err_with(|| format!("failed to copy {}", side.display()))?;
            }
        }
        Ok(copy)
    }

    fn remove_database(copy: &Path) {
        for suffix in ["", "-wal", "-shm"] {
            let path = PathBuf::from(format!("{}{suffix}", copy.display()));
            if let Err(error) = std::fs::remove_file(&path)
                && error.kind() != std::io::ErrorKind::NotFound
            {
                tracing::warn!("failed to remove {}: {error}", path.display());
            }
        }
    }
}
