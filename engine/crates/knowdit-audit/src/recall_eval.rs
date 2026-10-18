//! Recall evaluation against publicly disclosed bounty findings.
//!
//! The question this answers is the one the architecture has to be judged on:
//! given the code as it stood before a paid bug was fixed, does a run find it?
//! Each case is a disclosed finding reduced to what a scorer needs — where the
//! defect lives (contract, function), what class it is, what it cost — plus the
//! repository and commit the run is pointed at. No exploit narrative is stored:
//! a run is scored on whether it reached the defect's location and mechanism,
//! and whether its harness proved something there.
//!
//! A run is read from the layout `workflow streamloop -o <dir>` writes:
//! `specifications/specification_*.json` and `valid_findings/finding_*.json`,
//! both carrying the parsed [`AuditSpecification`] under `specification`.
//! `--only-spec` runs have no valid findings and top out at
//! [`RecallLevel::Mechanism`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, WrapErr};
use serde::{Deserialize, Serialize};

use crate::types::{AuditSpecification, ImpactClass};

/// Where a disclosed defect lives. `function: None` means the write-up pins it
/// to the contract only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisclosedLocation {
    pub contract: String,
    #[serde(default)]
    pub function: Option<String>,
}

/// One publicly disclosed finding, as a scorer needs it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisclosedCase {
    pub id: String,
    pub program: String,
    /// Public write-up the case is taken from.
    pub source: String,
    #[serde(default)]
    pub published: Option<String>,
    /// Git URL of the audited repository.
    pub repo: String,
    /// The commit the run audits: the code before the fix.
    pub vulnerable_commit: String,
    #[serde(default)]
    pub fixed_commit: Option<String>,
    /// Project root inside the repository, when it is not the top level.
    #[serde(default)]
    pub subdir: Option<String>,
    #[serde(default)]
    pub chain_id: Option<u64>,
    #[serde(default)]
    pub fork_block: Option<u64>,
    /// Severity the program paid at (`critical`, `high`, ...).
    pub severity: String,
    #[serde(default)]
    pub payout_usd: Option<u64>,
    #[serde(default)]
    pub impact: Option<ImpactClass>,
    /// Coarse class (`access-control`, `accounting`, `oracle`, ...), for
    /// slicing results.
    #[serde(default)]
    pub bug_class: String,
    pub locations: Vec<DisclosedLocation>,
    /// Terms a specification about this mechanism would use; any one of them
    /// in the specification's text counts. Empty: the mechanism level is
    /// decided by location alone.
    #[serde(default)]
    pub mechanism: Vec<String>,
    /// Known ways the case may be contaminated: the protocol's audits in the
    /// historical knowledge graph, a write-up old and famous enough for the
    /// model to have memorised it.
    #[serde(default)]
    pub leakage: Vec<String>,
    #[serde(default)]
    pub notes: String,
}

impl DisclosedCase {
    /// Directory name a prepared checkout and a run's output use.
    pub fn slug(&self) -> String {
        self.id
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '-' })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisclosedSet {
    pub name: String,
    pub cases: Vec<DisclosedCase>,
}

impl DisclosedSet {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .wrap_err_with(|| format!("failed to read case set {}", path.display()))?;
        serde_json::from_str(&text)
            .wrap_err_with(|| format!("failed to parse case set {}", path.display()))
    }
}

/// How far a run got toward one disclosed defect. Ordered: each level implies
/// the ones below it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecallLevel {
    /// The case has no run output.
    NotRun,
    /// Nothing the run specified touches the defect's contract.
    Missed,
    /// A specification drives the defect's contract.
    Contract,
    /// A specification drives the defect's function.
    Function,
    /// That specification also names the mechanism.
    Mechanism,
    /// A harness proved a specification at the defect's function.
    Proven,
}

impl RecallLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotRun => "not_run",
            Self::Missed => "missed",
            Self::Contract => "contract",
            Self::Function => "function",
            Self::Mechanism => "mechanism",
            Self::Proven => "proven",
        }
    }
}

/// One specification a run produced, with the file it came from.
#[derive(Debug, Clone)]
pub struct RunSpecification {
    pub file: PathBuf,
    pub specification: AuditSpecification,
}

#[derive(Debug, Deserialize)]
struct OnDisk {
    specification: Option<AuditSpecification>,
}

/// A reader's verdict on one run, written after reading its specifications.
/// Keyword matching cannot tell a specification that names the right
/// mechanism from one that uses the same words for something else, so the
/// automatic level is a first pass and this is the score of record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adjudication {
    pub level: RecallLevel,
    /// Specification ids that describe the disclosed defect.
    #[serde(default)]
    pub specifications: Vec<i32>,
    pub note: String,
}

/// What one run wrote.
#[derive(Debug, Clone, Default)]
pub struct RunOutput {
    pub specifications: Vec<RunSpecification>,
    pub valid_findings: Vec<RunSpecification>,
    /// `<dir>/adjudication.json`, when a reader has scored the run.
    pub adjudication: Option<Adjudication>,
}

impl RunOutput {
    /// `None` when `dir` does not exist (the case was not run).
    pub fn load(dir: &Path) -> Result<Option<Self>> {
        if !dir.is_dir() {
            return Ok(None);
        }
        let adjudication_path = dir.join("adjudication.json");
        let adjudication = match adjudication_path.is_file() {
            true => {
                let text = std::fs::read_to_string(&adjudication_path).wrap_err_with(|| {
                    format!("failed to read {}", adjudication_path.display())
                })?;
                Some(serde_json::from_str(&text).wrap_err_with(|| {
                    format!("failed to parse {}", adjudication_path.display())
                })?)
            }
            false => None,
        };
        Ok(Some(Self {
            specifications: Self::read_dir(&dir.join("specifications"))?,
            valid_findings: Self::read_dir(&dir.join("valid_findings"))?,
            adjudication,
        }))
    }

    fn read_dir(dir: &Path) -> Result<Vec<RunSpecification>> {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Ok(Vec::new());
        };
        let mut paths: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("json"))
            .collect();
        paths.sort();
        let mut out = Vec::with_capacity(paths.len());
        for file in paths {
            let text = std::fs::read_to_string(&file)
                .wrap_err_with(|| format!("failed to read {}", file.display()))?;
            let parsed: OnDisk = serde_json::from_str(&text)
                .wrap_err_with(|| format!("failed to parse {}", file.display()))?;
            if let Some(specification) = parsed.specification {
                out.push(RunSpecification {
                    file,
                    specification,
                });
            }
        }
        Ok(out)
    }
}

/// One case's result.
#[derive(Debug, Clone, Serialize)]
pub struct CaseScore {
    pub id: String,
    pub program: String,
    pub severity: String,
    pub bug_class: String,
    /// The score of record: the adjudicated level when there is one,
    /// otherwise the automatic one.
    pub level: RecallLevel,
    /// What keyword and location matching alone concluded.
    pub auto_level: RecallLevel,
    pub adjudication: Option<Adjudication>,
    /// Whether the best-matching specification declared the disclosed impact.
    /// `None` when the case records no impact or nothing matched.
    pub impact_match: Option<bool>,
    pub specifications: usize,
    pub on_target_specifications: usize,
    pub valid_findings: usize,
    pub on_target_findings: usize,
    /// File of the specification that reached `level`.
    pub best: Option<PathBuf>,
    pub leakage: Vec<String>,
}

/// Scores runs against a [`DisclosedSet`].
pub struct RecallScorer;

impl RecallScorer {
    /// Normalize a contract reference the way specifications write them:
    /// `src/Vault.sol`, `Vault.sol:Vault`, `VaultCore.vault`, `Vault(proxy)`.
    pub(crate) fn contract_key(raw: &str) -> String {
        let raw = raw.trim();
        let raw = raw.rsplit(':').next().unwrap_or(raw);
        let raw = raw.rsplit('/').next().unwrap_or(raw);
        let raw = raw.strip_suffix(".sol").unwrap_or(raw);
        let raw = raw.split('.').next().unwrap_or(raw);
        let raw = raw.split('(').next().unwrap_or(raw);
        raw.trim().to_lowercase()
    }

    pub(crate) fn function_key(raw: &str) -> String {
        raw.trim().split('(').next().unwrap_or("").trim().to_lowercase()
    }

    /// Every `(contract, function)` a specification drives, and every contract
    /// it names.
    fn footprint(spec: &AuditSpecification) -> (Vec<(String, String)>, Vec<String>) {
        let mut calls: Vec<(String, String)> = spec
            .sequence
            .iter()
            .map(|step| (Self::contract_key(&step.contract), Self::function_key(&step.function)))
            .collect();
        for property in &spec.properties {
            if let Some((contract, function)) = property.target.split_once('.') {
                calls.push((Self::contract_key(contract), Self::function_key(function)));
            }
        }
        let mut contracts: Vec<String> = calls.iter().map(|(c, _)| c.clone()).collect();
        for state in [&spec.setup, &spec.pre_attack, &spec.post_attack] {
            contracts.extend(state.contracts.keys().map(|c| Self::contract_key(c)));
            contracts.extend(
                state
                    .states_variables
                    .keys()
                    .filter_map(|key| key.split_once('.'))
                    .map(|(c, _)| Self::contract_key(c)),
            );
        }
        (calls, contracts)
    }

    fn level_of(case: &DisclosedCase, spec: &AuditSpecification) -> RecallLevel {
        let (calls, contracts) = Self::footprint(spec);
        let mut level = RecallLevel::Missed;
        for location in &case.locations {
            let contract = Self::contract_key(&location.contract);
            if !contracts.contains(&contract) {
                continue;
            }
            level = level.max(RecallLevel::Contract);
            // A contract-only location cannot be told apart from any other
            // specification on that contract, so the automatic score stops at
            // `Contract` there; only an adjudication lifts it.
            let Some(function) = &location.function else {
                continue;
            };
            let function = Self::function_key(function);
            if !calls.iter().any(|(c, f)| *c == contract && *f == function) {
                continue;
            }
            level = level.max(RecallLevel::Function);
            if Self::names_mechanism(case, spec) {
                level = level.max(RecallLevel::Mechanism);
            }
        }
        level
    }

    fn names_mechanism(case: &DisclosedCase, spec: &AuditSpecification) -> bool {
        if case.mechanism.is_empty() {
            return true;
        }
        let mut text = spec.searchable_text();
        for property in &spec.properties {
            text.push_str(&property.claim);
        }
        let text = text.to_lowercase();
        case.mechanism
            .iter()
            .any(|term| text.contains(&term.trim().to_lowercase()))
    }

    pub fn score(case: &DisclosedCase, run: Option<&RunOutput>) -> CaseScore {
        let mut score = CaseScore {
            id: case.id.clone(),
            program: case.program.clone(),
            severity: case.severity.clone(),
            bug_class: case.bug_class.clone(),
            level: RecallLevel::NotRun,
            auto_level: RecallLevel::NotRun,
            adjudication: None,
            impact_match: None,
            specifications: 0,
            on_target_specifications: 0,
            valid_findings: 0,
            on_target_findings: 0,
            best: None,
            leakage: case.leakage.clone(),
        };
        let Some(run) = run else {
            return score;
        };
        score.level = RecallLevel::Missed;
        score.specifications = run.specifications.len();
        score.valid_findings = run.valid_findings.len();
        let mut best: Option<(RecallLevel, &RunSpecification)> = None;
        for spec in &run.specifications {
            let level = Self::level_of(case, &spec.specification);
            if level >= RecallLevel::Function {
                score.on_target_specifications += 1;
            }
            if best.is_none_or(|(seen, _)| level > seen) {
                best = Some((level, spec));
            }
        }
        for finding in &run.valid_findings {
            if Self::level_of(case, &finding.specification) >= RecallLevel::Function {
                score.on_target_findings += 1;
                best = Some((RecallLevel::Proven, finding));
            }
        }
        if let Some((level, spec)) = best
            && level > RecallLevel::Missed
        {
            score.level = level;
            score.best = Some(spec.file.clone());
            score.impact_match = case
                .impact
                .map(|impact| spec.specification.impact == Some(impact));
        }
        score.auto_level = score.level;
        if let Some(adjudication) = &run.adjudication {
            score.level = adjudication.level;
            score.adjudication = Some(adjudication.clone());
        }
        score
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RecallReport {
    pub set: String,
    pub rows: Vec<CaseScore>,
}

impl RecallReport {
    /// Score every case in `set` against `<results>/<case slug>/`.
    pub fn build(set: &DisclosedSet, results: &Path) -> Result<Self> {
        let mut rows = Vec::with_capacity(set.cases.len());
        for case in &set.cases {
            let run = RunOutput::load(&results.join(case.slug()))?;
            rows.push(RecallScorer::score(case, run.as_ref()));
        }
        Ok(Self {
            set: set.name.clone(),
            rows,
        })
    }

    /// Cases that were run and reached at least `level`.
    pub fn reached(&self, level: RecallLevel) -> usize {
        self.rows.iter().filter(|row| row.level >= level).count()
    }

    pub fn run_count(&self) -> usize {
        self.rows
            .iter()
            .filter(|row| row.level != RecallLevel::NotRun)
            .count()
    }

    pub fn render(&self) -> String {
        let mut out = format!("recall set `{}`\n", self.set);
        for row in &self.rows {
            let level = match &row.adjudication {
                Some(_) => format!("{} (auto {})", row.level.as_str(), row.auto_level.as_str()),
                None => format!("{} (auto)", row.level.as_str()),
            };
            out.push_str(&format!(
                "  {:<30} {:<9} {:<22} {:<28} specs {}/{} on target, findings {}/{}{}\n",
                row.id,
                row.severity,
                row.bug_class,
                level,
                row.on_target_specifications,
                row.specifications,
                row.on_target_findings,
                row.valid_findings,
                if row.leakage.is_empty() { "" } else { "  [leakage]" },
            ));
        }
        let run = self.run_count().max(1);
        let mut by_level: BTreeMap<&str, usize> = BTreeMap::new();
        for level in [
            RecallLevel::Contract,
            RecallLevel::Function,
            RecallLevel::Mechanism,
            RecallLevel::Proven,
        ] {
            by_level.insert(level.as_str(), self.reached(level));
        }
        out.push_str(&format!("{} of {} case(s) run\n", self.run_count(), self.rows.len()));
        for (level, count) in by_level {
            out.push_str(&format!(
                "  reached {level:<9} {count:>3}  ({:.0}%)\n",
                100.0 * count as f64 / run as f64
            ));
        }
        out
    }
}

/// Where a planned link queue aims, before any model call.
///
/// A link carries the functions of the project extract it pairs with a
/// historical finding; the specification agent works from there. So whether a
/// queue can find a defect at all is decided when it is planned: a queue whose
/// links never reach the defect's function cannot produce a specification
/// there, whatever the model does. This separates the two halves of recall —
/// aiming (mapper plus scheduler, measurable for free) and reasoning (the agent,
/// measurable only by running it on the links that do aim at the defect).
pub struct AimScorer;

impl AimScorer {
    /// How close one link comes to `case`: `Function` when its extract holds
    /// a disclosed function, `Contract` when it holds only the contract.
    /// `functions` are `(contract path, function name)` pairs.
    pub fn level_of<'a>(
        case: &DisclosedCase,
        functions: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> RecallLevel {
        let functions: Vec<(String, String)> = functions
            .into_iter()
            .map(|(contract, name)| {
                (RecallScorer::contract_key(contract), RecallScorer::function_key(name))
            })
            .collect();
        let mut level = RecallLevel::Missed;
        for location in &case.locations {
            let contract = RecallScorer::contract_key(&location.contract);
            for (link_contract, link_function) in &functions {
                if *link_contract != contract {
                    continue;
                }
                level = level.max(RecallLevel::Contract);
                if location
                    .function
                    .as_deref()
                    .is_some_and(|function| RecallScorer::function_key(function) == *link_function)
                {
                    return RecallLevel::Function;
                }
            }
        }
        level
    }

    /// Score one planned queue (in run order) against `case`.
    pub fn score<'a, L>(
        case: &DisclosedCase,
        budget: Option<usize>,
        queue: &'a [L],
        functions_of: impl Fn(&'a L) -> Vec<(&'a str, &'a str)>,
    ) -> AimScore {
        let mut score = AimScore {
            id: case.id.clone(),
            severity: case.severity.clone(),
            budget,
            queue: queue.len(),
            level: RecallLevel::Missed,
            first_contract_rank: None,
            first_function_rank: None,
            contract_links: 0,
            function_links: 0,
        };
        for (index, link) in queue.iter().enumerate() {
            let level = Self::level_of(case, functions_of(link));
            if level >= RecallLevel::Contract {
                score.contract_links += 1;
                score.first_contract_rank.get_or_insert(index + 1);
            }
            if level >= RecallLevel::Function {
                score.function_links += 1;
                score.first_function_rank.get_or_insert(index + 1);
            }
            score.level = score.level.max(level);
        }
        score
    }
}

/// One case's aim under one budget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AimScore {
    pub id: String,
    pub severity: String,
    /// `None` is the whole pool the mapper produced (the aiming ceiling).
    pub budget: Option<usize>,
    /// Links in the planned queue.
    pub queue: usize,
    pub level: RecallLevel,
    /// 1-based queue position of the first link reaching the contract.
    pub first_contract_rank: Option<usize>,
    pub first_function_rank: Option<usize>,
    pub contract_links: usize,
    pub function_links: usize,
}

/// Aim scores for a case set across budgets.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AimReport {
    pub rows: Vec<AimScore>,
}

impl AimReport {
    fn budget_label(budget: Option<usize>) -> String {
        budget.map_or_else(|| "pool".to_string(), |budget| budget.to_string())
    }

    pub fn render(&self) -> String {
        let mut budgets: Vec<Option<usize>> = Vec::new();
        let mut ids: Vec<&str> = Vec::new();
        for row in &self.rows {
            if !budgets.contains(&row.budget) {
                budgets.push(row.budget);
            }
            if !ids.contains(&row.id.as_str()) {
                ids.push(&row.id);
            }
        }
        let mut out = String::from("aim (planned queue vs disclosed location; no model calls)\n");
        out.push_str(&format!("  {:<34} {:<9}", "case", "severity"));
        for budget in &budgets {
            out.push_str(&format!(" {:>24}", format!("@{}", Self::budget_label(*budget))));
        }
        out.push('\n');
        for id in &ids {
            let Some(first) = self.rows.iter().find(|row| row.id == *id) else {
                continue;
            };
            out.push_str(&format!("  {:<34} {:<9}", id, first.severity));
            for budget in &budgets {
                let cell = self
                    .rows
                    .iter()
                    .find(|row| row.id == *id && row.budget == *budget)
                    .map_or_else(
                        || "-".to_string(),
                        |row| {
                            let rank = row
                                .first_function_rank
                                .or(row.first_contract_rank)
                                .map_or_else(String::new, |rank| format!(" #{rank}"));
                            format!("{}{} ({}fn)", row.level.as_str(), rank, row.function_links)
                        },
                    );
                out.push_str(&format!(" {cell:>24}"));
            }
            out.push('\n');
        }
        for budget in &budgets {
            let rows: Vec<&AimScore> = self.rows.iter().filter(|row| row.budget == *budget).collect();
            let function = rows.iter().filter(|row| row.level >= RecallLevel::Function).count();
            let contract = rows.iter().filter(|row| row.level >= RecallLevel::Contract).count();
            out.push_str(&format!(
                "  @{:<6} function {}/{}  contract {}/{}\n",
                Self::budget_label(*budget),
                function,
                rows.len(),
                contract,
                rows.len()
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SpecProperty, SuqenceCallStep};

    fn case() -> DisclosedCase {
        DisclosedCase {
            id: "vault-rounding".to_string(),
            program: "Example".to_string(),
            source: String::new(),
            published: None,
            repo: String::new(),
            vulnerable_commit: String::new(),
            fixed_commit: None,
            subdir: None,
            chain_id: None,
            fork_block: None,
            severity: "critical".to_string(),
            payout_usd: None,
            impact: Some(ImpactClass::DirectTheftOfFunds),
            bug_class: "accounting".to_string(),
            locations: vec![DisclosedLocation {
                contract: "Vault".to_string(),
                function: Some("redeem".to_string()),
            }],
            mechanism: vec!["rounding".to_string()],
            leakage: Vec::new(),
            notes: String::new(),
        }
    }

    fn spec(contract: &str, function: &str, summary: &str) -> RunSpecification {
        RunSpecification {
            file: PathBuf::from(format!("{contract}-{function}.json")),
            specification: AuditSpecification {
                summary: summary.to_string(),
                impact: Some(ImpactClass::DirectTheftOfFunds),
                sequence: vec![SuqenceCallStep {
                    contract: contract.to_string(),
                    function: function.to_string(),
                    intention: String::new(),
                }],
                ..Default::default()
            },
        }
    }

    #[test]
    fn aim_ranks_the_first_link_reaching_the_disclosed_function() {
        let case = case();
        let queue: Vec<Vec<(&str, &str)>> = vec![
            vec![("src/Router.sol", "swap")],
            vec![("src/Vault.sol", "deposit")],
            vec![("src/Vault.sol", "redeem(uint256)"), ("src/Vault.sol", "deposit")],
            vec![("contracts/vault/Vault.sol", "redeem")],
        ];
        let score = AimScorer::score(&case, Some(4), &queue, |link| link.clone());
        assert_eq!(score.level, RecallLevel::Function);
        assert_eq!(score.first_contract_rank, Some(2));
        assert_eq!(score.first_function_rank, Some(3));
        assert_eq!(score.function_links, 2);
        assert_eq!(score.contract_links, 3);

        let missed = AimScorer::score(&case, Some(1), &queue[..1], |link| link.clone());
        assert_eq!(missed.level, RecallLevel::Missed);
        assert_eq!(missed.first_contract_rank, None);
    }

    #[test]
    fn levels_follow_location_then_mechanism_then_proof() {
        let case = case();
        let score = |run: RunOutput| RecallScorer::score(&case, Some(&run)).level;
        assert_eq!(RecallScorer::score(&case, None).level, RecallLevel::NotRun);
        assert_eq!(
            score(RunOutput {
                specifications: vec![spec("Router", "swap", "rounding")],
                ..Default::default()
            }),
            RecallLevel::Missed
        );
        assert_eq!(
            score(RunOutput {
                specifications: vec![spec("src/Vault.sol", "deposit", "x")],
                ..Default::default()
            }),
            RecallLevel::Contract
        );
        assert_eq!(
            score(RunOutput {
                specifications: vec![spec("Vault", "redeem(uint256)", "fee bypass")],
                ..Default::default()
            }),
            RecallLevel::Function
        );
        assert_eq!(
            score(RunOutput {
                specifications: vec![spec("Vault.vault", "redeem", "Rounding favours the caller")],
                ..Default::default()
            }),
            RecallLevel::Mechanism
        );
        assert_eq!(
            score(RunOutput {
                valid_findings: vec![spec("Vault", "redeem", "x")],
                ..Default::default()
            }),
            RecallLevel::Proven
        );
    }

    #[test]
    fn a_contract_only_location_stops_the_automatic_score_at_contract() {
        let mut case = case();
        case.locations[0].function = None;
        let run = RunOutput {
            specifications: vec![spec("Vault", "redeem", "rounding")],
            ..Default::default()
        };
        assert_eq!(RecallScorer::score(&case, Some(&run)).level, RecallLevel::Contract);
    }

    #[test]
    fn an_adjudication_is_the_score_of_record() {
        let case = case();
        let run = RunOutput {
            specifications: vec![spec("Vault", "redeem", "rounding favours the caller")],
            adjudication: Some(Adjudication {
                level: RecallLevel::Contract,
                specifications: Vec::new(),
                note: "names rounding, but about a different computation".to_string(),
            }),
            ..Default::default()
        };
        let score = RecallScorer::score(&case, Some(&run));
        assert_eq!(score.auto_level, RecallLevel::Mechanism);
        assert_eq!(score.level, RecallLevel::Contract);
    }

    #[test]
    fn a_property_target_counts_as_driving_the_function() {
        let case = case();
        let mut only_property = spec("Router", "swap", "");
        only_property.specification.properties.push(SpecProperty {
            target: "Vault.redeem".to_string(),
            claim: "redeem rounding must favour the vault".to_string(),
            ..Default::default()
        });
        let run = RunOutput {
            specifications: vec![only_property],
            ..Default::default()
        };
        assert_eq!(RecallScorer::score(&case, Some(&run)).level, RecallLevel::Mechanism);
    }
}
