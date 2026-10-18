//! The falsified-findings regression suite.
//!
//! Every entry is a specification the pipeline once graded `ValidFinding` and a
//! human later killed, together with the reason it died and the gate that
//! should have caught it. Replaying them through [`CommitGates`] — the exact
//! path `commit_specification` takes — turns "the refined pipeline would have
//! rejected these" from a claim into a check that runs on every build.
//!
//! A set is one JSON file: the program's scope, the chain answers observed
//! when the findings were falsified (replayed through a fixture reader, so the
//! suite never touches a network), a minimal project — the contracts and
//! function bodies the gates read for access guards and intent quotes — and
//! the cases. A case expects either a refusal by one of a list of rule ids, an
//! admission (the recall side: a real finding must still pass), or is pending
//! while its kill reason is still being written up.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use color_eyre::eyre::{Result, WrapErr};
use knowdit_repo_model::cg::{CallGraph, Contract, FileChunk, FileLocation, Function};
use knowdit_repo_model::inheritance::{ContractInherit, InheritanceGraph};
use knowdit_repo_model::storage::StorageGraph;
use serde::{Deserialize, Serialize};

use crate::chain::ChainReader;
use crate::scope::{ProgramScope, ScopeGate};
use crate::spec::ProjectIndex;
use crate::spec::economic_gate::{CommitGates, EconomicGate, EconomicGateOptions};
use crate::types::AuditSpecification;

fn default_min_loss_usd() -> f64 {
    EconomicGateOptions::default().min_loss_usd
}

fn default_true() -> bool {
    true
}

/// The economic gate's configuration for a set, defaulting to the run
/// defaults so a set tests what a run enforces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionGateConfig {
    #[serde(default = "default_min_loss_usd")]
    pub min_loss_usd: f64,
    #[serde(default = "default_true")]
    pub require_intent: bool,
}

impl Default for RegressionGateConfig {
    fn default() -> Self {
        Self {
            min_loss_usd: default_min_loss_usd(),
            require_intent: true,
        }
    }
}

/// What the chain said when the findings were falsified, keyed by
/// [`crate::chain::ChainQuery::key`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionChain {
    pub chain_id: u64,
    #[serde(default)]
    pub block: Option<u64>,
    #[serde(default)]
    pub responses: BTreeMap<String, String>,
    /// Where each response came from (review note, census, PoC log), keyed
    /// like `responses`. Not read by the replay; kept so a reader can trace
    /// every canned answer back to an observation.
    #[serde(default)]
    pub notes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionFunction {
    pub name: String,
    /// The function as written, header and modifiers included — what the
    /// access-guard check reads.
    #[serde(default)]
    pub source: String,
}

/// One contract of the minimal project a set carries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionContract {
    pub name: String,
    #[serde(default)]
    pub file: String,
    /// Contract source the intent quote is matched against. The function
    /// bodies are appended automatically, so this only needs the comments and
    /// declarations the functions do not carry.
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub inherits: Vec<String>,
    #[serde(default)]
    pub functions: Vec<RegressionFunction>,
}

/// What one case expects from the gates.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum RegressionExpectation {
    /// Refused by any of these rule ids.
    Refused { any_of: Vec<String> },
    /// Admitted — a finding that must survive refinement.
    Admitted,
    /// Not evaluated yet: the kill reason is still being established.
    Pending { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionCase {
    pub id: String,
    #[serde(default)]
    pub claim: String,
    /// Why the finding is not reportable, as the reviewer wrote it.
    #[serde(default)]
    pub kill: String,
    pub expect: RegressionExpectation,
    pub spec: AuditSpecification,
}

/// A regression set, as stored on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionSet {
    pub name: String,
    #[serde(default)]
    pub source: String,
    pub program: ProgramScope,
    #[serde(default)]
    pub gate: RegressionGateConfig,
    #[serde(default)]
    pub chain: Option<RegressionChain>,
    pub project: Vec<RegressionContract>,
    pub cases: Vec<RegressionCase>,
}

/// One case's result.
#[derive(Debug, Clone, Serialize)]
pub struct RegressionRow {
    pub id: String,
    pub expected: String,
    pub got: String,
    /// `None` for a pending case.
    pub pass: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RegressionReport {
    pub set: String,
    pub rows: Vec<RegressionRow>,
}

impl RegressionReport {
    pub fn failures(&self) -> usize {
        self.rows.iter().filter(|row| row.pass == Some(false)).count()
    }

    pub fn render(&self) -> String {
        let mut out = format!("regression set `{}`\n", self.set);
        for row in &self.rows {
            let mark = match row.pass {
                Some(true) => "PASS",
                Some(false) => "FAIL",
                None => "SKIP",
            };
            out.push_str(&format!(
                "  [{mark}] {:<24} expected {:<40} got {}\n",
                row.id, row.expected, row.got
            ));
        }
        let evaluated = self.rows.iter().filter(|row| row.pass.is_some()).count();
        out.push_str(&format!(
            "{} / {evaluated} passed, {} pending\n",
            evaluated - self.failures(),
            self.rows.len() - evaluated
        ));
        out
    }
}

impl RegressionSet {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .wrap_err_with(|| format!("failed to read regression set {}", path.display()))?;
        serde_json::from_str(&text)
            .wrap_err_with(|| format!("failed to parse regression set {}", path.display()))
    }

    /// Build the minimal project index the gates read.
    fn project_index(&self) -> Arc<ProjectIndex> {
        let loc = FileLocation {
            start_line: 1,
            start_column: 0,
            end_line: 1,
            end_column: 0,
        };
        let mut contracts = BTreeMap::new();
        let mut ids_by_name: BTreeMap<String, i32> = BTreeMap::new();
        let mut next_function_id = 1_000;
        for (index, declared) in self.project.iter().enumerate() {
            let id = index as i32 + 1;
            ids_by_name.insert(declared.name.to_lowercase(), id);
            let file = PathBuf::from(if declared.file.is_empty() {
                format!("src/{}.sol", declared.name)
            } else {
                declared.file.clone()
            });
            let functions: Vec<Function> = declared
                .functions
                .iter()
                .map(|function| {
                    next_function_id += 1;
                    Function {
                        id: next_function_id,
                        name: function.name.clone(),
                        args: String::new(),
                        relative_file_path: file.clone(),
                        loc: loc,
                        content: (!function.source.is_empty()).then(|| function.source.clone()),
                        calls: Vec::new(),
                        description: None,
                    }
                })
                .collect();
            let mut source = declared.source.clone();
            for function in &declared.functions {
                source.push('\n');
                source.push_str(&function.source);
            }
            contracts.insert(
                id,
                Contract {
                    id,
                    name: declared.name.clone(),
                    relative_file_path: file,
                    chunk: FileChunk {
                        loc: loc,
                        content: source,
                    },
                    functions,
                    description: None,
                },
            );
        }
        let inherits: Vec<ContractInherit> = self
            .project
            .iter()
            .enumerate()
            .flat_map(|(index, declared)| {
                let child = index as i32 + 1;
                declared
                    .inherits
                    .iter()
                    .filter_map(|parent| ids_by_name.get(&parent.to_lowercase()).copied())
                    .map(move |parent| ContractInherit {
                        contract_id: child,
                        inherited_id: parent,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        Arc::new(ProjectIndex::build(
            CallGraph {
                contracts,
                interfaces: BTreeMap::new(),
            },
            StorageGraph {
                state_variables: BTreeMap::new(),
                contract_variables: Vec::new(),
                function_state_variables: Vec::new(),
            },
            InheritanceGraph::new(inherits),
        ))
    }

    /// The gates as a run with this set's scope and chain would build them.
    /// The scope gate is rooted at a path with no deployment tree, so the
    /// deployment rule stays out of the replay — it depends on a repository
    /// the set does not carry.
    fn gates(&self) -> CommitGates {
        let chain = self.chain.as_ref().map(|chain| {
            Arc::new(ChainReader::fixture(
                chain.chain_id,
                chain.block,
                chain.responses.clone(),
                self.program.assets.clone(),
            ))
        });
        CommitGates {
            scope: Some(Arc::new(ScopeGate::new(
                self.program.clone(),
                PathBuf::from("/nonexistent-regression-root"),
            ))),
            economic: Some(Arc::new(EconomicGate::new(
                EconomicGateOptions {
                    min_loss_usd: self.gate.min_loss_usd,
                    require_intent: self.gate.require_intent,
                },
                Some(&self.program),
            ))),
            chain,
        }
    }

    pub async fn evaluate(&self) -> RegressionReport {
        let project = self.project_index();
        let gates = self.gates();
        let mut rows = Vec::with_capacity(self.cases.len());
        for case in &self.cases {
            let mut spec = case.spec.clone();
            let outcome = gates.admit(&mut spec, &project).await;
            let got = match &outcome {
                Ok(_) => "admitted".to_string(),
                Err(refusal) => format!("refused:{}", refusal.id),
            };
            let (expected, pass) = match &case.expect {
                RegressionExpectation::Refused { any_of } => (
                    format!("refused:{}", any_of.join("|")),
                    Some(matches!(&outcome, Err(refusal) if any_of.contains(&refusal.id))),
                ),
                RegressionExpectation::Admitted => ("admitted".to_string(), Some(outcome.is_ok())),
                RegressionExpectation::Pending { reason } => (format!("pending ({reason})"), None),
            };
            rows.push(RegressionRow {
                id: case.id.clone(),
                expected,
                got,
                pass,
            });
        }
        RegressionReport {
            set: self.name.clone(),
            rows,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every regression set shipped under `ops/refine/`. Each case a set marks
    /// refused must be refused by the rule it names, and each case it marks
    /// admitted must pass. No set shipped yet means nothing to hold the
    /// refinement to, which is reported rather than failed.
    #[tokio::test]
    async fn shipped_regression_sets_hold() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../ops/refine");
        let Ok(entries) = std::fs::read_dir(&root) else {
            println!("no regression directory at {}; skipping", root.display());
            return;
        };
        let mut evaluated = 0;
        for entry in entries {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let set = RegressionSet::load(&path).expect("regression set parses");
            let report = set.evaluate().await;
            println!("{}", report.render());
            assert_eq!(report.failures(), 0, "{}", report.render());
            evaluated += 1;
        }
        if evaluated == 0 {
            println!("no regression set under {}; nothing evaluated", root.display());
        }
    }
}
