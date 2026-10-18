//! Target-derived hypotheses — the queue's first-class unit.
//!
//! The mapper's link answers "which bug from another project does this code
//! resemble?". On code that has been audited before, the answer is mostly a
//! bug that was already found or already ruled out, and a failure mode no
//! corpus has seen yet has no link to arrive on at all. A bounty is paid for
//! the second kind.
//!
//! This module turns what the target's own code implies — the invariants the
//! miner derives from its storage and call graph — into work items that stand
//! on their own: one hypothesis per invariant, anchored on the project
//! semantic whose functions move the state it binds, scheduled ahead of the
//! mapper's links. The knowledge graph is still there, but as a pattern
//! library the agent searches when a hypothesis needs a precedent, not as the
//! thing that decides what gets looked at.
//!
//! A hypothesis travels in the same `(extract, historical, finding)` shape as
//! a mapper link, with a synthetic historical side under a negative id (the
//! knowledge graph only issues positive ones). That is deliberate: every stage
//! downstream — the harness, regen, reflection, review, provenance — reads and
//! persists that triple, and a second unit type would need a second copy of
//! each. The rows are written with
//! [`knowdit_repo_model::RepoDatabase::upsert_target_hypotheses`], which never
//! deletes, so work already done on a hypothesis survives a re-mine.

use std::collections::{BTreeMap, BTreeSet};

use knowdit_kg_model::ExtractedSemantic;
use knowdit_kg_model::audit_finding::FindingSeverity;
use knowdit_kg_model::db::{audit_finding, semantic_node};
use knowdit_repo_model::TargetHypothesisRow;

use crate::scope::{contract_name, same_contract};

use super::invariants::{Falsification, InvariantLayer, MinedInvariant};

/// Whether a link's historical side is a target-derived hypothesis.
pub fn is_target_link(historical_id: i32) -> bool {
    historical_id < 0
}

/// What the planner produced for one pass.
#[derive(Debug, Clone, Default)]
pub struct TargetPlan {
    pub rows: Vec<TargetHypothesisRow>,
    /// Invariants no project semantic anchors — reported, never invented.
    pub gaps: Vec<String>,
}

impl TargetPlan {
    /// The synthetic ids this pass planned, so the scheduler can ignore
    /// hypotheses a previous mine produced and this one no longer does.
    pub fn ids(&self) -> BTreeSet<i32> {
        self.rows.iter().map(|row| row.semantic.id).collect()
    }
}

/// Builds target hypotheses from mined invariants.
pub struct TargetHypothesisPlanner;

impl TargetHypothesisPlanner {
    /// A stable negative id for an invariant: FNV-1a over its identity, folded
    /// into 30 bits so it can never reach `i32::MIN`. Stable across runs so a
    /// resumed run finds the specifications it wrote last time.
    pub fn id_for(invariant: &MinedInvariant) -> i32 {
        let key = format!(
            "{:?}|{}|{}|{}",
            invariant.layer, invariant.kind, invariant.contract, invariant.statement
        );
        let mut hash: u32 = 0x811c_9dc5;
        for byte in key.as_bytes() {
            hash ^= u32::from(*byte);
            hash = hash.wrapping_mul(0x0100_0193);
        }
        -((hash & 0x3fff_ffff) as i32) - 1
    }

    /// Plan one hypothesis per invariant, up to `max`, in the miner's ranking
    /// order (economic first). `extracted_by_id` is keyed the way the project
    /// DB numbers `project_semantic` rows.
    pub fn plan(
        invariants: &[MinedInvariant],
        extracted_by_id: &BTreeMap<i32, ExtractedSemantic>,
        max: usize,
    ) -> TargetPlan {
        let mut plan = TargetPlan::default();
        let mut used: BTreeSet<i32> = BTreeSet::new();
        for invariant in invariants {
            if plan.rows.len() >= max {
                break;
            }
            let Some((extract_id, anchored)) = Self::anchor(invariant, extracted_by_id) else {
                plan.gaps.push(format!(
                    "{} `{}` on {}: no project semantic has a function on that contract",
                    Self::layer_name(invariant.layer),
                    invariant.kind,
                    invariant.contract
                ));
                continue;
            };
            // Collisions are vanishingly rare in 2^30, but a collision would
            // silently merge two hypotheses: probe downward instead.
            let mut id = Self::id_for(invariant);
            while !used.insert(id) {
                id = if id <= -0x3fff_ffff { -1 } else { id - 1 };
            }
            let extract = &extracted_by_id[&extract_id];
            plan.rows.push(TargetHypothesisRow {
                extract_id,
                semantic: semantic_node::Model {
                    id,
                    name: format!("Target invariant: {} on {}", invariant.kind, invariant.contract),
                    definition: invariant.statement.clone(),
                    description: format!(
                        "Derived from the target's own code, not from a historical finding. \
                         Evidence: {}",
                        invariant.evidence.join(", ")
                    ),
                    category: extract.category.clone(),
                },
                finding: Self::finding(id, invariant),
                evidence: format!(
                    "target-derived {} invariant `{}` on {}, anchored on project semantic `{}` \
                     ({anchored} function(s) on that contract)",
                    Self::layer_name(invariant.layer),
                    invariant.kind,
                    invariant.contract,
                    extract.name
                ),
            });
        }
        plan
    }

    /// The project semantic that owns the code the identity is about. A
    /// function the invariant's statement or evidence names — the writer that
    /// moves the state — weighs far more than merely sharing the contract,
    /// which a catch-all "administration" semantic does for every contract it
    /// touches. Ties go to the lowest id so the choice is deterministic.
    fn anchor(
        invariant: &MinedInvariant,
        extracted_by_id: &BTreeMap<i32, ExtractedSemantic>,
    ) -> Option<(i32, usize)> {
        let target = contract_name(&invariant.contract);
        let mentioned = format!("{} {}", invariant.statement, invariant.evidence.join(" "))
            .to_lowercase();
        extracted_by_id
            .iter()
            .map(|(id, extract)| {
                let on_contract: Vec<_> = extract
                    .functions
                    .iter()
                    .filter(|function| same_contract(contract_name(&function.contract), target))
                    .collect();
                let writers = on_contract
                    .iter()
                    .filter(|function| {
                        let name = function.name.trim().to_lowercase();
                        name.len() >= 3 && Self::names(&mentioned, &name)
                    })
                    .count();
                (*id, on_contract.len(), 10 * writers + on_contract.len())
            })
            .filter(|(_, on_contract, _)| *on_contract > 0)
            .max_by(|(left_id, _, left), (right_id, _, right)| {
                left.cmp(right).then(right_id.cmp(left_id))
            })
            .map(|(id, on_contract, _)| (id, on_contract))
    }

    /// Whether `text` names the identifier `name` as a whole word, so `mint`
    /// does not match inside `_minted`.
    fn names(text: &str, name: &str) -> bool {
        text.match_indices(name).any(|(at, _)| {
            let before = text[..at].chars().next_back();
            let after = text[at + name.len()..].chars().next();
            let boundary = |c: Option<char>| c.is_none_or(|c| !(c.is_alphanumeric() || c == '_'));
            boundary(before) && boundary(after)
        })
    }

    fn layer_name(layer: InvariantLayer) -> &'static str {
        match layer {
            InvariantLayer::Accounting => "accounting",
            InvariantLayer::CrossContract => "cross-contract",
            InvariantLayer::Economic => "economic",
        }
    }

    /// The synthetic finding: the hypothesis stated in the fields every prompt
    /// and report already reads.
    fn finding(id: i32, invariant: &MinedInvariant) -> audit_finding::Model {
        let severity = match invariant.layer {
            InvariantLayer::Economic => FindingSeverity::High,
            InvariantLayer::CrossContract | InvariantLayer::Accounting => FindingSeverity::Medium,
        };
        let falsification = match &invariant.falsification {
            Falsification::StateEquality => "Falsify by state: after a call sequence an \
                unprivileged actor can drive, the identity no longer holds — then say who that \
                costs and how much."
                .to_string(),
            Falsification::ValueDelta {
                attacker_gain,
                victim_metric,
            } => format!(
                "Falsify by value delta only: in the same run, the actor's net value \
                 ({attacker_gain}) rises AND the victim metric ({victim_metric}) falls. A state \
                 oddity with no one richer and no one poorer does not count."
            ),
        };
        let where_it_breaks = match invariant.layer {
            InvariantLayer::Accounting => "Look where the total and its parts are written by \
                different functions: rounding in opposite directions, an early return that skips \
                one side, a path that updates the part but not the total, first / last \
                participant and zero-amount edges.",
            InvariantLayer::CrossContract => "Look at the window between the writer and the \
                reader: a call ordering that reads the mirror before it is updated, a path that \
                updates one contract and not the other, a callback or external call in between.",
            InvariantLayer::Economic => "Look for a round trip that returns more than it put in: \
                price or share-rate reads an actor can move within one transaction, donation or \
                direct-transfer inflation, fee and rounding asymmetry between entry and exit, \
                ordering against other users.",
        };
        let mut description = format!(
            "The project's own code implies: {}.\nEvidence: {}.",
            invariant.statement,
            invariant.evidence.join(", ")
        );
        if let Some(asset) = &invariant.asset {
            description.push_str(&format!("\nInsured asset: {asset}"));
            if let Some(value) = invariant.value_usd {
                description.push_str(&format!(" (${value} at risk)"));
            }
        }
        audit_finding::Model {
            id,
            title: format!(
                "[target {}] {}",
                Self::layer_name(invariant.layer),
                invariant.statement
            ),
            severity,
            root_cause: "Target-derived hypothesis with no historical precedent: does any call \
                         sequence an accepted actor can drive break an identity the code relies \
                         on?"
                .to_string(),
            description,
            patterns: falsification,
            exploits: where_it_breaks.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use knowdit_kg_model::category::DeFiCategory;
    use knowdit_kg_model::extracted_semantic::ExtractedFunction;

    fn invariant(contract: &str, statement: &str) -> MinedInvariant {
        MinedInvariant {
            layer: InvariantLayer::Accounting,
            kind: "total-parts",
            contract: contract.to_string(),
            statement: statement.to_string(),
            evidence: vec!["src/Vault.sol:10".to_string()],
            falsification: Falsification::StateEquality,
            asset: None,
            value_usd: None,
        }
    }

    fn extract(name: &str, contracts: &[&str]) -> ExtractedSemantic {
        ExtractedSemantic {
            name: name.to_string(),
            category: DeFiCategory::Yield,
            definition: String::new(),
            description: String::new(),
            functions: contracts
                .iter()
                .map(|contract| ExtractedFunction {
                    name: "f".to_string(),
                    contract: contract.to_string(),
                    signature: None,
                })
                .collect(),
        }
    }

    #[test]
    fn ids_are_negative_and_stable() {
        let a = invariant("Vault", "totalAssets == sum(balances)");
        let id = TargetHypothesisPlanner::id_for(&a);
        assert!(id < 0);
        assert_eq!(id, TargetHypothesisPlanner::id_for(&a.clone()));
        assert_ne!(id, TargetHypothesisPlanner::id_for(&invariant("Vault", "other")));
    }

    #[test]
    fn anchors_on_the_semantic_with_most_functions_and_reports_gaps() {
        let mut extracted = BTreeMap::new();
        extracted.insert(1, extract("Deposit", &["src/Vault.sol"]));
        extracted.insert(2, extract("Withdraw", &["src/Vault.sol", "src/Vault.sol"]));
        let plan = TargetHypothesisPlanner::plan(
            &[
                invariant("Vault", "totalAssets == sum(balances)"),
                invariant("Oracle", "price fresh"),
            ],
            &extracted,
            10,
        );
        assert_eq!(plan.rows.len(), 1);
        assert_eq!(plan.rows[0].extract_id, 2);
        assert!(plan.rows[0].semantic.id < 0);
        assert_eq!(plan.rows[0].semantic.id, plan.rows[0].finding.id);
        assert_eq!(plan.gaps.len(), 1);
    }

    #[test]
    fn prefers_the_semantic_that_owns_the_writer() {
        let mut extracted = BTreeMap::new();
        let mut admin = extract("Administration", &["Token", "Token", "Token"]);
        for (function, name) in admin.functions.iter_mut().zip(["pause", "setOwner", "rescue"]) {
            function.name = name.to_string();
        }
        let mut issuance = extract("Issuance", &["Token"]);
        issuance.functions[0].name = "cashIn".to_string();
        extracted.insert(1, admin);
        extracted.insert(2, issuance);
        let mut identity = invariant("Token", "`Token._totalSupply` moves exactly by the amount `cashIn` minted");
        identity.evidence = vec!["src/Token.sol:40 cashIn".to_string()];
        let plan = TargetHypothesisPlanner::plan(&[identity], &extracted, 10);
        assert_eq!(plan.rows[0].extract_id, 2);
    }

    #[test]
    fn plan_respects_the_cap() {
        let mut extracted = BTreeMap::new();
        extracted.insert(1, extract("Deposit", &["Vault"]));
        let invariants: Vec<MinedInvariant> = (0..5)
            .map(|i| invariant("Vault", &format!("identity {i}")))
            .collect();
        assert_eq!(TargetHypothesisPlanner::plan(&invariants, &extracted, 3).rows.len(), 3);
    }

    /// Plan against a real project database, without a model: set
    /// `KNOWDIT_SMOKE_DB` to a `knowdit.sqlite3` and run with `--ignored`.
    /// The database is copied first — planning writes the hypothesis rows.
    #[tokio::test]
    #[ignore]
    async fn smoke_plan_on_project_db() {
        let Ok(source) = std::env::var("KNOWDIT_SMOKE_DB") else {
            return;
        };
        let copy = std::env::temp_dir().join(format!("knowdit-smoke-{}.sqlite3", std::process::id()));
        std::fs::copy(&source, &copy).expect("copy project db");
        let repo = knowdit_repo_model::RepoDatabase::open_sqlite(copy.clone())
            .await
            .expect("open copy");
        let options = crate::spec::SpecGenOptions::default().primed(&repo).await;
        println!(
            "mined invariants: {}, pattern library: {}",
            options.mined_invariants.len(),
            options.pattern_library.as_ref().map(|l| l.len()).unwrap_or(0)
        );
        let plan = crate::spec::SpecificationGenerator::new()
            .plan_link_inputs(&repo, &options)
            .await
            .expect("plan")
            .expect("links");
        let targets: Vec<_> = plan
            .links
            .iter()
            .filter(|link| is_target_link(link.historical_id))
            .collect();
        println!("links: {} total, {} target-derived", plan.links.len(), targets.len());
        for link in targets.iter().take(8) {
            println!("  [{}] {} -> extract `{}`", link.historical_id, link.finding.title, link.extract.name);
        }
        if let Some(first) = plan.links.first() {
            println!("first scheduled link is target-derived: {}", is_target_link(first.historical_id));
        }
        let _ = std::fs::remove_file(&copy);
    }
}
