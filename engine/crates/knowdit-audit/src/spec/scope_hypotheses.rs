//! Scope-driven hypothesis planning — the inverted generator.
//!
//! The mapper answers "which historical mechanism does this project's semantic
//! resemble?", and the pipeline then spends a full harness run asking whether
//! that resemblance reproduces. That question is aimed at a *mechanism*, and a
//! mechanism is not what a program pays for: it pays for an impact, on an
//! asset, by an actor it accepts. Two links can be equally strong mechanism
//! matches while only one of them can ever be submitted.
//!
//! This module re-points the queue at the quantities the program publishes:
//!
//! * **the selector is the engagement's own scope** — insured assets and the
//!   impacts the program buys ([`ProgramScope::payable_impacts`]), not the
//!   knowledge graph's categories;
//! * **the project side of a hypothesis is an entry point on an insured
//!   asset** — the functions the project's own extracted semantics anchor on
//!   that contract ([`AssetHypothesis::entry_points`]);
//! * **the knowledge graph is a pattern library** — the matched historical
//!   findings are hints about how a defect of this family arises elsewhere,
//!   which is what [`LinkScopeTarget::render`] tells the agent, so the
//!   specification is written to prove an impact instead of to mirror a
//!   mechanism.
//!
//! What is *not* invented here: the hypothesis still travels on a real
//! `(extract, historical, finding)` link, because that is the unit every
//! downstream stage reads, persists and deduplicates on. An insured asset with
//! no matched historical pattern at all therefore yields no hypothesis — the
//! pattern library has nothing to hint its entry points with — and that is
//! reported as a gap ([`ScopeCoverage::gaps`]) rather than papered over with a
//! synthetic row nobody can ground.
//!
//! Nothing in here is a model call: the plan is a pure read of the scope, the
//! project's extracted semantics and the link queue.

use std::collections::{BTreeMap, BTreeSet};

use knowdit_kg_model::{ExtractedFunction, ExtractedSemantic};
use knowdit_repo_model::LinkInput;

use crate::scope::{ProgramScope, contract_name, same_contract};
use crate::types::ImpactClass;

/// What one hypothesis is for, read from the engagement scope: the insured
/// asset its project side stands on, the entry points that side anchors, and
/// the impacts the program pays for. `None` when it stands on no insured
/// asset — either no scope was declared, or the asset list does not cover it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkScopeTarget {
    /// Insured asset the hypothesis stands on.
    pub asset: String,
    /// What the program says is at risk there, when it says.
    pub value_usd: Option<u64>,
    /// Entry points of that asset this hypothesis anchors
    /// (`Contract.function`), as the project's own semantics name them.
    pub entry_points: Vec<String>,
    /// Impacts the program buys, in its own order.
    pub payable: Vec<ImpactClass>,
}

impl LinkScopeTarget {
    /// Read the target off one link's project side.
    pub fn for_link(scope: &ProgramScope, link: &LinkInput) -> Option<Self> {
        Self::for_extract(scope, &link.extract.functions)
    }

    /// Read the target off one semantic's function anchors. An anchor-less
    /// semantic yields `None`: the surface is unknown, and inventing a target
    /// for it would be a guess the gate would then have to refuse.
    pub fn for_extract(
        scope: &ProgramScope,
        functions: &[ExtractedFunction],
    ) -> Option<Self> {
        let assets = scope.assets_matching(
            functions
                .iter()
                .map(|function| function.contract.as_str()),
        );
        let asset = assets.first()?.clone();
        let entry_points: Vec<String> = functions
            .iter()
            .filter(|function| same_contract(&asset, contract_name(&function.contract)))
            .map(|function| format!("{}.{}", contract_name(&function.contract), function.name))
            .collect();
        let value_usd = scope
            .assets
            .iter()
            .find(|candidate| same_contract(&candidate.name, &asset))
            .and_then(|candidate| candidate.value_usd);
        Some(Self {
            asset,
            value_usd,
            entry_points,
            payable: scope.payable_impacts(),
        })
    }

    /// The block the specification generator reads above the link: what the
    /// hypothesis is aimed at, and what the historical finding below is *not*.
    pub fn render(&self) -> String {
        let mut out = String::from(
            "## This link is aimed at the engagement scope\n\n\
             The program this audit is submitted to insures the asset below and pays for the \
             impacts listed. **That is the question**: whether an actor it accepts can reach one \
             of those impacts here. The historical finding further down is NOT the target — it is \
             one example of how a defect of this family arises in another project, and its \
             mechanism is a hint about what to look for on these entry points.\n\n",
        );
        out.push_str(&format!("- insured asset: `{}`", self.asset));
        if let Some(value) = self.value_usd {
            out.push_str(&format!(" — ${value} at risk"));
        }
        out.push('\n');
        if self.entry_points.is_empty() {
            out.push_str(
                "- entry points on that asset: (this semantic recorded none — use \
                 `lookup_call_graph` on the asset before naming a sequence)\n",
            );
        } else {
            out.push_str(&format!(
                "- entry points this link anchors on that asset: {}\n",
                self.entry_points
                    .iter()
                    .map(|entry| format!("`{entry}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        out.push_str(&format!(
            "- impacts the program pays for: {}\n\n\
             A specification that reaches anything else is not submittable: it costs the same \
             harness run and cannot be reported. Prefer the impact with the shortest unprivileged \
             path from the entry points above; when the only path needs a role the protocol \
             already grants, say so with `set_attacker_model` and expect the commit to be refused.\n",
            self.payable
                .iter()
                .map(|impact| format!("`{}`", impact.as_str()))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        out
    }
}

/// One insured asset as the planner sees it: where a hypothesis could stand,
/// and how much pattern-library material exists to hint it with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetHypothesis {
    pub asset: String,
    pub value_usd: Option<u64>,
    /// `Contract.function` entry points the project's own extracted semantics
    /// anchor on this asset, sorted.
    pub entry_points: Vec<String>,
    /// How many queue entries stand on this asset — i.e. how many historical
    /// patterns are available to hint it with.
    pub hint_links: usize,
}

/// What the planner decided, for the run log: the payable impact list it aimed
/// at, where hypotheses can stand, and where none can.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScopeCoverage {
    pub payable: Vec<ImpactClass>,
    pub assets: Vec<AssetHypothesis>,
    /// Assets the pattern library cannot hint at all — no matched historical
    /// finding touches them, so no hypothesis can be anchored there. One line
    /// per asset, naming what is missing.
    pub gaps: Vec<String>,
}

impl ScopeCoverage {
    /// One line per fact, for the run log.
    pub fn render(&self) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        lines.push(format!(
            "engagement scope: {} payable impact(s) — {}",
            self.payable.len(),
            self.payable
                .iter()
                .map(|impact| impact.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
        for asset in &self.assets {
            lines.push(format!(
                "asset `{}`: {} entry point(s) from the project's own semantics, {} hypothesis \
                 link(s) aiming at it",
                asset.asset,
                asset.entry_points.len(),
                asset.hint_links,
            ));
        }
        lines.extend(self.gaps.iter().cloned());
        lines
    }
}

/// One queue entry the planner reads: an extract id and the project side that
/// travels with it. A named type rather than a tuple so the sort and the
/// coverage report cannot disagree about which field is which.
#[derive(Debug, Clone, Copy)]
struct QueueEntry<'a> {
    extract_id: i32,
    semantic: &'a ExtractedSemantic,
}

/// The planning step itself. One static entry point so callers cannot half-run
/// it: [`Self::apply`] both prioritises the queue and reports coverage.
pub struct ScopeHypothesisPlanner;

impl ScopeHypothesisPlanner {
    /// Aim the link queue at the engagement's assets and impacts.
    ///
    /// Rewrites `links` in place, stably: links whose project side stands on an
    /// insured asset come first, everything else keeps its relative order. That
    /// ordering is the scheduling half — the *aiming* happens in the prompt
    /// ([`LinkScopeTarget`]) and in the commit gate, and the hypotheses
    /// themselves are the links that survive the caller's scope filter.
    ///
    /// Returns the coverage report so the caller can log where hypotheses can
    /// stand and where the pattern library leaves an asset unhintable.
    pub fn apply(
        scope: &ProgramScope,
        extracted_by_id: &BTreeMap<i32, ExtractedSemantic>,
        links: &mut Vec<LinkInput>,
    ) -> ScopeCoverage {
        let queue: Vec<QueueEntry<'_>> = links
            .iter()
            .map(|link| QueueEntry {
                extract_id: link.extract_id,
                semantic: &link.extract,
            })
            .collect();
        let (coverage, aimed) = Self::plan(scope, extracted_by_id, &queue);
        links.sort_by_key(|link| !aimed.contains(&link.extract_id));
        coverage
    }

    /// The pure half of [`Self::apply`]: the coverage report, plus which queue
    /// entries stand on an insured asset (the caller sorts by it).
    fn plan(
        scope: &ProgramScope,
        extracted_by_id: &BTreeMap<i32, ExtractedSemantic>,
        queue: &[QueueEntry<'_>],
    ) -> (ScopeCoverage, BTreeSet<i32>) {
        let payable = scope.payable_impacts();
        if scope.assets.is_empty() {
            return (
                ScopeCoverage {
                    payable,
                    assets: Vec::new(),
                    gaps: Vec::new(),
                },
                BTreeSet::new(),
            );
        }

        let aimed: BTreeSet<i32> = queue
            .iter()
            .filter(|entry| !assets_of(scope, &entry.semantic.functions).is_empty())
            .map(|entry| entry.extract_id)
            .collect();

        let mut assets: Vec<AssetHypothesis> = Vec::new();
        for asset in &scope.assets {
            let mut entry_points: BTreeSet<String> = BTreeSet::new();
            for semantic in extracted_by_id.values() {
                for function in &semantic.functions {
                    if same_contract(&asset.name, contract_name(&function.contract)) {
                        entry_points.insert(format!(
                            "{}.{}",
                            contract_name(&function.contract),
                            function.name
                        ));
                    }
                }
            }
            let hint_links = queue
                .iter()
                .filter(|entry| {
                    assets_of(scope, &entry.semantic.functions)
                        .iter()
                        .any(|matched| same_contract(matched, &asset.name))
                })
                .count();
            assets.push(AssetHypothesis {
                asset: asset.name.clone(),
                value_usd: asset.value_usd,
                entry_points: entry_points.into_iter().collect(),
                hint_links,
            });
        }

        let gaps: Vec<String> = assets
            .iter()
            .filter(|asset| asset.hint_links == 0)
            .map(|asset| {
                format!(
                    "gap: no matched historical pattern touches `{}`, so no hypothesis can stand \
                     there — the pattern library has nothing to hint its {} entry point(s) with",
                    asset.asset,
                    asset.entry_points.len(),
                )
            })
            .collect();

        (
            ScopeCoverage {
                payable,
                assets,
                gaps,
            },
            aimed,
        )
    }
}

/// The insured assets these function anchors stand on.
fn assets_of(scope: &ProgramScope, functions: &[ExtractedFunction]) -> Vec<String> {
    scope.assets_matching(
        functions
            .iter()
            .map(|function| function.contract.as_str()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCOPE: &str = r#"{
        "program": "Immunefi: Fixture",
        "primacy": "rules",
        "assets": [
            { "name": "VaultCore", "chain_id": 1, "value_usd": 5000000 },
            { "name": "RewardsPool", "chain_id": 1 }
        ],
        "impacts_in_scope": ["direct_theft_of_funds", "permanent_freeze"],
        "excluded_impacts": ["temporary_freeze"]
    }"#;

    fn scope() -> ProgramScope {
        ProgramScope::parse(SCOPE, std::path::Path::new("knowdit-scope.json"))
            .expect("scope fixture parses")
    }

    fn semantic(name: &str, functions: &[(&str, &str)]) -> ExtractedSemantic {
        ExtractedSemantic {
            name: name.to_string(),
            category: knowdit_kg::category::DeFiCategory::Others,
            definition: "the mechanism's one-sentence definition".to_string(),
            description: "how value moves through it".to_string(),
            functions: functions
                .iter()
                .map(|(contract, function)| ExtractedFunction {
                    name: function.to_string(),
                    contract: contract.to_string(),
                    signature: Some(format!("{function}()")),
                })
                .collect(),
        }
    }

    fn queue<'a>(entries: &[(i32, &'a ExtractedSemantic)]) -> Vec<QueueEntry<'a>> {
        entries
            .iter()
            .map(|(extract_id, semantic)| QueueEntry {
                extract_id: *extract_id,
                semantic,
            })
            .collect()
    }

    #[test]
    fn payable_impacts_follow_the_program_and_drop_the_excluded() {
        let payable = scope().payable_impacts();
        assert_eq!(
            payable,
            vec![ImpactClass::DirectTheftOfFunds, ImpactClass::PermanentFreeze]
        );
        // No list published: every class but the one nobody buys, minus what the
        // program excludes.
        let mut open = scope();
        open.impacts_in_scope.clear();
        let payable = open.payable_impacts();
        assert!(!payable.contains(&ImpactClass::NoValueAtRisk), "{payable:?}");
        assert!(!payable.contains(&ImpactClass::TemporaryFreeze), "{payable:?}");
        assert!(payable.contains(&ImpactClass::Griefing), "{payable:?}");
    }

    #[test]
    fn target_names_the_asset_its_entry_points_and_the_payable_impacts() {
        let vault = semantic("vault withdrawals", &[("VaultCore", "withdraw")]);
        let target = LinkScopeTarget::for_extract(&scope(), &vault.functions).expect("insured asset");
        assert_eq!(target.asset, "VaultCore");
        assert_eq!(target.value_usd, Some(5_000_000));
        assert_eq!(target.entry_points, ["VaultCore.withdraw".to_string()]);
        let block = target.render();
        assert!(block.contains("$5000000"), "{block}");
        assert!(block.contains("`VaultCore.withdraw`"), "{block}");
        assert!(block.contains("`direct_theft_of_funds`"), "{block}");
        assert!(
            block.contains("NOT the target"),
            "the historical must be framed as a hint: {block}"
        );
    }

    #[test]
    fn a_semantic_on_an_uninsured_contract_has_no_target() {
        let oracle = semantic("oracle reads", &[("PriceOracle", "latestAnswer")]);
        assert!(LinkScopeTarget::for_extract(&scope(), &oracle.functions).is_none());
    }

    #[test]
    fn an_anchorless_semantic_has_no_target() {
        // No recorded functions: the surface is unknown, not insured.
        let unknown = semantic("unknown", &[]);
        assert!(LinkScopeTarget::for_extract(&scope(), &unknown.functions).is_none());
    }

    #[test]
    fn plan_prioritises_insured_assets_and_reports_where_nothing_can_stand() {
        let scope = scope();
        let vault = semantic("vault", &[("VaultCore", "withdraw")]);
        let oracle = semantic("oracle", &[("PriceOracle", "latestAnswer")]);
        let rewards = semantic("rewards", &[("RewardsPool", "claim")]);
        let extracted = BTreeMap::from([
            (1, semantic("vault", &[("VaultCore", "withdraw")])),
            (2, semantic("oracle", &[("PriceOracle", "latestAnswer")])),
            (3, semantic("rewards", &[("RewardsPool", "claim")])),
        ]);

        let (coverage, aimed) = ScopeHypothesisPlanner::plan(
            &scope,
            &extracted,
            &queue(&[(2, &oracle), (1, &vault), (3, &rewards)]),
        );

        assert_eq!(coverage.payable.len(), 2);
        let planned_vault = coverage
            .assets
            .iter()
            .find(|asset| asset.asset == "VaultCore")
            .expect("VaultCore planned");
        assert_eq!(
            planned_vault.entry_points,
            ["VaultCore.withdraw".to_string()]
        );
        assert_eq!(planned_vault.hint_links, 1);
        assert_eq!(aimed, BTreeSet::from([1, 3]));
        assert!(
            coverage.render().iter().any(|line| line.contains("2 payable impact")),
            "{:?}",
            coverage.render()
        );
    }

    #[test]
    fn an_insured_asset_the_pattern_library_cannot_hint_is_reported_as_a_gap() {
        let scope = scope();
        let vault = semantic("vault", &[("VaultCore", "withdraw")]);
        // Nothing in the queue touches RewardsPool, although its entry points
        // are anchored by an extracted semantic.
        let extracted = BTreeMap::from([
            (1, semantic("vault", &[("VaultCore", "withdraw")])),
            (2, semantic("rewards", &[("RewardsPool", "claim")])),
        ]);

        let (coverage, _) = ScopeHypothesisPlanner::plan(&scope, &extracted, &queue(&[(1, &vault)]));

        let rewards = coverage
            .assets
            .iter()
            .find(|asset| asset.asset == "RewardsPool")
            .expect("RewardsPool planned");
        assert_eq!(rewards.entry_points, ["RewardsPool.claim".to_string()]);
        assert_eq!(rewards.hint_links, 0);
        let gap = coverage
            .gaps
            .iter()
            .find(|gap| gap.contains("RewardsPool"))
            .expect("the unhintable asset is reported");
        assert!(gap.contains("no matched historical pattern"), "{gap}");
    }

    #[test]
    fn an_engagement_without_assets_leaves_the_queue_alone() {
        let mut scope = scope();
        scope.assets.clear();
        let vault = semantic("vault", &[("VaultCore", "withdraw")]);
        let extracted = BTreeMap::from([(1, semantic("vault", &[("VaultCore", "withdraw")]))]);

        let (coverage, aimed) =
            ScopeHypothesisPlanner::plan(&scope, &extracted, &queue(&[(1, &vault)]));

        assert!(coverage.assets.is_empty());
        assert!(coverage.gaps.is_empty());
        assert!(aimed.is_empty(), "nothing is prioritised without assets");
        assert_eq!(coverage.payable.len(), 2, "the impact list still reads");
        assert!(LinkScopeTarget::for_extract(&scope, &vault.functions).is_none());
    }
}
