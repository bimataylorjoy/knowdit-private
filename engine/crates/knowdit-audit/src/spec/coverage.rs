//! Coverage-first link scheduling.
//!
//! A bounded run keeps a prefix of the link queue, so the queue's order decides
//! which parts of the project are ever looked at. Ordered by priority alone,
//! the prefix collapses onto whichever extract holds the best-ranked links: on
//! the disclosed-bug recall set a 20-link budget spent 19–20 links on a single
//! extract in five of seven projects, one of them drawing 21 of 22
//! specifications from a single historical finding, and every miss was a bug
//! in code no link was ever aimed at.
//!
//! The scheduler therefore orders for breadth before depth:
//!   * across extracts, round-robin — every extract gets its first link before
//!     any gets a second;
//!   * within a round, an extract that reaches a contract no earlier pick has
//!     touched goes ahead of one whose contracts are already covered;
//!   * within an extract, a link repeating a historical finding or a mechanism
//!     the extract has already queued waits behind every link that does not.
//!
//! Priority still decides everything the breadth rules leave open: the rank of
//! an extract is the position of its best link, and ties inside an extract keep
//! the incoming order.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::str::FromStr;

use knowdit_kg_model::ExtractedFunction;

use crate::recall_eval::RecallScorer;

/// A location a run is aimed at: `Contract` or `Contract.function`
/// (`path/Contract.sol:function` also parses). Names are compared the way the
/// recall scorer compares them, so "a focused link" and "a link that aims at a
/// disclosed defect" are the same test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkFocus {
    contract: String,
    function: Option<String>,
}

impl LinkFocus {
    /// The normalized contract key this focus names.
    pub fn contract(&self) -> &str {
        &self.contract
    }

    /// Whether an extract with these functions reaches the location.
    pub fn reaches(&self, functions: &[ExtractedFunction]) -> bool {
        functions.iter().any(|function| {
            RecallScorer::contract_key(&function.contract) == self.contract
                && self
                    .function
                    .as_ref()
                    .is_none_or(|name| RecallScorer::function_key(&function.name) == *name)
        })
    }

    pub fn any_reaches(focus: &[Self], functions: &[ExtractedFunction]) -> bool {
        focus.iter().any(|location| location.reaches(functions))
    }
}

impl FromStr for LinkFocus {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let raw = raw.trim();
        let (contract, function) = match raw.rsplit_once(':') {
            Some((contract, function)) => (contract, Some(function)),
            None => match raw.split_once('.') {
                Some((contract, "sol")) => (contract, None),
                Some((contract, function)) => (contract, Some(function)),
                None => (raw, None),
            },
        };
        let contract = RecallScorer::contract_key(contract);
        if contract.is_empty() {
            return Err(format!("focus {raw:?}: expected Contract or Contract.function"));
        }
        let function = function
            .map(RecallScorer::function_key)
            .filter(|function| !function.is_empty());
        Ok(Self { contract, function })
    }
}

impl fmt::Display for LinkFocus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.function {
            Some(function) => write!(f, "{}.{}", self.contract, function),
            None => write!(f, "{}", self.contract),
        }
    }
}

/// What the scheduler needs to know about one queued link.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverageKey {
    pub extract: i32,
    pub historical: i32,
    /// Corpus mechanism slug, when the finding classifies.
    pub mechanism: Option<&'static str>,
    /// Contracts the extract's functions live in.
    pub contracts: BTreeSet<String>,
}

/// How much of the project a scheduled prefix reaches.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CoverageSummary {
    pub links: usize,
    pub extracts: usize,
    pub historicals: usize,
    pub contracts: usize,
    /// Links sharing their `(extract, historical)` pair with an earlier link.
    pub repeated_pairs: usize,
}

impl CoverageSummary {
    pub fn of<T>(items: &[T], key: impl Fn(&T) -> CoverageKey) -> Self {
        let mut extracts = BTreeSet::new();
        let mut historicals = BTreeSet::new();
        let mut contracts = BTreeSet::new();
        let mut pairs = BTreeSet::new();
        let mut repeated_pairs = 0;
        for item in items {
            let key = key(item);
            extracts.insert(key.extract);
            historicals.insert(key.historical);
            if !pairs.insert((key.extract, key.historical)) {
                repeated_pairs += 1;
            }
            contracts.extend(key.contracts);
        }
        Self {
            links: items.len(),
            extracts: extracts.len(),
            historicals: historicals.len(),
            contracts: contracts.len(),
            repeated_pairs,
        }
    }

    pub fn render(&self) -> String {
        format!(
            "{} link(s) over {} extract(s), {} contract(s), {} historical finding(s); {} repeat(s) of an (extract, historical) pair",
            self.links, self.extracts, self.contracts, self.historicals, self.repeated_pairs
        )
    }
}

pub struct CoverageScheduler;

impl CoverageScheduler {
    /// Reorder `items` (given best-first) breadth-first. The result is a
    /// permutation of the input; truncating it keeps the widest prefix.
    pub fn order<T>(items: Vec<T>, key: impl Fn(&T) -> CoverageKey) -> Vec<T> {
        let mut lanes: Vec<Lane<T>> = Vec::new();
        let mut lane_of: BTreeMap<i32, usize> = BTreeMap::new();
        for item in items {
            let key = key(&item);
            let index = *lane_of.entry(key.extract).or_insert_with(|| {
                lanes.push(Lane::default());
                lanes.len() - 1
            });
            lanes[index].push(key, item);
        }
        for lane in &mut lanes {
            lane.settle();
        }

        let total: usize = lanes.iter().map(|lane| lane.queue.len()).sum();
        let mut out = Vec::with_capacity(total);
        let mut covered: BTreeSet<String> = BTreeSet::new();
        while out.len() < total {
            // One round: every lane with work left gives up one link. Picks
            // are greedy — the best-ranked lane whose next link reaches a
            // contract nothing picked so far has touched, else the
            // best-ranked lane left — so coverage is re-judged after each pick.
            let mut pending: Vec<usize> = (0..lanes.len())
                .filter(|index| !lanes[*index].queue.is_empty())
                .collect();
            while !pending.is_empty() {
                let position = pending
                    .iter()
                    .position(|index| {
                        lanes[*index].queue.front().is_some_and(|(key, _)| {
                            key.contracts.iter().any(|c| !covered.contains(c))
                        })
                    })
                    .unwrap_or(0);
                let index = pending.remove(position);
                if let Some((key, item)) = lanes[index].queue.pop_front() {
                    covered.extend(key.contracts);
                    out.push(item);
                }
            }
        }
        out
    }
}

struct Lane<T> {
    queue: VecDeque<(CoverageKey, T)>,
}

impl<T> Default for Lane<T> {
    fn default() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }
}

impl<T> Lane<T> {
    fn push(&mut self, key: CoverageKey, item: T) {
        self.queue.push_back((key, item));
    }

    /// Demote repeats inside the extract: a link's tier is how many earlier
    /// links already carried its historical finding or its mechanism, and the
    /// lane is stably ordered by tier.
    fn settle(&mut self) {
        let mut historicals: BTreeMap<i32, usize> = BTreeMap::new();
        let mut mechanisms: BTreeMap<&'static str, usize> = BTreeMap::new();
        let mut tiered: Vec<(usize, (CoverageKey, T))> = Vec::with_capacity(self.queue.len());
        for (key, item) in self.queue.drain(..) {
            let by_historical = historicals.entry(key.historical).or_insert(0);
            let mut tier = *by_historical;
            *by_historical += 1;
            if let Some(mechanism) = key.mechanism {
                let by_mechanism = mechanisms.entry(mechanism).or_insert(0);
                tier = tier.max(*by_mechanism);
                *by_mechanism += 1;
            }
            tiered.push((tier, (key, item)));
        }
        tiered.sort_by_key(|(tier, _)| *tier);
        self.queue = tiered.into_iter().map(|(_, entry)| entry).collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (label, extract, historical, mechanism, contracts)
    type Row = (&'static str, i32, i32, Option<&'static str>, &'static [&'static str]);

    fn key(row: &Row) -> CoverageKey {
        CoverageKey {
            extract: row.1,
            historical: row.2,
            mechanism: row.3,
            contracts: row.4.iter().map(|c| c.to_string()).collect(),
        }
    }

    fn labels(rows: Vec<Row>) -> Vec<&'static str> {
        CoverageScheduler::order(rows, key)
            .into_iter()
            .map(|row| row.0)
            .collect()
    }

    #[test]
    fn focus_parses_the_usual_spellings_and_matches_like_the_scorer() {
        let function = |contract: &str, name: &str| ExtractedFunction {
            name: name.to_string(),
            contract: contract.to_string(),
            signature: None,
        };
        let extract = [function("src/Vault.sol", "redeem"), function("src/Vault.sol", "deposit")];

        let by_function: LinkFocus = "Vault.redeem".parse().expect("parse");
        assert!(by_function.reaches(&extract));
        let by_path: LinkFocus = "contracts/vault/Vault.sol:redeem(uint256)".parse().expect("parse");
        assert_eq!(by_path, by_function);
        let by_contract: LinkFocus = "Vault.sol".parse().expect("parse");
        assert!(by_contract.reaches(&extract));
        assert_eq!(by_contract.to_string(), "vault");

        let elsewhere: LinkFocus = "Vault.withdraw".parse().expect("parse");
        assert!(!elsewhere.reaches(&extract));
        assert!(LinkFocus::any_reaches(&[elsewhere, by_function], &extract));
        assert!("".parse::<LinkFocus>().is_err());
    }

    #[test]
    fn every_extract_gets_a_link_before_any_gets_a_second() {
        let rows: Vec<Row> = vec![
            ("a1", 1, 10, None, &["A"]),
            ("a2", 1, 11, None, &["A"]),
            ("a3", 1, 12, None, &["A"]),
            ("b1", 2, 20, None, &["B"]),
            ("c1", 3, 30, None, &["C"]),
            ("b2", 2, 21, None, &["B"]),
        ];
        assert_eq!(labels(rows), ["a1", "b1", "c1", "a2", "b2", "a3"]);
    }

    #[test]
    fn repeated_historical_or_mechanism_waits_inside_its_extract() {
        let rows: Vec<Row> = vec![
            ("same-hist-1", 1, 10, None, &["A"]),
            ("same-hist-2", 1, 10, None, &["A"]),
            ("slippage-1", 1, 11, Some("slippage"), &["A"]),
            ("slippage-2", 1, 12, Some("slippage"), &["A"]),
            ("other", 1, 13, Some("rounding"), &["A"]),
        ];
        assert_eq!(
            labels(rows),
            ["same-hist-1", "slippage-1", "other", "same-hist-2", "slippage-2"]
        );
    }

    #[test]
    fn extract_reaching_an_uncovered_contract_goes_first_in_a_round() {
        let rows: Vec<Row> = vec![
            ("vault-1", 1, 10, None, &["Vault"]),
            ("vault-2", 1, 11, None, &["Vault"]),
            ("vault-again", 2, 20, None, &["Vault"]),
            ("vault-again-2", 2, 21, None, &["Vault"]),
            ("token", 3, 30, None, &["Token"]),
        ];
        // Round one: Vault is new when extract 1 leads, extract 3 still reaches
        // Token, extract 2 adds nothing and goes last. Round two keeps rank.
        assert_eq!(
            labels(rows),
            ["vault-1", "token", "vault-again", "vault-2", "vault-again-2"]
        );
    }

    #[test]
    fn a_truncated_prefix_is_wider_than_priority_order() {
        let mut rows: Vec<Row> = (0..19).map(|i| ("hot", 1, 100 + i % 3, None, &["Zap"][..])).collect();
        rows.push(("cold", 2, 200, None, &["Assimilator"]));
        rows.push(("colder", 3, 300, None, &["Curve"]));
        let before = CoverageSummary::of(&rows[..5], key);
        let ordered = CoverageScheduler::order(rows, key);
        let after = CoverageSummary::of(&ordered[..5], key);
        assert_eq!(before.extracts, 1);
        assert_eq!(after.extracts, 3);
        assert_eq!(after.contracts, 3);
        assert_eq!(after.repeated_pairs, 0);
    }
}
