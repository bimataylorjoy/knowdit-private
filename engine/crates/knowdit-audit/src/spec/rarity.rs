//! Which historical findings deserve a run.
//!
//! The corpus is not a uniform prior. It is a stack of contest findings in
//! which a handful of mechanisms account for most of the entries — "missing
//! zero-address check", "unchecked return value", "no slippage bound" — and a
//! long tail of mechanisms that occur once or twice. Spending an equal budget
//! on both answers the question "what does this project have in common with the
//! average audited project", which is close to free, and ignores the rare
//! mechanisms, which is where a submission that is not a duplicate comes from.
//!
//! Two deterministic quantities decide the order, and both are needed:
//!
//! * **rarity** — how often the mechanism appears in the corpus, measured over
//!   the whole corpus rather than over the links that happen to be scheduled;
//! * **precondition match** — whether the target project contains the code
//!   shape the mechanism needs at all. Rarity alone would push the queue toward
//!   exotic patterns that cannot occur here; a rare mechanism whose
//!   prerequisite is absent is not a hypothesis, it is a coincidence.
//!
//! `priority = rarity × precondition_match`, so a common mechanism that this
//! project can host still outranks a rare one it cannot. The score orders links
//! *within one extract* (extract fairness is a separate axis and is preserved),
//! and it is printed per link so a run can be ablated without guessing.
//!
//! Nothing here calls a model: the classifier is a keyword table over the
//! finding's own text, the frequency is a count over the corpus, and the
//! precondition is a substring probe against the target's stored functions.

use std::collections::BTreeMap;

use knowdit_repo_model::ProjectFacts;

use crate::types::ImpactClass;

/// One mechanism the corpus contains, with the code signals a project needs
/// before the mechanism can happen there at all.
///
/// The signal lists are probes, not proofs: each is a substring of a function
/// name or body (lowercased), and a miss lowers priority rather than removing
/// the link, because a table written from memory cannot be trusted to be
/// exhaustive.
const MECHANISMS: [(&str, &[&str]); 12] = [
    (
        "oracle-manipulation",
        &[
            "latestanswer",
            "latestrounddata",
            "getreserves",
            "slot0",
            "observe(",
            "consult(",
            "oracle",
            "getprice",
            "pricefeed",
            "twap",
            "oracle manipulation",
            "price manipulation",
            "spot price",
        ],
    ),
    (
        "flash-loan",
        &[
            "flashloan",
            "flashloan(",
            "onflashloan",
            "receiver",
            "executeoperation",
        ],
    ),
    (
        "reentrancy",
        &[
            ".call{",
            ".call(",
            "onerc721received",
            "onerc1155received",
            "receive(",
            "fallback(",
            "externalswap",
            "callback",
            "reentrancy",
            "reentrant",
            "single-function",
        ],
    ),
    (
        "share-inflation",
        &[
            "totalsupply",
            "convertto",
            "pricepershare",
            "shares",
            "deposit",
            "previewdeposit",
        ],
    ),
    (
        "rounding-precision",
        &["decimals", "scale", "div", "muldiv", "/ 10", "round"],
    ),
    (
        "signature-replay",
        &["ecrecover", "ecdsa", "nonce", "permit", "signature", "isvalidsignature"],
    ),
    (
        "cross-chain-replay",
        &["chainid", "lzreceive", "ccip", "messageid", "bridge", "relay"],
    ),
    (
        "access-control",
        &["onlyowner", "onlyrole", "hasrole", "requiresauth", "_checkrole"],
    ),
    (
        "liquidation-accounting",
        &["liquidat", "healthfactor", "collateral", "borrow", "repay"],
    ),
    (
        "fee-accounting",
        &["fee", "commission", "royalt", "surcharge"],
    ),
    (
        "accounting-drift",
        &["totalassets", "totalsupply", "accrue", "index", "exchange_rate"],
    ),
    (
        "upgrade-storage",
        &["initializ", "upgradeto", "proxy", "reinitializ", "gap"],
    ),
];

/// Classify one historical finding's text into a mechanism slug, when the text
/// names one we know.
///
/// The text scanned is the finding's own fields — title, root cause, patterns,
/// description, exploits. Patterns and root cause are weighted by being present
/// at all: they are the fields the original reviewer filled in to say *how*.
pub fn classify(text: &str) -> Option<&'static str> {
    let lower = text.to_lowercase();
    let mut best: Option<(&'static str, usize)> = None;
    for (slug, signals) in MECHANISMS {
        let hits = signals
            .iter()
            .filter(|signal| lower.contains(**signal))
            .count();
        if hits == 0 {
            continue;
        }
        if best.map(|(_, best_hits)| hits > best_hits).unwrap_or(true) {
            best = Some((slug, hits));
        }
    }
    best.map(|(slug, _)| slug)
}

/// How often each mechanism occurs in the corpus.
#[derive(Debug, Clone, Default)]
pub struct RarityIndex {
    counts: BTreeMap<String, usize>,
    total: usize,
}

impl RarityIndex {
    /// Build the index over the whole corpus. `texts` are the finding digests;
    /// unclassifiable findings still count toward the corpus size, which is
    /// what keeps the scale honest.
    pub fn from_texts(texts: &[String]) -> Self {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for text in texts {
            if let Some(slug) = classify(text) {
                *counts.entry(slug.to_string()).or_insert(0) += 1;
            }
        }
        Self {
            total: texts.len(),
            counts,
        }
    }

    /// Rarity of one mechanism in `[0, 1]`: `1` when the corpus contains it
    /// exactly once, lower as it becomes the corpus's steady diet. Uses a log
    /// scale so a mechanism that occurs 5 times is not swamped by one that
    /// occurs 500 times, and normalises by the corpus size so the number means
    /// the same thing on a small and a large corpus.
    pub fn rarity(&self, mechanism: &str) -> f64 {
        if self.total == 0 {
            return 0.5;
        }
        let count = self.counts.get(mechanism).copied().unwrap_or(0);
        let raw = ((1.0 + self.total as f64) / (1.0 + count as f64)).ln();
        let ceiling = (1.0 + self.total as f64).ln();
        (raw / ceiling).clamp(0.0, 1.0)
    }

    pub fn corpus_size(&self) -> usize {
        self.total
    }

    pub fn count(&self, mechanism: &str) -> usize {
        self.counts.get(mechanism).copied().unwrap_or(0)
    }

    /// The mechanisms the corpus contains, rarest first — the shape a report
    /// or a log line wants.
    pub fn rarest(&self, limit: usize) -> Vec<(String, usize, f64)> {
        let mut rows: Vec<(String, usize, f64)> = self
            .counts
            .iter()
            .map(|(slug, count)| (slug.clone(), *count, self.rarity(slug)))
            .collect();
        rows.sort_by(|left, right| left.1.cmp(&right.1).then(left.0.cmp(&right.0)));
        rows.truncate(limit);
        rows
    }
}

/// Which impacts a mechanism can produce at all.
///
/// The gate this feeds answers a different question from the score above: not
/// "is this link likely to yield something" but "if it yields, could the
/// program pay for it". A link whose mechanism can only ever produce griefing
/// fails on a program that buys theft, no matter how rare and well-matched it
/// is — and it fails *before* an agent is paid to write a specification for it,
/// which is the whole point of checking here.
pub fn impacts_of(mechanism: &str) -> &'static [ImpactClass] {
    use ImpactClass::*;
    match mechanism {
        "oracle-manipulation" => &[DirectTheftOfFunds, ProtocolInsolvency, Griefing],
        "flash-loan" => &[DirectTheftOfFunds, ProtocolInsolvency],
        "reentrancy" => &[DirectTheftOfFunds, ProtocolInsolvency, PermanentFreeze],
        "share-inflation" | "accounting-drift" => {
            &[DirectTheftOfFunds, ProtocolInsolvency, TemporaryFreeze]
        }
        "rounding-precision" | "fee-accounting" => {
            &[DirectTheftOfFunds, Griefing, TemporaryFreeze]
        }
        "signature-replay" => &[DirectTheftOfFunds, ProtocolDos],
        // A replay that mints or releases value is theft; one that reverts is
        // an availability bug. Both stay eligible.
        "cross-chain-replay" => &[DirectTheftOfFunds, ProtocolDos, TemporaryFreeze],
        "access-control" => &[DirectTheftOfFunds, PermanentFreeze, ProtocolDos],
        "liquidation-accounting" => {
            &[DirectTheftOfFunds, ProtocolInsolvency, ProtocolDos]
        }
        "upgrade-storage" => &[PermanentFreeze, ProtocolDos, DirectTheftOfFunds],
        // Unclassified findings are not gated: an empty table entry would be a
        // claim about a mechanism nobody has identified.
        _ => &[],
    }
}

/// The score attached to one link.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkPriority {
    pub mechanism: Option<&'static str>,
    pub rarity: f64,
    /// `1.0` when the target contains the mechanism's code signals, `0.35` when
    /// it does not, `0.5` when the finding could not be classified at all.
    pub precondition_match: f64,
    pub priority: f64,
}

impl LinkPriority {
    /// The ordering key: **whether the mechanism can occur here at all**, then
    /// rarity within that group.
    ///
    /// The gate is separate from the score on purpose. A weight would let a
    /// rare mechanism whose prerequisite is absent outrank a common one the
    /// project can actually host — the arithmetic is close enough that the
    /// outcome would depend on how many findings the corpus happens to hold —
    /// and "this cannot happen here" is not a matter of degree. Within the
    /// hostable group, rarity decides, which is the part worth optimising.
    pub fn rank_key(&self) -> (u8, f64) {
        let hostable = (self.precondition_match >= 1.0) as u8;
        (hostable, self.priority)
    }
}

/// Score one historical finding against the target project.
pub fn score(finding_text: &str, facts: &ProjectFacts, index: &RarityIndex) -> LinkPriority {
    match classify(finding_text) {
        Some(mechanism) => {
            let rarity = index.rarity(mechanism);
            let signals = MECHANISMS
                .iter()
                .find(|(slug, _)| *slug == mechanism)
                .map(|(_, signals)| *signals)
                .unwrap_or(&[]);
            let match_score = if project_has_signals(facts, signals) {
                1.0
            } else {
                0.35
            };
            LinkPriority {
                mechanism: Some(mechanism),
                rarity,
                precondition_match: match_score,
                priority: rarity * match_score,
            }
        }
        None => LinkPriority {
            mechanism: None,
            rarity: 0.4,
            precondition_match: 0.5,
            priority: 0.4 * 0.5,
        },
    }
}

/// Whether any non-vendored project function matches one of the signals.
fn project_has_signals(facts: &ProjectFacts, signals: &[&str]) -> bool {
    if signals.is_empty() {
        return false;
    }
    facts.functions.iter().any(|function| {
        if knowdit_repo_model::project_facts::is_vendored_path(&function.path) {
            return false;
        }
        let name = function.name.to_lowercase();
        if signals.iter().any(|signal| name.contains(*signal)) {
            return true;
        }
        let code = function.code();
        signals.iter().any(|signal| code.contains(*signal))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use knowdit_repo_model::{ContractFacts, FunctionFacts, ProjectFacts, StateVarFacts};

    fn facts_with(function_name: &str, body: &str) -> ProjectFacts {
        ProjectFacts {
            contracts: vec![ContractFacts {
                id: 1,
                name: "Vault".to_string(),
                path: "src/Vault.sol".to_string(),
                line: 1,
            }],
            state_variables: vec![StateVarFacts {
                id: 1,
                name: "totalSupply".to_string(),
                type_name: "uint256".to_string(),
                path: "src/Vault.sol".to_string(),
                line: 10,
                contract: Some("Vault".to_string()),
            }],
            functions: vec![FunctionFacts {
                id: 1,
                name: function_name.to_string(),
                contract: Some("Vault".to_string()),
                path: "src/Vault.sol".to_string(),
                line: 30,
                content: body.to_string(),
                writes: vec![1],
                reads: vec![],
            }],
            inherits: Vec::new(),
        }
    }

    #[test]
    fn findings_are_classified_by_mechanism_not_by_title() {
        assert_eq!(
            classify("Reentrancy in withdraw() drains the pool"),
            Some("reentrancy")
        );
        assert_eq!(
            classify("Stale price from a manipulable oracle lets an attacker mint"),
            Some("oracle-manipulation")
        );
        assert_eq!(
            classify("Missing deadline check on the swap router"),
            None,
            "a class outside the table stays unclassified rather than being forced into one"
        );
        assert_eq!(classify("naming inconsistency in the docs"), None);
    }

    #[test]
    fn a_mechanism_that_appears_once_is_rarer_than_one_that_appears_everywhere() {
        // 200 findings: 1 oracle-manipulation, 150 reentrancy.
        let mut texts = vec!["oracle manipulation on the price feed".to_string()];
        texts.extend(vec!["reentrancy with .call{ and callback".to_string(); 150]);
        texts.extend(vec!["something else entirely".to_string(); 49]);
        let index = RarityIndex::from_texts(&texts);

        assert_eq!(index.corpus_size(), 200);
        assert!(index.count("oracle-manipulation") == 1);
        assert!(index.count("reentrancy") == 150);
        assert!(
            index.rarity("oracle-manipulation") > index.rarity("reentrancy"),
            "rare mechanism must score higher: {} vs {}",
            index.rarity("oracle-manipulation"),
            index.rarity("reentrancy")
        );
        let rarest = index.rarest(1);
        assert_eq!(rarest[0].0, "oracle-manipulation");
    }

    #[test]
    fn rarity_alone_does_not_win_when_the_project_cannot_host_the_mechanism() {
        let mut texts = vec!["reentrancy via an external call and callback".to_string()];
        texts.extend(vec!["oracle manipulation of the price feed".to_string(); 40]);
        let index = RarityIndex::from_texts(&texts);

        // The target has an oracle read but no external-call shape.
        let facts = facts_with("getPrice", "return IOracle(feed).latestAnswer();");
        let oracle = score("oracle manipulation of the price feed", &facts, &index);
        let reentrancy = score("reentrancy via an external call and callback", &facts, &index);

        assert_eq!(oracle.mechanism, Some("oracle-manipulation"));
        assert_eq!(oracle.precondition_match, 1.0, "the target reads an oracle");
        assert_eq!(
            reentrancy.precondition_match, 0.35,
            "the target has no external-call shape"
        );
        assert!(
            oracle.rank_key() > reentrancy.rank_key(),
            "a common mechanism this project can host must outrank a rare one it cannot: \
             {:?} vs {:?}",
            oracle.rank_key(),
            reentrancy.rank_key()
        );
        assert!(
            oracle.priority < reentrancy.priority,
            "and the raw score still says the other one is rarer: {} vs {}",
            oracle.priority,
            reentrancy.priority
        );
    }

    #[test]
    fn every_classified_mechanism_names_the_impacts_it_can_produce() {
        for (slug, _) in MECHANISMS {
            let impacts = impacts_of(slug);
            assert!(
                !impacts.is_empty(),
                "mechanism `{slug}` has no impact mapping; the impact-first gate would                  silently drop every link carrying it"
            );
            assert!(
                impacts.iter().all(|impact| impact.is_reportable()),
                "a mechanism mapping to a non-reportable impact is a contradiction: {slug}"
            );
        }
        assert!(impacts_of("something-not-in-the-table").is_empty());
    }

    #[test]
    fn an_unclassifiable_finding_is_neutral_not_buried() {
        let index = RarityIndex::from_texts(&vec!["some finding".to_string(); 10]);
        let facts = facts_with("deposit", "_mint(msg.sender, amount);");
        let scored = score("a finding about nothing in our table", &facts, &index);
        assert_eq!(scored.mechanism, None);
        assert!((scored.priority - 0.2).abs() < 1e-9, "{}", scored.priority);
    }

    #[test]
    fn vendored_code_does_not_satisfy_a_precondition() {
        let mut facts = facts_with("deposit", "balance += amount;");
        facts.functions.push(FunctionFacts {
            id: 2,
            name: "latestAnswer".to_string(),
            contract: Some("Vault".to_string()),
            path: "lib/oracle/src/Feed.sol".to_string(),
            line: 5,
            content: "return latestAnswer();".to_string(),
            writes: vec![],
            reads: vec![],
        });
        let index = RarityIndex::from_texts(&["oracle manipulation".to_string()]);
        let scored = score("oracle manipulation", &facts, &index);
        assert_eq!(
            scored.precondition_match, 0.35,
            "a library's oracle call is not this project's oracle call"
        );
    }
}
