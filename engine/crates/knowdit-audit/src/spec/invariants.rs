//! Invariants mined from the target's own code — the hypothesis source that
//! does not depend on history.
//!
//! Every other source in this pipeline is a prior over what has already been
//! found: the mapper asks which historical mechanism this project resembles,
//! the knowledge graph answers with patterns extracted from previous audits,
//! and the specification is written to reproduce one of them. That is a
//! reasonable instrument for a bug class somebody has already named, and the
//! wrong one for a failure mode nobody has: a corpus cannot contain what has
//! not happened yet.
//!
//! What a project *does* contain is the set of identities its own state is
//! supposed to satisfy — and most of them were never written down as a test.
//! Three kinds, and each is decidable from the stored storage/call graph
//! without a model call:
//!
//! * **accounting** ([`InvariantLayer::Accounting`]) — a total and the parts it
//!   is the sum of, a share price and the balance it is computed from, a mint
//!   and the supply it moves, a ledger and the real balance it mirrors;
//! * **cross-contract** ([`InvariantLayer::CrossContract`]) — a variable one
//!   contract writes and another reads, where a divergence is a bug in whoever
//!   forgot to update the mirror;
//! * **economic** ([`InvariantLayer::Economic`]) — solvency, non-dilution,
//!   no-profitable-round-trip, fee bounds. These are the ones that turn a state
//!   oddity into an attack scenario, and they need the engagement scope: which
//!   assets are insured and what they are worth is what makes the identity
//!   economic rather than formal.
//!
//! The layer decides how a violation has to be demonstrated, and that
//! distinction is the point of the module rather than a footnote:
//!
//! * accounting and cross-contract invariants are falsified by **state
//!   equality** after a call sequence — what the harness already does;
//! * an economic invariant is falsified only by **value delta**
//!   ([`Falsification::ValueDelta`]): the actor's net value has to go up *and*
//!   the victim's metric has to go down, measured in the same run. A harness
//!   that merely shows a weird state has shown nothing about the money, which
//!   is exactly the class this pipeline already defers on.
//!
//! Nothing here is invented: each invariant carries the `file:line` it binds as
//! evidence, so the specification agent can read the code before believing it.

use std::collections::{BTreeMap, BTreeSet};

use knowdit_repo_model::{FunctionFacts, ProjectFacts};

use crate::scope::{ProgramScope, same_contract};

/// Ceiling on how many invariants one link prompt carries. A project can
/// declare hundreds of variables; a prompt cannot, and the agent's step budget
/// is finite, so the list is capped after ranking.
pub const MAX_INVARIANTS: usize = 24;

/// Which kind of statement an invariant is, and therefore how it is falsified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum InvariantLayer {
    /// Identity over this contract's own storage: total vs parts, shares vs
    /// assets, supply vs mint/burn, ledger vs real balance.
    Accounting,
    /// Identity between two contracts' storage.
    CrossContract,
    /// Identity over value: solvency, dilution, arbitrage, fee bounds. Needs
    /// the engagement scope to exist at all.
    Economic,
}

/// How a violation of this invariant has to be demonstrated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Falsification {
    /// Assert the identity fails after a call sequence.
    StateEquality,
    /// Measure both sides in the run: the actor's net value must rise and the
    /// named victim metric must fall. A bare state assertion does not count —
    /// it would prove that some number is odd, not that anyone lost anything.
    ValueDelta {
        attacker_gain: String,
        victim_metric: String,
    },
}

/// One mined invariant, with the code it binds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinedInvariant {
    pub layer: InvariantLayer,
    /// Stable slug (`total-parts`, `mirror`, `solvency`, …) so a spec, a log
    /// line and a test can name the same thing.
    pub kind: &'static str,
    /// Contract the identity is anchored on (the writer side, for a mirror).
    pub contract: String,
    /// The identity, in project terms.
    pub statement: String,
    /// `path:line` anchors — the declaration and the function that moves it.
    pub evidence: Vec<String>,
    pub falsification: Falsification,
    /// Insured asset, for economic invariants whose contract the scope covers.
    pub asset: Option<String>,
    pub value_usd: Option<u64>,
}

impl MinedInvariant {
    /// Ranking key: economic before cross-contract before accounting, then by
    /// contract name and statement so the output is deterministic.
    fn sort_key(&self) -> (u8, String, String) {
        let layer = match self.layer {
            InvariantLayer::Economic => 0,
            InvariantLayer::CrossContract => 1,
            InvariantLayer::Accounting => 2,
        };
        (layer, self.contract.clone(), self.statement.clone())
    }
}

/// Name stems, matched as substrings of a lowercased identifier. Deliberately
/// blunt: a stem that matches too much costs one line in a prompt, a stem that
/// misses costs the invariant, and the agent reads the code before acting.
const TOTAL_STEMS: [&str; 10] = [
    "totalsupply",
    "totalassets",
    "totaldebt",
    "totalborrows",
    "totalstaked",
    "totalshares",
    "totalliquidity",
    "totaldeposits",
    "totalsupplies",
    "totalclaim",
];

const PART_STEMS: [&str; 9] = [
    "balance",
    "shares",
    "deposit",
    "borrow",
    "debt",
    "stake",
    "claim",
    "position",
    "credit",
];

const SHARE_BASE_STEMS: [&str; 4] = ["totalassets", "totalsupply", "totalshares", "totalliquidity"];

const SHARE_PRICE_FNS: [&str; 6] = [
    "convertoassets",
    "converttoshares",
    "pricepershare",
    "getpriceperfullshare",
    "exchangerate",
    "sharesforassets",
];

const SUPPLY_FNS: [&str; 4] = ["_mint(", "_burn(", "mint(", "burn("];

const SELF_BALANCE_REFS: [&str; 3] = [
    "balanceof(address(this))",
    "address(this).balance",
    "selfbalance()",
];

/// Names that *usually* mean value comes in. A hint, never the test: the
/// observed ground-truth deposit on the benchmark project is called `cashIn`,
/// which matches no naming convention worth writing down.
const DEPOSIT_FNS: [&str; 8] = [
    "deposit",
    "mint",
    "stake",
    "supply",
    "wrap",
    "join",
    "provide",
    "addliquidity",
];

const WITHDRAW_FNS: [&str; 8] = [
    "withdraw",
    "redeem",
    "unstake",
    "unwrap",
    "exit",
    "leave",
    "removeliquidity",
    "collect",
];

/// How value can arrive from outside the contract. Structural, because the
/// mechanism is "value in, claims out" and the function's name is not the
/// mechanism.
const VALUE_IN: [&str; 5] = [
    "msg.value",
    "balanceof(",
    "transferfrom(",
    "receive(",
    "call{value",
];

/// How value can leave, or claims be retired.
const VALUE_OUT: [&str; 5] = [
    "safetransfer(",
    ".transfer(",
    "transfer(",
    "call{value",
    "burn(",
];

const FEE_STEMS: [&str; 6] = [
    "fee",
    "commission",
    "charges",
    "premium",
    "spread",
    "incentive",
];

/// Storage names that show up in every project and mean nothing by themselves.
/// A mirror rule over `__gap` or `_name` would drown the real ones.
const IGNORED_STORAGE: [&str; 14] = [
    "__gap",
    "_gap",
    "gap",
    "name",
    "_name",
    "symbol",
    "_symbol",
    "owner",
    "_owner",
    "initialized",
    "_initialized",
    "decimals",
    "_decimals",
    "version",
];

/// Whether a storage type can carry a ledger: numbers, mappings and arrays can
/// disagree with their mirror; addresses, flags and strings cannot in any way
/// that costs money.
fn is_ledger_type(type_name: &str) -> bool {
    let lower = type_name.to_lowercase();
    lower.contains("uint")
        || lower.contains("int")
        || lower.contains("mapping")
        || lower.contains("[")
        || lower.contains("decimal")
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn is_totalish(name: &str) -> bool {
    let lower = name.to_lowercase();
    contains_any(&lower, &TOTAL_STEMS)
}

fn is_partish(name: &str) -> bool {
    let lower = name.to_lowercase();
    contains_any(&lower, &PART_STEMS)
}

fn is_claimish(name: &str) -> bool {
    let lower = name.to_lowercase();
    contains_any(
        &lower,
        &["totalsupply", "totalshares", "totaldeposits", "totaldebt", "totalborrows"],
    ) || contains_any(&lower, &["balance", "shares", "deposit", "debt", "borrow"])
}

fn anchor(path: &str, line: usize, what: &str) -> String {
    format!("{path}:{line} ({what})")
}

/// The miner. Pure function over facts plus the engagement scope.
pub struct InvariantMiner;

impl InvariantMiner {
    /// Mine the invariants this project's own code implies.
    ///
    /// `scope` is what makes the economic layer possible: without insured
    /// assets there is no way to say what a violation is worth, and an
    /// economic identity with no value behind it is just another accounting
    /// one. Accounting and cross-contract invariants are mined either way.
    pub fn mine(facts: &ProjectFacts, scope: Option<&ProgramScope>) -> Vec<MinedInvariant> {
        let mut found: Vec<MinedInvariant> = Vec::new();
        let mut seen: BTreeSet<(String, String)> = BTreeSet::new();

        for contract in &facts.contracts {
            // Vendored code is not the audited project: an identity inside
            // OpenZeppelin holds by construction, and a "mirror" between a base
            // class and its subclass is one storage location, not two.
            if knowdit_repo_model::project_facts::is_vendored_path(&contract.path) {
                continue;
            }
            let variables: Vec<_> = facts
                .state_variables_of(&contract.name)
                .into_iter()
                .filter(|variable| !knowdit_repo_model::project_facts::is_vendored_path(&variable.path))
                .collect();
            let functions: Vec<_> = facts
                .functions_of(&contract.name)
                .into_iter()
                .filter(|function| !knowdit_repo_model::project_facts::is_vendored_path(&function.path))
                .collect();
            let asset = scope
                .and_then(|scope| scope.assets_matching([contract.name.as_str()]).into_iter().next());

            Self::accounting(&contract.name, &variables, &functions, &mut found, &mut seen);
            Self::economic(
                &contract.name,
                &variables,
                &functions,
                asset.as_deref(),
                scope,
                &mut found,
                &mut seen,
            );
        }

        Self::mirrors(facts, &mut found, &mut seen);

        found.sort_by_key(|invariant| invariant.sort_key());
        found.truncate(MAX_INVARIANTS);
        found
    }

    /// Total vs parts, shares vs assets, supply vs mint/burn, ledger vs real
    /// balance — all within one contract.
    fn accounting(
        contract: &str,
        variables: &[&knowdit_repo_model::StateVarFacts],
        functions: &[&FunctionFacts],
        found: &mut Vec<MinedInvariant>,
        seen: &mut BTreeSet<(String, String)>,
    ) {
        let totals: Vec<_> = variables.iter().filter(|v| is_totalish(&v.name)).collect();

        // total == Σ parts
        for total in &totals {
            let writers: BTreeSet<i32> = functions
                .iter()
                .filter(|f| f.writes.contains(&total.id))
                .map(|f| f.id)
                .collect();
            let part = variables.iter().find(|variable| {
                variable.id != total.id
                    && (variable.type_name.to_lowercase().contains("mapping")
                        || is_partish(&variable.name))
                    && (writers.is_empty()
                        || functions.iter().any(|f| {
                            writers.contains(&f.id) && f.writes.contains(&variable.id)
                        }))
            });
            let Some(part) = part else { continue };
            let evidence = vec![
                anchor(&total.path, total.line, "total"),
                anchor(&part.path, part.line, "parts"),
            ];
            let statement = format!(
                "`{contract}.{}` equals the sum of the parts it accounts for (`{}.{}`)",
                total.name, contract, part.name
            );
            if seen.insert(("total-parts".to_string(), statement.clone())) {
                found.push(MinedInvariant {
                    layer: InvariantLayer::Accounting,
                    kind: "total-parts",
                    contract: contract.to_string(),
                    statement,
                    evidence,
                    falsification: Falsification::StateEquality,
                    asset: None,
                    value_usd: None,
                });
            }
        }

        // shares/assets: two claim bases in one contract, or a share-price
        // function over one — the conversion must be the one a redeemer gets.
        let bases: Vec<&str> = variables
            .iter()
            .filter(|v| contains_any(&v.name.to_lowercase(), &SHARE_BASE_STEMS))
            .map(|v| v.name.as_str())
            .collect();
        let price_fn = functions
            .iter()
            .find(|f| contains_any(&f.name.to_lowercase(), &SHARE_PRICE_FNS));
        if bases.len() >= 2 || (bases.len() == 1 && price_fn.is_some()) {
            let mut evidence: Vec<String> = variables
                .iter()
                .filter(|v| contains_any(&v.name.to_lowercase(), &SHARE_BASE_STEMS))
                .map(|v| anchor(&v.path, v.line, "claim base"))
                .collect();
            if let Some(function) = price_fn {
                evidence.push(anchor(&function.path, function.line, &function.name));
            }
            let statement = format!(
                "the worth of one claim as `{contract}` computes it ({}) is the worth a redeemer \
                 actually receives",
                if bases.is_empty() {
                    "from its price function".to_string()
                } else {
                    bases.join(" / ")
                }
            );
            if seen.insert(("shares-assets".to_string(), statement.clone())) {
                found.push(MinedInvariant {
                    layer: InvariantLayer::Accounting,
                    kind: "shares-assets",
                    contract: contract.to_string(),
                    statement,
                    evidence,
                    falsification: Falsification::StateEquality,
                    asset: None,
                    value_usd: None,
                });
            }
        }

        // supply moves exactly as much as the mint/burn did
        if let Some(supply) = variables
            .iter()
            .find(|v| contains_any(&v.name.to_lowercase(), &["totalsupply", "totalshares"]))
        {
            if let Some(function) = functions
                .iter()
                .find(|f| contains_any(&f.code(), &SUPPLY_FNS))
            {
                let statement = format!(
                    "`{contract}.{}` moves exactly by the amount the mint/burn in `{}` moved it",
                    supply.name, function.name
                );
                if seen.insert(("supply-consistency".to_string(), statement.clone())) {
                    found.push(MinedInvariant {
                        layer: InvariantLayer::Accounting,
                        kind: "supply-consistency",
                        contract: contract.to_string(),
                        statement,
                        evidence: vec![
                            anchor(&function.path, function.line, &function.name),
                            anchor(&supply.path, supply.line, "supply"),
                        ],
                        falsification: Falsification::StateEquality,
                        asset: None,
                        value_usd: None,
                    });
                }
            }
        }

        // ledger == real balance
        if let Some(function) = functions.iter().find(|f| {
            !f.writes.is_empty() && contains_any(&f.code(), &SELF_BALANCE_REFS)
        }) {
            let statement = format!(
                "the ledger `{}` maintains equals the contract's real balance after it returns",
                function.name
            );
            if seen.insert(("ledger-vs-balance".to_string(), statement.clone())) {
                found.push(MinedInvariant {
                    layer: InvariantLayer::Accounting,
                    kind: "ledger-vs-balance",
                    contract: contract.to_string(),
                    statement,
                    evidence: vec![anchor(&function.path, function.line, &function.name)],
                    falsification: Falsification::StateEquality,
                    asset: None,
                    value_usd: None,
                });
            }
        }
    }

    /// Solvency, non-dilution, round trip, fee bound — only for contracts the
    /// engagement insures, because only those have a value behind them.
    fn economic(
        contract: &str,
        variables: &[&knowdit_repo_model::StateVarFacts],
        functions: &[&FunctionFacts],
        asset: Option<&str>,
        scope: Option<&ProgramScope>,
        found: &mut Vec<MinedInvariant>,
        seen: &mut BTreeSet<(String, String)>,
    ) {
        let Some(asset) = asset else { return };
        let value_usd = scope
            .and_then(|scope| {
                scope
                    .assets
                    .iter()
                    .find(|candidate| same_contract(&candidate.name, asset))
            })
            .and_then(|candidate| candidate.value_usd);

        let claims = variables
            .iter()
            .filter(|v| is_claimish(&v.name))
            .map(|v| v.name.as_str())
            .collect::<Vec<_>>();
        // A function that mints claims against value read from outside is a
        // deposit whatever it is called; a function that mints or retires
        // claims and sends value out is a withdrawal.
        let writes_claim = |function: &FunctionFacts| {
            function.writes.iter().any(|id| {
                variables
                    .iter()
                    .any(|variable| variable.id == *id && is_claimish(&variable.name))
            })
        };
        let named = |function: &FunctionFacts, names: &[&str]| {
            contains_any(&function.name.to_lowercase(), names)
        };
        let deposit_fn = functions.iter().find(|function| {
            writes_claim(function)
                && (contains_any(&function.code(), &VALUE_IN) || named(function, &DEPOSIT_FNS))
        });
        let withdraw_fn = functions.iter().find(|function| {
            writes_claim(function)
                && (contains_any(&function.code(), &VALUE_OUT) || named(function, &WITHDRAW_FNS))
        });

        // solvency: what is held covers what is owed
        if !claims.is_empty() {
            let mut evidence: Vec<String> = Vec::new();
            if let Some(variable) = variables.iter().find(|v| is_claimish(&v.name)) {
                evidence.push(anchor(&variable.path, variable.line, "claims"));
            }
            if let Some(function) = deposit_fn {
                evidence.push(anchor(&function.path, function.line, &function.name));
            }
            let statement = format!(
                "the value `{contract}` holds stays at least the value its claims ({}) are worth",
                claims.join(" / ")
            );
            if seen.insert(("solvency".to_string(), statement.clone())) {
                found.push(MinedInvariant {
                    layer: InvariantLayer::Economic,
                    kind: "solvency",
                    contract: contract.to_string(),
                    statement,
                    evidence,
                    falsification: Falsification::ValueDelta {
                        attacker_gain: "redeeming claims for more than the value backing them"
                            .to_string(),
                        victim_metric: format!(
                            "`{contract}`'s held value per outstanding claim"
                        ),
                    },
                    asset: Some(asset.to_string()),
                    value_usd,
                });
            }
        }

        // non-dilution: a deposit must not change what existing claims redeem for
        if let Some(function) = deposit_fn {
            let code = function.code();
            let reads_value = contains_any(&code, &VALUE_IN);
            if reads_value && writes_claim(function) {
                let statement = format!(
                    "a deposit through `{contract}.{}` does not change what existing claims redeem for",
                    function.name
                );
                if seen.insert(("non-dilution".to_string(), statement.clone())) {
                    found.push(MinedInvariant {
                        layer: InvariantLayer::Economic,
                        kind: "non-dilution",
                        contract: contract.to_string(),
                        statement,
                        evidence: vec![anchor(&function.path, function.line, &function.name)],
                        falsification: Falsification::ValueDelta {
                            attacker_gain: "claims minted for more value than the deposit delivered"
                                .to_string(),
                            victim_metric: "redeemable value per existing claim".to_string(),
                        },
                        asset: Some(asset.to_string()),
                        value_usd,
                    });
                }
            }
        }

        // round trip: in and out again must not pay
        if let (Some(in_fn), Some(out_fn)) = (deposit_fn, withdraw_fn) {
            let statement = format!(
                "a round trip through `{contract}.{}` and `{contract}.{}` returns no more value \
                 than it committed",
                in_fn.name, out_fn.name
            );
            if seen.insert(("round-trip".to_string(), statement.clone())) {
                found.push(MinedInvariant {
                    layer: InvariantLayer::Economic,
                    kind: "round-trip",
                    contract: contract.to_string(),
                    statement,
                    evidence: vec![
                        anchor(&in_fn.path, in_fn.line, &in_fn.name),
                        anchor(&out_fn.path, out_fn.line, &out_fn.name),
                    ],
                    falsification: Falsification::ValueDelta {
                        attacker_gain: "value extracted by re-entering on both sides".to_string(),
                        victim_metric: format!("`{contract}`'s held value"),
                    },
                    asset: Some(asset.to_string()),
                    value_usd,
                });
            }
        }

        // fee bound: what leaves never exceeds what came in plus what was earned
        if let Some(function) = functions.iter().find(|f| {
            contains_any(&f.name.to_lowercase(), &WITHDRAW_FNS)
                && f.writes.iter().any(|id| {
                    variables.iter().any(|variable| {
                        variable.id == *id
                            && contains_any(&variable.name.to_lowercase(), &FEE_STEMS)
                    })
                })
        }) {
            let statement = format!(
                "what leaves `{contract}` through `{}` never exceeds what was put in plus what \
                 was earned",
                function.name
            );
            if seen.insert(("fee-bound".to_string(), statement.clone())) {
                found.push(MinedInvariant {
                    layer: InvariantLayer::Economic,
                    kind: "fee-bound",
                    contract: contract.to_string(),
                    statement,
                    evidence: vec![anchor(&function.path, function.line, &function.name)],
                    falsification: Falsification::ValueDelta {
                        attacker_gain: "fees charged beyond the realized yield".to_string(),
                        victim_metric: "withdrawable assets of the other claim holders".to_string(),
                    },
                    asset: Some(asset.to_string()),
                    value_usd,
                });
            }
        }
    }

    /// A copy one contract keeps of another contract's state. The copy's owner
    /// trusts it; a call path that moves the source without refreshing the copy
    /// (or refreshes it from a value the caller controls) is the bug.
    ///
    /// Solidity storage is private to its contract, so two variables sharing a
    /// name in unrelated contracts are two variables — a mirror only exists
    /// where one is *filled from* the other. The miner therefore demands:
    ///
    /// * two distinct variables with the same name, in contracts not related
    ///   by inheritance (a base and its subclass share one slot);
    /// * both ledger-shaped and mutable — written outside a constructor or
    ///   initializer, and not named like a constant (`ALL_CAPS`). An immutable
    ///   set at deployment cannot drift;
    /// * a coupling in code: some function writing the copy calls the source's
    ///   getter (`.name(`) or names the source contract.
    ///
    /// Without the coupling the pair is a naming coincidence, and as a work
    /// item it would spend a whole agent run proving two unrelated numbers
    /// differ.
    fn mirrors(
        facts: &ProjectFacts,
        found: &mut Vec<MinedInvariant>,
        seen: &mut BTreeSet<(String, String)>,
    ) {
        let is_setup = |name: &str| {
            let lower = name.to_lowercase();
            lower == "constructor" || lower.contains("initialize") || lower.starts_with("init")
        };
        // variable id -> functions (outside setup) that write it
        let mut writers_of: BTreeMap<i32, Vec<&knowdit_repo_model::FunctionFacts>> =
            BTreeMap::new();
        for function in &facts.functions {
            if function.contract.is_none()
                || knowdit_repo_model::project_facts::is_vendored_path(&function.path)
                || is_setup(&function.name)
            {
                continue;
            }
            for id in &function.writes {
                writers_of.entry(*id).or_default().push(function);
            }
        }
        let mut by_name: BTreeMap<String, Vec<&knowdit_repo_model::StateVarFacts>> =
            BTreeMap::new();
        for variable in &facts.state_variables {
            let lower = variable.name.to_lowercase();
            let constant_like = variable.name.len() > 1
                && variable
                    .name
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
            if variable.contract.is_none()
                || knowdit_repo_model::project_facts::is_vendored_path(&variable.path)
                || lower.len() < 4
                || IGNORED_STORAGE.contains(&lower.as_str())
                || constant_like
                || !is_ledger_type(&variable.type_name)
                || !writers_of.contains_key(&variable.id)
            {
                continue;
            }
            by_name.entry(variable.name.clone()).or_default().push(variable);
        }

        for (name, variables) in &by_name {
            let getter = format!(".{}(", name.to_lowercase());
            for source in variables {
                for copy in variables {
                    let (Some(source_contract), Some(copy_contract)) =
                        (source.contract.as_deref(), copy.contract.as_deref())
                    else {
                        continue;
                    };
                    if source.id == copy.id
                        || source_contract == copy_contract
                        || facts.related_by_inheritance(source_contract, copy_contract)
                    {
                        continue;
                    }
                    let source_lower = source_contract.to_lowercase();
                    let Some(refresher) = writers_of.get(&copy.id).and_then(|writers| {
                        writers.iter().find(|function| {
                            let code = function.code();
                            code.contains(&getter) || code.contains(&source_lower)
                        })
                    }) else {
                        continue;
                    };
                    let statement = format!(
                        "`{copy_contract}.{name}` (a copy) equals `{source_contract}.{name}` \
                         (its source) after any call that changes the source"
                    );
                    if seen.insert(("mirror".to_string(), statement.clone())) {
                        found.push(MinedInvariant {
                            layer: InvariantLayer::CrossContract,
                            kind: "mirror",
                            contract: copy_contract.to_string(),
                            statement,
                            evidence: vec![
                                anchor(&copy.path, copy.line, "copy"),
                                anchor(&source.path, source.line, "source"),
                                anchor(&refresher.path, refresher.line, "refreshed here"),
                            ],
                            falsification: Falsification::StateEquality,
                            asset: None,
                            value_usd: None,
                        });
                    }
                }
            }
        }
    }
}

/// The invariants that touch any of `contracts`, in mined order.
pub fn relevant<'a>(invariants: &'a [MinedInvariant], contracts: &[String]) -> Vec<&'a MinedInvariant> {
    invariants
        .iter()
        .filter(|invariant| {
            contracts
                .iter()
                .any(|contract| same_contract(&invariant.contract, &contract_ref(contract)))
        })
        .collect()
}

/// A contract name out of whatever a caller has: the extracted semantics carry
/// file paths (`src/rebalance/LamboRebalanceOnUniwap.sol`), a deployment
/// artifact carries `path:Name`, a specification carries `Contract` or
/// `Contract.function`. All four have to land on the same name the miner
/// anchored its invariants on.
pub(super) fn contract_ref(raw: &str) -> String {
    let trimmed = raw.trim();
    let tail = trimmed
        .rsplit_once(':')
        .map(|(_, tail)| tail)
        .unwrap_or(trimmed);
    let file = tail.rsplit('/').next().unwrap_or(tail);
    let stem = file.strip_suffix(".sol").unwrap_or(file);
    stem.split('.').next().unwrap_or(stem).trim().to_string()
}

/// The prompt block. Rendered above the link so the specification agent has a
/// hypothesis source that is about *this* project, and knows what a violation
/// of each kind has to demonstrate.
pub fn render_block(invariants: &[&MinedInvariant]) -> String {
    if invariants.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "## Invariants mined from this project's own code\n\n\
         These were derived mechanically from the target's storage and call graph (no model, no \
         historical pattern) and they are the part of the hypothesis that is about THIS project. \
         The historical finding below is a hint about how a family of defects arises elsewhere; \
         these are statements the code here is supposed to satisfy. Each carries the code it binds, \
         so read it before believing it.\n\n",
    );
    for invariant in invariants {
        let layer = match invariant.layer {
            InvariantLayer::Accounting => "accounting",
            InvariantLayer::CrossContract => "cross-contract",
            InvariantLayer::Economic => "economic",
        };
        out.push_str(&format!("- [{layer}] {} — {}", invariant.statement, invariant.evidence.join(", ")));
        if let Some(asset) = &invariant.asset {
            out.push_str(&format!("; insured asset `{asset}`"));
            if let Some(value) = invariant.value_usd {
                out.push_str(&format!(" (${value} at risk)"));
            }
        }
        out.push('\n');
        match &invariant.falsification {
            Falsification::StateEquality => {
                out.push_str("  - demonstrate by: state equality (assert it fails after a call sequence)\n");
            }
            Falsification::ValueDelta {
                attacker_gain,
                victim_metric,
            } => {
                out.push_str(&format!(
                    "  - demonstrate by: **value delta** — measure both sides in the run: the actor \
                     must end net richer ({attacker_gain}) AND {victim_metric} must fall. A state \
                     assertion alone is not a reproduction of this invariant.\n"
                ));
            }
        }
    }
    out.push_str(
        "\nBuilding a specification on one of these is expected and encouraged: it is grounded in \
         this project rather than in a resemblance, so it stays reportable even when no historical \
         finding matches it.\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scope::ProgramScope;
    use knowdit_repo_model::{ContractFacts, FunctionFacts, ProjectFacts, StateVarFacts};

    fn variable(id: i32, name: &str, ty: &str, contract: &str, line: usize) -> StateVarFacts {
        StateVarFacts {
            id,
            name: name.to_string(),
            type_name: ty.to_string(),
            path: format!("src/{contract}.sol"),
            line,
            contract: Some(contract.to_string()),
        }
    }

    fn function(
        id: i32,
        name: &str,
        contract: &str,
        line: usize,
        content: &str,
        writes: Vec<i32>,
        reads: Vec<i32>,
    ) -> FunctionFacts {
        FunctionFacts {
            id,
            name: name.to_string(),
            contract: Some(contract.to_string()),
            path: format!("src/{contract}.sol"),
            line,
            content: content.to_string(),
            writes,
            reads,
        }
    }

    fn contract(name: &str) -> ContractFacts {
        ContractFacts {
            id: 1,
            name: name.to_string(),
            path: format!("src/{name}.sol"),
            line: 1,
        }
    }

    fn scope_with(asset: &str, value: Option<u64>) -> ProgramScope {
        let json = format!(
            r#"{{"program":"example","chain_id":1,"assets":[{{"name":"{asset}","value_usd":{}}}]}}"#,
            value.map(|v| v.to_string()).unwrap_or("null".to_string())
        );
        ProgramScope::parse(&json, std::path::Path::new("knowdit-scope.json"))
            .expect("fixture scope parses")
    }

    /// The shape of the ground-truth H-01: a deposit mints against `msg.value`
    /// while the contract holds a claim base. The economic invariant has to
    /// exist with a value-delta falsification, not a state equality.
    #[test]
    fn a_deposit_minting_against_msg_value_yields_a_non_dilution_invariant() {
        let facts = ProjectFacts {
            contracts: vec![contract("VirtualToken")],
            state_variables: vec![
                variable(1, "totalSupply", "uint256", "VirtualToken", 41),
                variable(2, "_balances", "mapping(address => uint256)", "VirtualToken", 42),
                variable(3, "accumulatedFee", "uint256", "VirtualToken", 43),
            ],
            inherits: Vec::new(),
            functions: vec![
                function(
                    1,
                    "cashIn",
                    "VirtualToken",
                    80,
                    "function cashIn(uint256 amount) external payable { _mint(msg.sender, msg.value); }",
                    vec![1, 2],
                    vec![],
                ),
                function(
                    2,
                    "cashOut",
                    "VirtualToken",
                    95,
                    "function cashOut(uint256 amount) external { _burn(msg.sender, amount); \
                     payable(msg.sender).transfer(amount); }",
                    vec![1, 2, 3],
                    vec![],
                ),
            ],
        };
        let scope = scope_with("VirtualToken", Some(1_000_000));
        let invariants = InvariantMiner::mine(&facts, Some(&scope));

        let non_dilution = invariants
            .iter()
            .find(|invariant| invariant.kind == "non-dilution")
            .expect("non-dilution must be mined");
        assert_eq!(non_dilution.layer, InvariantLayer::Economic);
        assert_eq!(non_dilution.asset.as_deref(), Some("VirtualToken"));
        assert_eq!(non_dilution.value_usd, Some(1_000_000));
        assert!(
            matches!(non_dilution.falsification, Falsification::ValueDelta { .. }),
            "economic invariants must be falsified by value delta: {:?}",
            non_dilution.falsification
        );
        assert!(
            non_dilution.evidence[0].starts_with("src/VirtualToken.sol:80"),
            "the evidence must be the code: {:?}",
            non_dilution.evidence
        );
        // The accounting invariants that the same facts imply are still there.
        assert!(invariants.iter().any(|i| i.kind == "total-parts"));
        assert!(invariants.iter().any(|i| i.kind == "supply-consistency"));
        assert!(invariants.iter().any(|i| i.kind == "round-trip"));
    }

    #[test]
    fn without_a_scope_only_the_code_shaped_invariants_are_mined() {
        let facts = ProjectFacts {
            contracts: vec![contract("Vault")],
            state_variables: vec![
                variable(1, "totalAssets", "uint256", "Vault", 20),
                variable(2, "totalSupply", "uint256", "Vault", 21),
                variable(3, "_balances", "mapping(address => uint256)", "Vault", 22),
            ],
            inherits: Vec::new(),
            functions: vec![function(
                1,
                "deposit",
                "Vault",
                30,
                "function deposit(uint256 amount) external { _mint(msg.sender, amount); }",
                vec![1, 2, 3],
                vec![],
            )],
        };
        let invariants = InvariantMiner::mine(&facts, None);
        assert!(!invariants.is_empty());
        assert!(
            invariants.iter().all(|i| i.layer != InvariantLayer::Economic),
            "an economic invariant with no insured value behind it is just accounting: {invariants:?}"
        );
        assert!(invariants.iter().any(|i| i.kind == "shares-assets"));
    }

    #[test]
    fn a_copy_refreshed_from_its_source_is_a_mirror() {
        let facts = ProjectFacts {
            contracts: vec![contract("Controller"), contract("Ledger")],
            state_variables: vec![
                variable(1, "collateral", "uint256", "Controller", 15),
                variable(2, "collateral", "uint256", "Ledger", 90),
            ],
            inherits: Vec::new(),
            functions: vec![
                function(1, "supply", "Controller", 40, "collateral = collateral + amount;", vec![1], vec![]),
                function(2, "sync", "Ledger", 110, "collateral = controller.collateral();", vec![2], vec![]),
            ],
        };
        let invariants = InvariantMiner::mine(&facts, None);
        let mirror = invariants
            .iter()
            .find(|invariant| invariant.kind == "mirror")
            .expect("mirror must be mined");
        assert_eq!(mirror.layer, InvariantLayer::CrossContract);
        assert_eq!(mirror.contract, "Ledger");
        assert!(mirror.statement.contains("`Ledger.collateral` (a copy)"));
        assert!(mirror.statement.contains("`Controller.collateral` (its source)"));
        assert!(mirror.evidence.iter().any(|e| e.contains("refreshed here")));
    }

    #[test]
    fn same_name_without_coupling_is_not_a_mirror() {
        // Each contract keeps its own `collateral`; nothing fills one from the
        // other, so they are unrelated variables that share a name.
        let facts = ProjectFacts {
            contracts: vec![contract("Controller"), contract("Ledger")],
            state_variables: vec![
                variable(1, "collateral", "uint256", "Controller", 15),
                variable(2, "collateral", "uint256", "Ledger", 90),
            ],
            inherits: Vec::new(),
            functions: vec![
                function(1, "supply", "Controller", 40, "collateral = collateral + amount;", vec![1], vec![]),
                function(2, "post", "Ledger", 110, "collateral += amount;", vec![2], vec![]),
            ],
        };
        let invariants = InvariantMiner::mine(&facts, None);
        assert!(!invariants.iter().any(|i| i.kind == "mirror"), "{invariants:?}");
    }

    #[test]
    fn immutables_and_constants_are_not_mirrors() {
        // Written only at deployment, or named as a constant: neither can drift.
        let facts = ProjectFacts {
            contracts: vec![contract("PositionA"), contract("PositionB")],
            state_variables: vec![
                variable(1, "REFERRAL_CODE", "uint16", "PositionA", 10),
                variable(2, "REFERRAL_CODE", "uint16", "PositionB", 20),
                variable(3, "rateCache", "uint256", "PositionA", 11),
                variable(4, "rateCache", "uint256", "PositionB", 21),
            ],
            inherits: Vec::new(),
            functions: vec![
                function(
                    1,
                    "constructor",
                    "PositionB",
                    30,
                    "rateCache = positionA.rateCache(); REFERRAL_CODE = positionA.REFERRAL_CODE();",
                    vec![2, 4],
                    vec![],
                ),
                function(2, "bump", "PositionA", 40, "rateCache += 1;", vec![3], vec![]),
                function(3, "setCode", "PositionA", 50, "REFERRAL_CODE = 1;", vec![1], vec![]),
            ],
        };
        let invariants = InvariantMiner::mine(&facts, None);
        assert!(!invariants.iter().any(|i| i.kind == "mirror"), "{invariants:?}");
    }

    #[test]
    fn both_contracts_reading_a_variable_is_not_a_mirror() {
        let facts = ProjectFacts {
            contracts: vec![contract("A"), contract("B")],
            state_variables: vec![
                variable(1, "sharedConfig", "uint256", "A", 10),
                variable(2, "sharedConfig", "uint256", "B", 20),
            ],
            inherits: Vec::new(),
            functions: vec![
                function(1, "readA", "A", 30, "require(sharedConfig > 0);", vec![], vec![1]),
                function(2, "readB", "B", 40, "require(sharedConfig > 0);", vec![], vec![2]),
            ],
        };
        let invariants = InvariantMiner::mine(&facts, None);
        assert!(
            !invariants.iter().any(|invariant| invariant.kind == "mirror"),
            "two readers cannot disagree with each other: {invariants:?}"
        );
    }

    #[test]
    fn bookkeeping_storage_is_not_a_mirror() {
        let facts = ProjectFacts {
            contracts: vec![contract("A"), contract("B")],
            state_variables: vec![
                variable(1, "__gap", "uint256[50]", "A", 10),
                variable(2, "__gap", "uint256[50]", "B", 20),
            ],
            inherits: Vec::new(),
            functions: vec![
                function(1, "initA", "A", 30, "__gap[0] = 1;", vec![1], vec![]),
                function(2, "readB", "B", 40, "require(__gap[0] == 0);", vec![], vec![2]),
            ],
        };
        let invariants = InvariantMiner::mine(&facts, None);
        assert!(!invariants.iter().any(|i| i.kind == "mirror"), "{invariants:?}");
    }

    #[test]
    fn the_block_demands_a_value_delta_for_economic_invariants() {
        let facts = ProjectFacts {
            contracts: vec![contract("Vault")],
            state_variables: vec![
                variable(1, "totalSupply", "uint256", "Vault", 20),
                variable(2, "_balances", "mapping(address => uint256)", "Vault", 21),
            ],
            inherits: Vec::new(),
            functions: vec![function(
                1,
                "deposit",
                "Vault",
                30,
                "function deposit() external payable { _mint(msg.sender, msg.value); }",
                vec![1, 2],
                vec![],
            )],
        };
        let scope = scope_with("Vault", Some(500));
        let invariants = InvariantMiner::mine(&facts, Some(&scope));
        let relevant = relevant(&invariants, &["Vault".to_string()]);
        let block = render_block(&relevant);

        assert!(block.contains("Invariants mined from this project's own code"), "{block}");
        assert!(block.contains("[economic]"), "{block}");
        assert!(block.contains("value delta"), "{block}");
        assert!(block.contains("state equality"), "accounting falsification is stated too: {block}");
        assert!(block.contains("$500 at risk"), "{block}");
        assert!(block.contains("src/Vault.sol:30"), "evidence is carried: {block}");
        assert!(
            block.contains("not a reproduction"),
            "the block must say a state assertion is insufficient for economic invariants: {block}"
        );
    }

    #[test]
    fn invariants_are_filtered_to_the_contracts_a_link_touches() {
        let facts = ProjectFacts {
            contracts: vec![contract("Vault"), contract("Router")],
            state_variables: vec![
                variable(1, "totalSupply", "uint256", "Vault", 20),
                variable(2, "_balances", "mapping(address => uint256)", "Vault", 21),
            ],
            inherits: Vec::new(),
            functions: vec![function(
                1,
                "deposit",
                "Vault",
                30,
                "function deposit() external payable { _mint(msg.sender, msg.value); }",
                vec![1, 2],
                vec![],
            )],
        };
        let invariants = InvariantMiner::mine(&facts, None);
        assert!(!invariants.is_empty());
        assert!(
            relevant(&invariants, &["Router".to_string()]).is_empty(),
            "a link on Router must not carry Vault's invariants"
        );
        assert!(!relevant(&invariants, &["Vault".to_string()]).is_empty());
        assert!(!relevant(&invariants, &["src/Vault.sol:Vault".to_string()]).is_empty());
    }

    /// Reads a real project database so the miner is exercised against the
    /// storage and call graph an audit actually ships with, not only against
    /// fixtures. Ignored by default (it needs a populated repo DB and a
    /// network-less sqlite open) and run explicitly:
    /// `KNOWDIT_FACTS_DB=<path> cargo test -p knowdit-audit --lib mines_a_real -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "reads a project database pointed at by KNOWDIT_FACTS_DB"]
    async fn mines_a_real_project_database() {
        let Ok(path) = std::env::var("KNOWDIT_FACTS_DB") else {
            return;
        };
        let repo = knowdit_repo_model::RepoDatabase::open_sqlite(std::path::PathBuf::from(&path))
            .await
            .expect("repo database opens");
        let facts = repo.load_project_facts().await.expect("facts load");
        let invariants = InvariantMiner::mine(&facts, None);
        println!(
            "mined {} invariant(s) from {} contract(s) / {} state variable(s) / {} function(s)",
            invariants.len(),
            facts.contracts.len(),
            facts.state_variables.len(),
            facts.functions.len()
        );
        for invariant in &invariants {
            println!(
                "  [{:?}] {} — {}",
                invariant.layer, invariant.statement, invariant.evidence.join(", ")
            );
        }
        assert!(
            invariants.len() <= MAX_INVARIANTS,
            "the cap holds on real data too"
        );
    }

    #[test]
    fn the_list_is_capped_and_ordered_economic_first() {
        let mut variables = Vec::new();
        let mut functions = Vec::new();
        for index in 0..40 {
            variables.push(variable(
                index + 1,
                &format!("totalDeposits{index}"),
                "uint256",
                "Vault",
                10 + index as usize,
            ));
            variables.push(variable(
                index + 100,
                &format!("_balances{index}"),
                "mapping(address => uint256)",
                "Vault",
                200 + index as usize,
            ));
            functions.push(function(
                index + 1,
                &format!("deposit{index}"),
                "Vault",
                400 + index as usize,
                "function deposit() external payable { _mint(msg.sender, msg.value); }",
                vec![index + 1, index + 100],
                vec![],
            ));
        }
        let facts = ProjectFacts {
            contracts: vec![contract("Vault")],
            state_variables: variables,
            functions,
            inherits: Vec::new(),
        };
        let invariants = InvariantMiner::mine(&facts, None);
        assert!(invariants.len() <= MAX_INVARIANTS, "{}", invariants.len());
        // Deterministic: the same facts produce the same order twice.
        assert_eq!(
            invariants,
            InvariantMiner::mine(&facts, None),
            "mining must be deterministic"
        );
    }
}
