use std::collections::BTreeMap;

use knowdit_kg_model::{ExtractedFinding, ExtractedSemantic};
use knowdit_repo_model::{cg::CallGraph, storage::StorageGraph};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// What an exploit takes away, stated before the run spends anything on it.
///
/// The vocabulary is the one bounty programs publish in their own words: a
/// program pays for a loss it recognises, and a specification that cannot name
/// one is not a candidate no matter how clean its proof is. Declaring the class
/// at specification time moves that judgement ahead of the expensive stages —
/// the harness agent, the PoC, the verdict pass — instead of leaving it to
/// whoever reads the report at the end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ImpactClass {
    /// Funds leave the protocol or a user for an address the attacker controls.
    DirectTheftOfFunds,
    /// Funds become unrecoverable, with no path back.
    PermanentFreeze,
    /// Funds are unavailable for a period and recoverable afterwards.
    TemporaryFreeze,
    /// The protocol owes more than it holds and cannot cover withdrawals.
    ProtocolInsolvency,
    /// A core operation cannot be completed by its legitimate callers.
    ProtocolDos,
    /// Damage to users or the protocol with no profit motive: blocked flows,
    /// wasted funds, degraded service.
    Griefing,
    /// A real behavioural divergence with nothing of value at stake. No
    /// program pays for it, so it is never reportable.
    NoValueAtRisk,
}

impl ImpactClass {
    /// Every class, in declaration order — the order a program's impact table
    /// reads in, and the order the prompts and reports should preserve.
    pub const ALL: [ImpactClass; 7] = [
        Self::DirectTheftOfFunds,
        Self::PermanentFreeze,
        Self::TemporaryFreeze,
        Self::ProtocolInsolvency,
        Self::ProtocolDos,
        Self::Griefing,
        Self::NoValueAtRisk,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::DirectTheftOfFunds => "direct_theft_of_funds",
            Self::PermanentFreeze => "permanent_freeze",
            Self::TemporaryFreeze => "temporary_freeze",
            Self::ProtocolInsolvency => "protocol_insolvency",
            Self::ProtocolDos => "protocol_dos",
            Self::Griefing => "griefing",
            Self::NoValueAtRisk => "no_value_at_risk",
        }
    }

    /// Whether a program that pays for losses could pay for this one. The only
    /// mechanically decidable part of "does this qualify", and the part that
    /// catches a whole specification before it costs a harness run.
    pub fn is_reportable(self) -> bool {
        self != Self::NoValueAtRisk
    }
}

/// Who carries out the exploit, stated before the run spends anything on it.
///
/// Programs publish this as a rule — the standing one is that an attacker is an
/// unprivileged participant — and it decides findings that are otherwise
/// identical: the same defect reached by a holder is payable, reached by the
/// protocol's own guardian is not. It is a property of the specification, not
/// of the code, so it has to be declared where the specification is written.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AttackerPrivilege {
    /// No role, permission or special position is required: any participant can
    /// drive the sequence. The only model most programs pay for.
    #[default]
    Unprivileged,
    /// The exploit needs an account the protocol has already granted authority
    /// (owner, guardian, admin, operator, a whitelisted market…) and needs no
    /// more than that authority.
    Privileged,
    /// An unprivileged caller acquires authority the protocol never granted
    /// them. Payable wherever an unprivileged actor is, because the escalation
    /// is the finding.
    PrivilegeEscalation,
}

impl AttackerPrivilege {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unprivileged => "unprivileged",
            Self::Privileged => "privileged",
            Self::PrivilegeEscalation => "privilege_escalation",
        }
    }
}

/// The actor one specification's sequence assumes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AttackerModel {
    pub privilege: AttackerPrivilege,
    /// Which authority, when `privilege` is `Privileged` (e.g. `GUARDIAN_ROLE`,
    /// `owner()`). Ignored otherwise.
    #[serde(default)]
    pub role: Option<String>,
    /// One sentence on why the actor model holds — how the caller comes to be in
    /// that position. Read by the review agent, never asserted as proof.
    #[serde(default)]
    pub note: String,
}

#[derive(Debug, Clone)]
pub enum StateGraph {
    Solidity(StorageGraph),
}

/// Fetch from repo database
#[derive(Debug, Clone)]
pub struct RepoContext {
    pub cg: CallGraph,
    pub state_graph: StorageGraph,
    pub semantics: Vec<ExtractedSemantic>,
}

#[derive(Debug, Clone)]
pub struct VulnLinkCandidate {
    pub semantic: ExtractedSemantic,
    pub findings: ExtractedFinding,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditStateSpecification {
    /// General description of the state, this might contain multiple rules!
    pub description: String,
    /// The specification requirements for state variables, i.e., the reseves should be drained after being attacked
    pub states_variables: BTreeMap<String, String>,
    /// The specification requirements for contracts, i.e., contract Manager must be present and an admin should be added
    pub contracts: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuqenceCallStep {
    /// The contract of the call
    pub contract: String,
    /// The function name of the call
    pub function: String,
    /// The intention of the call
    pub intention: String,
}

/// One state variable declared on an [`AuxiliaryContract`]. Auxiliary
/// contracts are synthesized by the PoC harness, so unlike project state
/// variables these are defined by the agent rather than grounded against
/// the project source.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuxStateVariable {
    /// Solidity type of the variable, e.g. `uint256`, `address`.
    pub solidity_type: String,
    /// What this variable is for in the attack scenario.
    pub purpose: String,
}

/// A helper contract the PoC must deploy itself — an attacker contract,
/// malicious callback, mock token, etc. — that does not exist in the
/// project source. Declared explicitly by the agent so spec references to
/// it (and to its state variables / functions) are grounded against this
/// registry instead of the project, and so the harness knows to
/// synthesize it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuxiliaryContract {
    /// Why this auxiliary contract exists in the PoC, e.g. "reentrancy attacker".
    pub purpose: String,
    /// State variables the auxiliary contract declares: name -> metadata.
    pub state_variables: BTreeMap<String, AuxStateVariable>,
    /// The external protocol this contract impersonates, when it is a
    /// stand-in rather than an attacker-side helper — an address, or a name
    /// the deployment book resolves.
    #[serde(default)]
    pub stands_in_for: Option<String>,
    /// How the stand-in's behaviour was checked against the real deployment.
    #[serde(default)]
    pub fidelity: Option<String>,
}

/// What kind of functional claim a [`SpecProperty`] makes.
///
/// The four shapes a *computation* defect can take. They exist because a
/// state-shaped specification cannot describe them: "the balances are drained"
/// is not what is wrong when a formula assumes the wrong curve, and asking for
/// a state invariant makes the agent report the symptom it can reach (a revert,
/// an overflow) instead of the miscomputation that matters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PropertyKind {
    #[default]
    /// A quoted computation must agree with what actually executing the
    /// described path produces — the preview vs the swap, the estimate vs the
    /// transfer.
    AgreesWithActual,
    /// For the stated inputs, a branch or selector outcome must be the named
    /// one (`directionMask` must select SELL here).
    BranchExpected,
    /// The result must be monotone in the stated input (more in, not less out).
    Monotone,
    /// Converting out and back must not lose more than the stated bound.
    RoundTrip,
}

impl PropertyKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AgreesWithActual => "agrees_with_actual",
            Self::BranchExpected => "branch_expected",
            Self::Monotone => "monotone",
            Self::RoundTrip => "round_trip",
        }
    }
}

/// One claim about a computation, instead of about state.
///
/// `claim` is written in project terms and states what must hold — it is what
/// the harness turns into an assertion and what a reviewer reads when the
/// violation lands.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SpecProperty {
    pub kind: PropertyKind,
    /// The project function the claim is about (`Contract.function`), grounded
    /// against the project at the tool call.
    pub target: String,
    /// The claim in one sentence: what must hold, for which inputs.
    pub claim: String,
}

/// Which way value moves for one party to an exploit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ValueDirection {
    /// Ends the sequence holding more value than it started with.
    Gains,
    /// Ends the sequence holding less value.
    Loses,
    /// Keeps its value but cannot move it (a freeze, a blocked withdrawal).
    Frozen,
    /// Nothing moves for this party.
    Unchanged,
}

impl ValueDirection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gains => "gains",
            Self::Loses => "loses",
            Self::Frozen => "frozen",
            Self::Unchanged => "unchanged",
        }
    }

    /// Whether this party is harmed: it lost value or lost access to it.
    pub fn is_harm(self) -> bool {
        matches!(self, Self::Loses | Self::Frozen)
    }
}

/// Which party a value-flow row is about. The attacker row is singled out
/// because the direction of every claim is read against it: an exploit whose
/// only loser is the caller is a donation, not a finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ActorRole {
    /// Whoever drives the sequence (the caller and any contract it deploys).
    Attacker,
    /// A specific user or position the exploit targets.
    Victim,
    /// The rest of the holders / depositors / LPs, collectively.
    OtherHolders,
    /// The protocol's own treasury, reserves or vault.
    Protocol,
    /// Whoever the protocol routes fees to.
    FeeRecipient,
    /// Anyone else the sequence touches (an external pool, a keeper).
    ThirdParty,
}

impl ActorRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Attacker => "attacker",
            Self::Victim => "victim",
            Self::OtherHolders => "other_holders",
            Self::Protocol => "protocol",
            Self::FeeRecipient => "fee_recipient",
            Self::ThirdParty => "third_party",
        }
    }
}

/// One row of an exploit's value-flow table: who ends up where, in what, and
/// through which mechanism.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ActorDelta {
    pub role: ActorRole,
    /// The concrete party, in project terms (`the redeeming shareholder`,
    /// `VaultProxy`, `fee recipient of the ManagementFee instance`).
    #[serde(default)]
    pub who: String,
    /// What moves: a token, shares, a claim on a position.
    pub asset: String,
    pub direction: ValueDirection,
    /// The code path that moves the value between parties.
    pub mechanism: String,
}

/// Who can bring about the state a specification's `setup` describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SetupAuthorityKind {
    /// Any participant can create it through public entry points.
    Permissionless,
    /// It already exists on the deployed system; a live instance proves it.
    LiveState,
    /// It needs authority the protocol grants (owner, deployer, manager,
    /// timelock-gated configuration).
    Privileged,
}

impl SetupAuthorityKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Permissionless => "permissionless",
            Self::LiveState => "live_state",
            Self::Privileged => "privileged",
        }
    }
}

/// The authority a specification's precondition needs. The attacker model
/// says who drives the *sequence*; this says who can create the *state* the
/// sequence starts from, which is where a privileged configuration hides.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SetupAuthority {
    pub kind: SetupAuthorityKind,
    /// The project functions that create the setup state (`Contract.function`).
    /// Read against the project source for access guards.
    #[serde(default)]
    pub calls: Vec<String>,
    /// The authority needed, when `kind` is `privileged`.
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub note: String,
}

/// How a live observation is compared with the value a specification expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Comparison {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
    /// Anything but zero, the zero address, an empty string or `false`.
    Nonzero,
    /// The observation contains the expected text (case-insensitive).
    Contains,
}

impl Comparison {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Eq => "==",
            Self::Ne => "!=",
            Self::Gt => ">",
            Self::Gte => ">=",
            Self::Lt => "<",
            Self::Lte => "<=",
            Self::Nonzero => "nonzero",
            Self::Contains => "contains",
        }
    }
}

/// What a live read returned when it was checked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct LiveCheck {
    pub chain_id: u64,
    pub address: String,
    #[serde(default)]
    pub block: Option<u64>,
    pub observed: String,
    pub holds: bool,
}

/// A deployed instance exhibiting a specification's precondition, stated as a
/// read against the fork: call `signature` on `target` and compare.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct LiveInstance {
    #[serde(default)]
    pub chain_id: Option<u64>,
    /// An address, or a contract name the deployment book / scope resolves.
    pub target: String,
    /// `cast`-style signature with return types, e.g.
    /// `balanceOf(address)(uint256)`.
    pub signature: String,
    #[serde(default)]
    pub args: Vec<String>,
    pub comparison: Comparison,
    #[serde(default)]
    pub expected: String,
    /// Why this read proves the precondition.
    #[serde(default)]
    pub note: String,
    /// Filled when the read was executed; absent when no chain was reachable.
    #[serde(default)]
    pub check: Option<LiveCheck>,
}

/// The promise a specification says is broken, quoted from the project.
///
/// The quote is checked verbatim against the contract's source: a claim that
/// the code promises something has to point at where it does, and the nearest
/// comment is also where a "this is by design" note lives — so writing the
/// quote down is how the agent is made to read it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct IntentEvidence {
    /// `Contract.function` whose code or comments carry the promise.
    pub function: String,
    /// The comment, NatSpec line or `require` the promise rests on, verbatim.
    pub quote: String,
    /// Why the nearest documentation does not already bless the reached state.
    pub why_not_by_design: String,
}

/// What a harness has to measure to prove a specification — the oracle.
///
/// A state invariant is one kind of proof, and the wrong one for most of what
/// a bounty pays for: theft is shown by someone ending up richer, a rounding
/// defect by an error bound, a miscomputation by disagreement with a
/// reference, a freeze by a legitimate call that no longer goes through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OracleKind {
    /// `post_attack` fails to hold after the sequence (the default).
    StateInvariant,
    /// The actor's net value, across the measured assets, rises while a
    /// victim metric falls — measured in the same run.
    ValueDelta,
    /// The target's output disagrees with a reference beyond a tolerance.
    Differential,
    /// A numeric error (rounding, round-trip loss, monotonicity) exceeds a
    /// bound, or rounds in the caller's favour.
    Bounds,
    /// A legitimate caller's core operation stops succeeding.
    Liveness,
}

impl OracleKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StateInvariant => "state_invariant",
            Self::ValueDelta => "value_delta",
            Self::Differential => "differential",
            Self::Bounds => "bounds",
            Self::Liveness => "liveness",
        }
    }
}

/// Whether the call sequence is known or has to be searched for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    /// The specification's sequence is the path: replay it deterministically.
    #[default]
    Fixed,
    /// The sequence is a hint: fuzz handler entry points (and their inputs)
    /// against the oracle.
    Fuzz,
}

impl SearchMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Fuzz => "fuzz",
        }
    }
}

/// The oracle a harness must implement for one specification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct OracleSpec {
    pub kind: OracleKind,
    /// What to measure: assets whose balances make up the actor's value
    /// (`value_delta`), the functions compared (`differential`, `bounds`), the
    /// operations that must keep succeeding (`liveness`).
    #[serde(default)]
    pub measured: Vec<String>,
    /// The victim metric for `value_delta` (e.g. `other holders' redeemable
    /// assets`), or the reference for `differential` (a closed form, a
    /// deployed prior version's address, a sibling function).
    #[serde(default)]
    pub reference: Option<String>,
    /// The allowed error (`bounds`, `differential`) or profit threshold
    /// (`value_delta`), as the harness should state it.
    #[serde(default)]
    pub tolerance: Option<String>,
    /// Starting capital the actor may use — e.g. flash-loanable liquidity of
    /// the measured asset on the fork.
    #[serde(default)]
    pub capital: Option<String>,
    #[serde(default)]
    pub search: SearchMode,
    #[serde(default)]
    pub note: String,
}

/// Why a specification is not a known issue it resembles — the agent's answer
/// when the gate matches it against one. The reviewer reads it; the gate only
/// checks that it was given.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct KnownIssueDistinction {
    /// The known issue's id, as the refusal named it.
    pub id: String,
    /// The different root cause, in one or two sentences.
    pub difference: String,
}

/// How much the exploit moves, in the asset and (when a price exists) in USD.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LossMagnitude {
    /// Amount in the asset's units, as a decimal string.
    pub amount: String,
    pub asset: String,
    #[serde(default)]
    pub usd: Option<f64>,
    /// Where the number comes from: a live balance, a TVL read, a bound.
    #[serde(default)]
    pub basis: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditSpecification {
    /// One- or two-sentence summary of the overall vulnerable behavior
    /// this spec captures. Required at commit time.
    #[serde(default)]
    pub summary: String,
    /// What the exploit takes away. Required at commit time; the gate refuses
    /// a missing class and a `no_value_at_risk` one. `None` on specifications
    /// written before the class was required — they were never judged, and
    /// saying "no value" for them would be an invention.
    #[serde(default)]
    pub impact: Option<ImpactClass>,
    /// Who carries out the sequence. Required at commit time whenever the
    /// engagement declared a scope (see [`crate::scope::ScopeGate`]); `None` on
    /// specifications written before the field existed, which the gate reads as
    /// "not declared" rather than guessing.
    #[serde(default)]
    pub attacker: Option<AttackerModel>,
    /// The state that needs to be setup, for example, deploying contracts, setup admin
    pub setup: AuditStateSpecification,
    /// The state before the given attack
    pub pre_attack: AuditStateSpecification,
    /// The state after the given attack
    pub post_attack: AuditStateSpecification,
    /// The intended sequence
    pub sequence: Vec<SuqenceCallStep>,
    /// Functional claims about computations this specification is about.
    ///
    /// Empty for a state-shaped specification (the common case). Present when
    /// the defect lives in arithmetic, a ratio, a rounding step or a branch
    /// selector — the shapes a `post_attack` state invariant cannot state. A
    /// specification with at least one property may commit without a
    /// `post_attack` invariant.
    #[serde(default)]
    pub properties: Vec<SpecProperty>,
    /// Auxiliary contracts the PoC must deploy itself (attacker contracts,
    /// malicious callbacks, mocks). Keyed by the name the agent chose.
    #[serde(default)]
    pub auxiliary_contracts: BTreeMap<String, AuxiliaryContract>,
    /// Who ends up short and who ends up ahead. Empty on specifications
    /// written before the table existed.
    #[serde(default)]
    pub value_flow: Vec<ActorDelta>,
    /// Who can create the `setup` state.
    #[serde(default)]
    pub setup_authority: Option<SetupAuthority>,
    /// Deployed instances exhibiting the precondition, each checked against
    /// the fork when one is reachable.
    #[serde(default)]
    pub live_instances: Vec<LiveInstance>,
    /// Why the precondition must arise even though no instance exists yet —
    /// the alternative to a live instance for a permissionless setup.
    #[serde(default)]
    pub inevitability: Option<String>,
    #[serde(default)]
    pub magnitude: Option<LossMagnitude>,
    #[serde(default)]
    pub intent: Option<IntentEvidence>,
    /// What the harness must measure; `None` means a state invariant.
    #[serde(default)]
    pub oracle: Option<OracleSpec>,
    /// Known issues this specification was matched against and argues it is
    /// not.
    #[serde(default)]
    pub distinct_from: Vec<KnownIssueDistinction>,
}

impl AuditSpecification {
    /// Contracts the call sequence drives, in order, exactly as the agent wrote
    /// them (a step may name `VaultCore.vault` or leave a role label). Callers
    /// that compare against project or deployment names normalize with
    /// [`crate::scope::contract_name`].
    pub fn sequence_contracts(&self) -> Vec<String> {
        self.sequence
            .iter()
            .map(|step| step.contract.trim().to_string())
            .collect()
    }

    /// Everything the specification says about itself, as one haystack.
    /// Used to match a program's known-issue keywords against the claim rather
    /// than against the code — the specification's own words are where a
    /// rediscovery gives itself away.
    pub fn searchable_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&self.summary);
        for state in [&self.setup, &self.pre_attack, &self.post_attack] {
            out.push_str(&state.description);
            for (key, requirement) in &state.states_variables {
                out.push_str(key);
                out.push_str(requirement);
            }
            for (contract, requirement) in &state.contracts {
                out.push_str(contract);
                out.push_str(requirement);
            }
        }
        for step in &self.sequence {
            out.push_str(&step.contract);
            out.push_str(&step.function);
            out.push_str(&step.intention);
        }
        out
    }
}
