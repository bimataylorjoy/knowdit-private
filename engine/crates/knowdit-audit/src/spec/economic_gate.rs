//! The commit-time economic gate: the claims a bounty triager kills a finding
//! on, checked before the specification costs a harness run.
//!
//! Every candidate that survived the harness on the reference engagement and
//! then died in manual review died on one of a handful of questions, and none
//! of them needed a fuzzer to answer:
//!
//! * **direction** — does anyone other than the caller end up short? A caller
//!   who donates, forfeits or pays a fee they chose has harmed only themself;
//! * **setup authority** — can an unprivileged actor create the state the
//!   sequence starts from, or does it need the deployer / owner / a timelock?
//! * **live instance** — does any deployed position actually look like the
//!   setup, or is the precondition a state nobody is in?
//! * **magnitude** — is the loss worth a report, or is it cents?
//! * **fidelity** — is the external protocol the sequence leans on the real
//!   one, or a stand-in whose behaviour was invented?
//! * **intent** — does the code actually promise what the claim says it
//!   breaks, and does the nearest comment say the reached state is by design?
//!
//! The specification states each as a structured field, and this gate checks
//! them deterministically: the value-flow table by rule, setup calls against
//! the project's own access guards, live instances by reading the fork, the
//! intent quote against the source. What the gate cannot decide it leaves to
//! the reviewer, and what it can decide it never leaves to a prompt.

use std::collections::BTreeSet;
use std::sync::Arc;

use regex::Regex;

use crate::chain::ChainReader;
use crate::scope::{ProgramScope, ScopeGate, ScopeVerdict, TriageProfile};
use crate::types::{
    ActorRole, AuditSpecification, ImpactClass, OracleKind, SetupAuthorityKind, ValueDirection,
};

use super::ProjectIndex;

/// Configuration for [`EconomicGate`].
#[derive(Debug, Clone)]
pub struct EconomicGateOptions {
    /// Loss floor in USD. A declared magnitude below it is refused; `0.0`
    /// turns the floor off.
    pub min_loss_usd: f64,
    /// Require the intent quote. Off only for ablation.
    pub require_intent: bool,
}

impl Default for EconomicGateOptions {
    fn default() -> Self {
        Self {
            min_loss_usd: 50.0,
            require_intent: true,
        }
    }
}

/// Why the gate refused one specification. Each variant names the question
/// the specification failed, so a refusal reads as the reason a triager
/// would give.
#[derive(Debug, Clone, PartialEq)]
pub enum EconomicRule {
    ValueFlowMissing,
    NoLoserBesidesAttacker,
    AttackerDoesNotGain { impact: ImpactClass },
    SetupAuthorityMissing,
    SetupPrivileged { role: String },
    SetupGuarded { call: String, guard: String },
    SetupCallUnknown { call: String },
    LiveInstanceMissing { kind: SetupAuthorityKind },
    LiveInstanceFailed {
        target: String,
        signature: String,
        observed: String,
        comparison: String,
        expected: String,
    },
    LiveInstanceUnreadable { target: String, reason: String },
    MagnitudeMissing { impact: ImpactClass },
    MagnitudeBelowFloor { usd: f64, floor: f64 },
    StandInWithoutFidelity { aux: String },
    StandInForDeployed { aux: String, address: String },
    IntentMissing,
    IntentNotInSource { function: String },
    OracleMismatch { impact: ImpactClass, oracle: String },
}

impl EconomicRule {
    /// Short machine-readable id, for logs and the regression suite.
    pub fn id(&self) -> &'static str {
        match self {
            Self::ValueFlowMissing => "value-flow-missing",
            Self::NoLoserBesidesAttacker => "value-flow-direction",
            Self::AttackerDoesNotGain { .. } => "value-flow-no-gain",
            Self::SetupAuthorityMissing => "setup-authority-missing",
            Self::SetupPrivileged { .. } => "setup-privileged",
            Self::SetupGuarded { .. } => "setup-guarded",
            Self::SetupCallUnknown { .. } => "setup-call-unknown",
            Self::LiveInstanceMissing { .. } => "live-instance-missing",
            Self::LiveInstanceFailed { .. } => "live-instance-failed",
            Self::LiveInstanceUnreadable { .. } => "live-instance-unreadable",
            Self::MagnitudeMissing { .. } => "magnitude-missing",
            Self::MagnitudeBelowFloor { .. } => "magnitude-below-floor",
            Self::StandInWithoutFidelity { .. } => "stand-in-fidelity",
            Self::StandInForDeployed { .. } => "stand-in-for-deployed",
            Self::IntentMissing => "intent-missing",
            Self::IntentNotInSource { .. } => "intent-not-in-source",
            Self::OracleMismatch { .. } => "oracle-mismatch",
        }
    }

    pub fn clause(&self) -> String {
        match self {
            Self::ValueFlowMissing => "no value-flow table: call `set_value_flow` with one row per \
                party (at least the attacker and whoever ends up short), saying who gains, who \
                loses and through which code path"
                .to_string(),
            Self::NoLoserBesidesAttacker => "direction reversed: nobody other than the attacker \
                ends up short. A caller who donates, forfeits or pays a fee they chose harms only \
                themself — that is not theft and not griefing. Find the party that loses, or \
                abandon this direction"
                .to_string(),
            Self::AttackerDoesNotGain { impact } => format!(
                "`{}` needs the attacker to end up ahead, and the value-flow table has no \
                 attacker row that gains. Declare the class the table actually shows (griefing, \
                 freeze, insolvency), or show the gain",
                impact.as_str()
            ),
            Self::SetupAuthorityMissing => "no setup authority: call `set_setup_authority` saying \
                who can create the `setup` state (permissionless / already live / privileged) \
                and which project functions create it"
                .to_string(),
            Self::SetupPrivileged { role } => format!(
                "the setup state needs `{role}`, and this program pays only for what an \
                 unprivileged actor can reach — however real the arithmetic is, no unprivileged \
                 trigger exists"
            ),
            Self::SetupGuarded { call, guard } => format!(
                "the setup is declared permissionless, but `{call}` is guarded by `{guard}` in \
                 the project source. Declare the authority it needs, or name the permissionless \
                 path that creates the same state"
            ),
            Self::SetupCallUnknown { call } => format!(
                "setup call `{call}` does not resolve to a project function; name it as \
                 `Contract.function` from the call graph"
            ),
            Self::LiveInstanceMissing { kind } => match kind {
                SetupAuthorityKind::LiveState => "the setup is declared to exist on the deployed \
                    system but no live instance was given: call `add_live_instance` with a read \
                    that shows it (a fund with the configuration, a position with a balance)"
                    .to_string(),
                _ => "no live instance and no inevitability argument: either show a deployed \
                    instance in the setup state (`add_live_instance`) or explain with \
                    `set_inevitability` why any user of the permissionless entry point creates it"
                    .to_string(),
            },
            Self::LiveInstanceFailed {
                target,
                signature,
                observed,
                comparison,
                expected,
            } => format!(
                "live instance does not hold: `{signature}` on `{target}` returned `{observed}`, \
                 which is not {comparison} `{expected}`. The precondition is a state the deployed \
                 system is not in — find an instance that is, or abandon"
            ),
            Self::LiveInstanceUnreadable { target, reason } => format!(
                "live instance on `{target}` could not be read ({reason}); fix the target or \
                 signature — an unread precondition has not been shown to exist"
            ),
            Self::MagnitudeMissing { impact } => format!(
                "`{}` needs a magnitude: call `set_magnitude` with the amount at risk, the asset, \
                 and its USD value where a price exists",
                impact.as_str()
            ),
            Self::MagnitudeBelowFloor { usd, floor } => format!(
                "the loss is ${usd:.2}, below the ${floor:.2} floor for a report. A bounded, \
                 cent-sized payout is informational however real the mechanism is"
            ),
            Self::StandInWithoutFidelity { aux } => format!(
                "auxiliary contract `{aux}` stands in for an external protocol without a \
                 fidelity note: say how its behaviour was checked against the real deployment \
                 (address, selectors, the state transitions the sequence relies on)"
            ),
            Self::StandInForDeployed { aux, address } => format!(
                "auxiliary contract `{aux}` stands in for a dependency that is deployed on the \
                 fork at {address}. Use the real one — a stand-in whose behaviour was invented \
                 fabricates the mechanism"
            ),
            Self::IntentMissing => "no intent evidence: call `set_intent_evidence` quoting the \
                comment, NatSpec or `require` the broken promise rests on, and why it does not \
                already bless the reached state"
                .to_string(),
            Self::OracleMismatch { impact, oracle } => format!(
                "`{}` is proved by someone ending up richer, and the declared oracle is `{oracle}`. \
                 Call `set_oracle` with `value_delta` — the actor's net value across the measured \
                 assets rises while the victim metric falls, in the same run — or declare the \
                 impact the proof actually shows",
                impact.as_str()
            ),
            Self::IntentNotInSource { function } => format!(
                "the intent quote for `{function}` is not in that contract's source. Quote the \
                 text verbatim (read it with `read_contract_source`) — a paraphrase is not \
                 evidence"
            ),
        }
    }
}

/// What the gate concluded about one specification.
#[derive(Debug, Clone, Default)]
pub struct EconomicVerdict {
    pub refusal: Option<EconomicRule>,
    /// Observations worth keeping with the specification (unverified live
    /// instances, a skipped check) — carried to the reviewer, never a refusal.
    pub notes: Vec<String>,
}

/// The commit-time economic gate. See the module docs.
#[derive(Debug)]
pub struct EconomicGate {
    options: EconomicGateOptions,
    /// The program pays for unprivileged actors only.
    requires_unprivileged: bool,
    /// An audit competition on a commit: nothing is deployed to read live
    /// state from, and loss size is graded by the judge rather than floored
    /// in dollars (there is no TVL yet to measure it against).
    contest: bool,
    access_guard: Regex,
    /// What makes a modifier body a check on its caller rather than on its
    /// arguments or on state.
    caller_reference: Regex,
    /// A call site inside a modifier body.
    call_site: Regex,
}

impl EconomicGate {
    pub fn new(options: EconomicGateOptions, scope: Option<&ProgramScope>) -> Self {
        let requires_unprivileged = scope
            .map(|s| s.requires_unprivileged_actor && !s.allow_privileged_impacts)
            .unwrap_or(true);
        let contest = scope.is_some_and(|s| s.triage_profile == TriageProfile::Contest);
        // Access guards as they appear in Solidity: `only*` modifiers, the
        // solmate / OZ helpers, and inline sender checks. Case-sensitive on
        // `only` so `onlyOwner` matches and `commonly` does not.
        let access_guard = Regex::new(
            r"\b(only[A-Z_]\w*|requiresAuth|_checkOwner\s*\(|_checkRole\s*\(|hasRole\s*\(|__assertIs\w*\s*\(|__validateIs\w*\s*\(|require\s*\(\s*msg\.sender\s*==|msg\.sender\s*==\s*\w+|\w+\s*==\s*msg\.sender)",
        )
        .expect("static access-guard pattern compiles");
        let caller_reference = Regex::new(
            r"(?i)msg\.sender|_msgSender|tx\.origin|owner|auth|role|admin|govern|guardian|keeper|operator|minter|manager|controller",
        )
        .expect("static caller-reference pattern compiles");
        let call_site =
            Regex::new(r"\b([A-Za-z_]\w*)\s*\(").expect("static call-site pattern compiles");
        Self {
            options,
            requires_unprivileged,
            contest,
            access_guard,
            caller_reference,
            call_site,
        }
    }

    /// Check `spec`, recording what the fork said into its live instances.
    pub async fn admit(
        &self,
        spec: &mut AuditSpecification,
        project: &ProjectIndex,
        chain: Option<&ChainReader>,
    ) -> EconomicVerdict {
        let mut verdict = EconomicVerdict::default();
        let checks = [
            self.check_value_flow(spec),
            self.check_oracle(spec),
            self.check_magnitude(spec),
            self.check_intent(spec, project),
            self.check_setup(spec, project),
        ];
        if let Some(rule) = checks.into_iter().flatten().next() {
            verdict.refusal = Some(rule);
            return verdict;
        }
        if let Some(rule) = self.check_live_instances(spec, chain, &mut verdict.notes).await {
            verdict.refusal = Some(rule);
            return verdict;
        }
        verdict.refusal = self.check_stand_ins(spec, chain, &mut verdict.notes).await;
        verdict
    }

    fn check_value_flow(&self, spec: &AuditSpecification) -> Option<EconomicRule> {
        if spec.value_flow.is_empty() {
            return Some(EconomicRule::ValueFlowMissing);
        }
        let others_harmed = spec
            .value_flow
            .iter()
            .any(|row| row.role != ActorRole::Attacker && row.direction.is_harm());
        if !others_harmed {
            return Some(EconomicRule::NoLoserBesidesAttacker);
        }
        let impact = spec.impact?;
        let attacker_gains = spec
            .value_flow
            .iter()
            .any(|row| row.role == ActorRole::Attacker && row.direction == ValueDirection::Gains);
        if impact == ImpactClass::DirectTheftOfFunds && !attacker_gains {
            return Some(EconomicRule::AttackerDoesNotGain { impact });
        }
        None
    }

    /// Theft is a claim about value, so its proof has to measure value: a
    /// state invariant can show a number is odd without showing anyone is
    /// richer, which is the gap that let fee-avoidance and self-harm through
    /// as "theft" before.
    fn check_oracle(&self, spec: &AuditSpecification) -> Option<EconomicRule> {
        let impact = spec.impact?;
        let kind = spec
            .oracle
            .as_ref()
            .map(|oracle| oracle.kind)
            .unwrap_or(OracleKind::StateInvariant);
        (impact == ImpactClass::DirectTheftOfFunds && kind != OracleKind::ValueDelta).then(|| {
            EconomicRule::OracleMismatch {
                impact,
                oracle: kind.as_str().to_string(),
            }
        })
    }

    fn check_magnitude(&self, spec: &AuditSpecification) -> Option<EconomicRule> {
        let impact = spec.impact?;
        let needs_magnitude = matches!(
            impact,
            ImpactClass::DirectTheftOfFunds
                | ImpactClass::ProtocolInsolvency
                | ImpactClass::PermanentFreeze
        );
        match &spec.magnitude {
            None if needs_magnitude => Some(EconomicRule::MagnitudeMissing { impact }),
            Some(magnitude) => match magnitude.usd {
                Some(usd)
                    if !self.contest
                        && self.options.min_loss_usd > 0.0
                        && usd < self.options.min_loss_usd =>
                {
                    Some(EconomicRule::MagnitudeBelowFloor {
                        usd,
                        floor: self.options.min_loss_usd,
                    })
                }
                _ => None,
            },
            None => None,
        }
    }

    fn check_intent(&self, spec: &AuditSpecification, project: &ProjectIndex) -> Option<EconomicRule> {
        let Some(intent) = &spec.intent else {
            return self.options.require_intent.then_some(EconomicRule::IntentMissing);
        };
        let contract = intent
            .function
            .split_once('.')
            .map(|(contract, _)| contract)
            .unwrap_or(&intent.function)
            .trim();
        (!project.source_contains(contract, &intent.quote)).then(|| {
            EconomicRule::IntentNotInSource {
                function: intent.function.clone(),
            }
        })
    }

    fn check_setup(&self, spec: &AuditSpecification, project: &ProjectIndex) -> Option<EconomicRule> {
        let Some(setup) = &spec.setup_authority else {
            return Some(EconomicRule::SetupAuthorityMissing);
        };
        if setup.kind == SetupAuthorityKind::Privileged {
            return self.requires_unprivileged.then(|| EconomicRule::SetupPrivileged {
                role: setup
                    .role
                    .clone()
                    .filter(|role| !role.trim().is_empty())
                    .unwrap_or_else(|| "authority the protocol grants".to_string()),
            });
        }
        // A permissionless or live setup is read against the code that creates
        // it: a guarded entry point contradicts the claim.
        for call in &setup.calls {
            let Some((contract, function)) = call.trim().split_once('.') else {
                return Some(EconomicRule::SetupCallUnknown { call: call.clone() });
            };
            let bodies = project.functions_named(contract.trim(), function.trim());
            if bodies.is_empty() {
                return Some(EconomicRule::SetupCallUnknown { call: call.clone() });
            }
            if setup.kind != SetupAuthorityKind::Permissionless {
                continue;
            }
            // Every overload guarded means no permissionless path through
            // this name; one open overload is enough to keep the claim.
            let guards: Vec<String> = bodies
                .iter()
                .filter_map(|f| f.content.as_deref())
                .filter_map(|body| self.caller_guard_in(body, contract.trim(), project))
                .collect();
            if guards.len() == bodies.len()
                && let Some(guard) = guards.into_iter().next()
            {
                return Some(EconomicRule::SetupGuarded {
                    call: call.clone(),
                    guard,
                });
            }
        }
        None
    }

    /// The first guard in a function that restricts who may call it.
    ///
    /// An `only*` modifier is a naming convention, not a guarantee: Lambo's
    /// `onlyWhiteListed(token)` checks that an *argument* is whitelisted and
    /// lets any caller through. So a modifier the project declares is read,
    /// and one whose body neither names the caller nor calls out to code that
    /// might is not an access guard. A modifier the project does not declare
    /// (`onlyOwner` from a vendored library) keeps its conventional meaning.
    fn caller_guard_in(&self, body: &str, contract: &str, project: &ProjectIndex) -> Option<String> {
        self.guards_in(body).into_iter().find(|guard| {
            !guard.starts_with("only")
                || project
                    .modifier_source(contract, guard)
                    .is_none_or(|modifier| !self.checks_only_arguments_or_state(&modifier))
        })
    }

    /// Whether a modifier body checks only its arguments or contract state:
    /// no reference to the caller or to an authority, and no call besides
    /// `require` / `revert` / `assert` and error or type constructors, since
    /// a helper it calls could check the caller where this cannot see.
    fn checks_only_arguments_or_state(&self, modifier: &str) -> bool {
        let code = Self::strip_strings(&Self::strip_comments(modifier));
        let Some((_, body)) = code.split_once('{') else {
            return false;
        };
        if self.caller_reference.is_match(body) {
            return false;
        }
        self.call_site.captures_iter(body).all(|call| {
            let name = &call[1];
            matches!(name, "require" | "revert" | "assert")
                || name.starts_with(|c: char| c.is_ascii_uppercase())
        })
    }

    /// Every access guard in a function, looking at the header (where
    /// modifiers live) and the first statements of the body (where inline
    /// sender checks live). Comments are stripped so a guard *mentioned* in
    /// prose does not count.
    fn guards_in(&self, body: &str) -> Vec<String> {
        let code = Self::strip_comments(body);
        let (header, rest) = code.split_once('{').unwrap_or((code.as_str(), ""));
        let head_of_body: String = rest.lines().take(6).collect::<Vec<_>>().join("\n");
        self.access_guard
            .find_iter(header)
            .chain(self.access_guard.find_iter(&head_of_body))
            .map(|m| m.as_str().trim().to_string())
            .collect()
    }

    fn strip_comments(code: &str) -> String {
        code.lines()
            .map(|line| line.split("//").next().unwrap_or(""))
            .filter(|line| !line.trim_start().starts_with('*') && !line.trim_start().starts_with("/*"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Blank out string literals, so a revert message that says "owner" is
    /// not read as a check on the owner.
    fn strip_strings(code: &str) -> String {
        let mut out = String::with_capacity(code.len());
        let mut quote: Option<char> = None;
        for ch in code.chars() {
            match quote {
                Some(q) if ch == q => quote = None,
                Some(_) => {}
                None if ch == '"' || ch == '\'' => quote = Some(ch),
                None => out.push(ch),
            }
        }
        out
    }

    async fn check_live_instances(
        &self,
        spec: &mut AuditSpecification,
        chain: Option<&ChainReader>,
        notes: &mut Vec<String>,
    ) -> Option<EconomicRule> {
        let kind = spec.setup_authority.as_ref().map(|s| s.kind)?;
        let has_inevitability = spec
            .inevitability
            .as_deref()
            .is_some_and(|text| text.trim().len() >= 20);
        match kind {
            // A contest commit has no deployment to read: the state has to be
            // argued to arise, the same as a permissionless one.
            SetupAuthorityKind::LiveState
                if spec.live_instances.is_empty() && !(self.contest && has_inevitability) =>
            {
                return Some(EconomicRule::LiveInstanceMissing { kind });
            }
            SetupAuthorityKind::Permissionless
                if spec.live_instances.is_empty() && !has_inevitability =>
            {
                return Some(EconomicRule::LiveInstanceMissing { kind });
            }
            _ => {}
        }
        let Some(chain) = chain else {
            if !spec.live_instances.is_empty() {
                notes.push(format!(
                    "{} live instance(s) declared but not verified: no chain endpoint in this run",
                    spec.live_instances.len()
                ));
            }
            return None;
        };
        for instance in spec.live_instances.iter_mut() {
            match chain.check(instance).await {
                Ok(check) => {
                    let holds = check.holds;
                    let observed = check.observed.clone();
                    instance.check = Some(check);
                    if !holds {
                        return Some(EconomicRule::LiveInstanceFailed {
                            target: instance.target.clone(),
                            signature: instance.signature.clone(),
                            observed,
                            comparison: instance.comparison.as_str().to_string(),
                            expected: instance.expected.clone(),
                        });
                    }
                }
                Err(reason) => {
                    return Some(EconomicRule::LiveInstanceUnreadable {
                        target: instance.target.clone(),
                        reason,
                    });
                }
            }
        }
        None
    }

    async fn check_stand_ins(
        &self,
        spec: &AuditSpecification,
        chain: Option<&ChainReader>,
        notes: &mut Vec<String>,
    ) -> Option<EconomicRule> {
        for (name, aux) in &spec.auxiliary_contracts {
            let Some(real) = aux.stands_in_for.as_deref().filter(|s| !s.trim().is_empty()) else {
                continue;
            };
            let fidelity = aux.fidelity.as_deref().unwrap_or("").trim();
            if fidelity.is_empty() {
                return Some(EconomicRule::StandInWithoutFidelity { aux: name.clone() });
            }
            let Some(chain) = chain else {
                continue;
            };
            match chain.resolve(None, real) {
                Ok((chain_id, address)) => {
                    if chain.has_code(chain_id, &address).await == Some(true) {
                        return Some(EconomicRule::StandInForDeployed {
                            aux: name.clone(),
                            address,
                        });
                    }
                }
                Err(_) => notes.push(format!(
                    "stand-in `{name}` names `{real}`, which does not resolve on the fork; its \
                     fidelity note is the only evidence"
                )),
            }
        }
        None
    }
}

/// Why a specification was refused at commit, by whichever gate refused it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateRefusal {
    /// Stable id of the rule (`scope-*`, `value-flow-*`, `impact-*`, ...).
    pub id: String,
    pub clause: String,
}

/// Every deterministic check a specification faces at commit, in order: the
/// impact class, the program's scope, then the economic claim. One struct so
/// the commit tool and the regression suite run the same path — a suite that
/// re-implemented the order would test something the agent never meets.
#[derive(Debug, Clone, Default)]
pub struct CommitGates {
    pub scope: Option<Arc<ScopeGate>>,
    pub economic: Option<Arc<EconomicGate>>,
    pub chain: Option<Arc<ChainReader>>,
}

impl CommitGates {
    /// Admit or refuse `spec`. On admission, returns the notes worth keeping
    /// with it; `spec` carries what the fork said about its live instances.
    pub async fn admit(
        &self,
        spec: &mut AuditSpecification,
        project: &Arc<ProjectIndex>,
    ) -> std::result::Result<Vec<String>, GateRefusal> {
        let mut notes: Vec<String> = Vec::new();
        match spec.impact {
            None => {
                return Err(GateRefusal {
                    id: "impact-not-declared".to_string(),
                    clause: "no impact class declared; call set_impact_class before committing — \
                             a specification that cannot name what the exploit takes away cannot \
                             be submitted to any program, however clean its proof is"
                        .to_string(),
                });
            }
            Some(impact) if !impact.is_reportable() => {
                return Err(GateRefusal {
                    id: "impact-not-reportable".to_string(),
                    clause: format!(
                        "impact class `{}` is not reportable; find the loss that makes this \
                         direction matter, or abandon it with finalize",
                        impact.as_str()
                    ),
                });
            }
            Some(_) => {}
        }
        if let Some(gate) = &self.scope {
            let project = project.clone();
            let is_project_contract = move |name: &str| project.has_contract(name);
            let aux: BTreeSet<String> = spec.auxiliary_contracts.keys().cloned().collect();
            match gate.admit(spec, &is_project_contract, Some(&aux)).await {
                ScopeVerdict::Excluded { rule } => {
                    tracing::info!(
                        program = %gate.scope().program,
                        "[scope] refused a specification: {}",
                        rule.clause()
                    );
                    return Err(GateRefusal {
                        id: rule.id().to_string(),
                        clause: rule.clause(),
                    });
                }
                ScopeVerdict::Admissible {
                    matched_assets,
                    notes: scope_notes,
                } => {
                    for note in &scope_notes {
                        tracing::info!("[scope] {}", note);
                    }
                    if !matched_assets.is_empty() {
                        tracing::debug!("[scope] admissible on asset(s): {}", matched_assets.join(", "));
                    }
                }
            }
        }
        if let Some(gate) = &self.economic {
            let verdict = gate.admit(spec, project, self.chain.as_deref()).await;
            if let Some(rule) = verdict.refusal {
                tracing::info!("[economic] refused a specification ({}): {}", rule.id(), rule.clause());
                return Err(GateRefusal {
                    id: rule.id().to_string(),
                    clause: rule.clause(),
                });
            }
            notes.extend(verdict.notes);
        }
        // Last, the judgment no field can make: would the program pay for this?
        // A judge that fails to answer admits — a provider error must not cost
        // a finding — and says so in the log.
        if let Some(judge) = self.scope.as_ref().and_then(|gate| gate.judge()) {
            match judge.judge(spec).await {
                Ok(verdict) if !verdict.admit => {
                    let rule = if verdict.rule.trim().is_empty() {
                        "program rules".to_string()
                    } else {
                        verdict.rule.trim().to_string()
                    };
                    tracing::info!("[scope-judge] refused a specification ({rule}): {}", verdict.reason);
                    return Err(GateRefusal {
                        id: "scope-judge".to_string(),
                        clause: format!(
                            "the program's triage judge would not pay for this candidate ({rule}): \
                             {}. If the judge misread the claim, state the attacker, the victim, \
                             the attacker's cost and the loss explicitly and commit again; \
                             otherwise abandon",
                            verdict.reason.trim()
                        ),
                    });
                }
                Ok(verdict) => {
                    tracing::info!("[scope-judge] admitted a specification: {}", verdict.reason);
                }
                Err(err) => {
                    tracing::warn!("[scope-judge] no verdict, admitting unjudged: {err:#}");
                    notes.push("scope judge gave no verdict".to_string());
                }
            }
        }
        Ok(notes)
    }
}

#[cfg(test)]
mod tests {
    use crate::regression::{
        RegressionCase, RegressionContract, RegressionExpectation, RegressionFunction,
        RegressionGateConfig, RegressionSet,
    };
    use crate::scope::{ProgramScope, ScopeAsset, TriageProfile};
    use crate::types::{
        ActorDelta, ActorRole, AttackerModel, AuditSpecification, ImpactClass, IntentEvidence,
        LossMagnitude, OracleKind, OracleSpec, SearchMode, SetupAuthority, SetupAuthorityKind,
        SuqenceCallStep, ValueDirection,
    };

    /// A fictional vault: `deposit` is open, `setFee` is owner-only, and the
    /// contract carries one NatSpec promise the intent check can quote.
    fn project() -> Vec<RegressionContract> {
        vec![RegressionContract {
            name: "Vault".to_string(),
            file: String::new(),
            source: "/// @notice Shares are always redeemable for a pro-rata slice of assets.\n\
                     modifier onlyListed(address token) {\n\
                         require(listed[token], \"not an owner-listed token\");\n\
                         _;\n\
                     }\n\
                     modifier onlyKeeper() {\n\
                         require(isKeeper[msg.sender]);\n\
                         _;\n\
                     }\n\
                     modifier onlyChecked(address token) {\n\
                         _validate(token);\n\
                         _;\n\
                     }\n"
                .to_string(),
            inherits: Vec::new(),
            functions: vec![
                RegressionFunction {
                    name: "launch".to_string(),
                    source: "function launch(address t) external onlyListed(t) { pools[t] = true; }"
                        .to_string(),
                },
                RegressionFunction {
                    name: "harvest".to_string(),
                    source: "function harvest() external onlyKeeper { _harvest(); }".to_string(),
                },
                RegressionFunction {
                    name: "sweep".to_string(),
                    source: "function sweep(address t) external onlyChecked(t) { _sweep(t); }"
                        .to_string(),
                },
                RegressionFunction {
                    name: "mixed".to_string(),
                    source: "function mixed(address t) external onlyListed(t) onlyOwner { _mix(t); }"
                        .to_string(),
                },
                RegressionFunction {
                    name: "deposit".to_string(),
                    source: "function deposit(uint256 a) external { shares[msg.sender] += a; }"
                        .to_string(),
                },
                RegressionFunction {
                    name: "setFee".to_string(),
                    source: "function setFee(uint256 f) external onlyOwner { fee = f; }".to_string(),
                },
            ],
        }]
    }

    fn row(role: ActorRole, direction: ValueDirection) -> ActorDelta {
        ActorDelta {
            role,
            who: String::new(),
            asset: "WETH".to_string(),
            direction,
            mechanism: "Vault.redeem pays from the shared pool".to_string(),
        }
    }

    /// A specification every rule admits; each case below breaks one thing.
    fn good() -> AuditSpecification {
        AuditSpecification {
            summary: "redeem overpays the caller from other holders' assets".to_string(),
            impact: Some(ImpactClass::DirectTheftOfFunds),
            attacker: Some(AttackerModel::default()),
            sequence: vec![SuqenceCallStep {
                contract: "Vault".to_string(),
                function: "deposit".to_string(),
                intention: "enter".to_string(),
            }],
            value_flow: vec![
                row(ActorRole::Attacker, ValueDirection::Gains),
                row(ActorRole::OtherHolders, ValueDirection::Loses),
            ],
            setup_authority: Some(SetupAuthority {
                kind: SetupAuthorityKind::Permissionless,
                calls: vec!["Vault.deposit".to_string()],
                role: None,
                note: "any depositor".to_string(),
            }),
            inevitability: Some("any user reaches the setup with one public deposit call".to_string()),
            magnitude: Some(LossMagnitude {
                amount: "100".to_string(),
                asset: "WETH".to_string(),
                usd: Some(250_000.0),
                basis: "pool balance".to_string(),
            }),
            intent: Some(IntentEvidence {
                function: "Vault.redeem".to_string(),
                quote: "Shares are always redeemable for a pro-rata slice of assets.".to_string(),
                why_not_by_design: "the caller receives more than pro-rata".to_string(),
            }),
            oracle: Some(OracleSpec {
                kind: OracleKind::ValueDelta,
                measured: vec!["WETH".to_string()],
                reference: Some("other holders' redeemable WETH".to_string()),
                tolerance: None,
                capital: None,
                search: SearchMode::Fixed,
                note: String::new(),
            }),
            ..Default::default()
        }
    }

    fn case(id: &str, expect: RegressionExpectation, edit: impl Fn(&mut AuditSpecification)) -> RegressionCase {
        let mut spec = good();
        edit(&mut spec);
        RegressionCase {
            id: id.to_string(),
            claim: String::new(),
            kill: String::new(),
            expect,
            spec,
        }
    }

    fn refused(rule: &str) -> RegressionExpectation {
        RegressionExpectation::Refused {
            any_of: vec![rule.to_string()],
        }
    }

    fn program(triage_profile: TriageProfile) -> ProgramScope {
        ProgramScope {
            program: "synthetic".to_string(),
            primacy: Default::default(),
            chain_id: None,
            assets: vec![ScopeAsset {
                name: "Vault".to_string(),
                chain_id: None,
                address: None,
                value_usd: None,
            }],
            impacts_in_scope: Vec::new(),
            out_of_scope: Vec::new(),
            use_platform_defaults: true,
            severity_table: Vec::new(),
            excluded_impacts: Vec::new(),
            min_asset_value_usd: None,
            requires_unprivileged_actor: true,
            allow_privileged_impacts: false,
            excluded_contracts: Vec::new(),
            known_issues: Vec::new(),
            assumptions: Vec::new(),
            triage_profile,
        }
    }

    #[tokio::test]
    async fn each_rule_refuses_its_own_failure_and_admits_the_rest() {
        let set = RegressionSet {
            name: "economic-gate-rules".to_string(),
            source: "synthetic".to_string(),
            program: program(TriageProfile::Bounty),
            gate: RegressionGateConfig::default(),
            chain: None,
            project: project(),
            cases: vec![
                case("admitted", RegressionExpectation::Admitted, |_| {}),
                case("donation", refused("value-flow-direction"), |s| {
                    s.value_flow = vec![
                        row(ActorRole::Attacker, ValueDirection::Loses),
                        row(ActorRole::OtherHolders, ValueDirection::Gains),
                    ];
                }),
                case("no-gain", refused("value-flow-no-gain"), |s| {
                    s.value_flow[0].direction = ValueDirection::Unchanged;
                }),
                case("oracle", refused("oracle-mismatch"), |s| s.oracle = None),
                case("cents", refused("magnitude-below-floor"), |s| {
                    s.magnitude.as_mut().map(|m| m.usd = Some(2.42));
                }),
                case("paraphrase", refused("intent-not-in-source"), |s| {
                    s.intent.as_mut().map(|i| i.quote = "redemption is always fair".to_string());
                }),
                case("guarded", refused("setup-guarded"), |s| {
                    s.setup_authority.as_mut().map(|a| a.calls = vec!["Vault.setFee".to_string()]);
                }),
                case("argument-modifier", RegressionExpectation::Admitted, |s| {
                    s.setup_authority.as_mut().map(|a| a.calls = vec!["Vault.launch".to_string()]);
                }),
                case("sender-modifier", refused("setup-guarded"), |s| {
                    s.setup_authority.as_mut().map(|a| a.calls = vec!["Vault.harvest".to_string()]);
                }),
                case("helper-modifier", refused("setup-guarded"), |s| {
                    s.setup_authority.as_mut().map(|a| a.calls = vec!["Vault.sweep".to_string()]);
                }),
                case("stacked-modifier", refused("setup-guarded"), |s| {
                    s.setup_authority.as_mut().map(|a| a.calls = vec!["Vault.mixed".to_string()]);
                }),
                case("privileged", refused("setup-privileged"), |s| {
                    s.setup_authority = Some(SetupAuthority {
                        kind: SetupAuthorityKind::Privileged,
                        calls: Vec::new(),
                        role: Some("owner".to_string()),
                        note: "owner configures".to_string(),
                    });
                }),
                case("absent-state", refused("live-instance-missing"), |s| s.inevitability = None),
            ],
        };
        let report = set.evaluate().await;
        assert_eq!(report.failures(), 0, "{}", report.render());
    }

    /// A contest commit has no deployment and no TVL: an argued state and a
    /// small loss go on to the judge, while who loses and who may act are
    /// held exactly as in a bounty.
    #[tokio::test]
    async fn a_contest_grades_size_and_argued_state_but_keeps_the_rest() {
        let live_state = |s: &mut AuditSpecification| {
            s.setup_authority = Some(SetupAuthority {
                kind: SetupAuthorityKind::LiveState,
                calls: vec!["Vault.deposit".to_string()],
                role: None,
                note: "a vault with depositors".to_string(),
            });
        };
        let set = RegressionSet {
            name: "economic-gate-contest".to_string(),
            source: "synthetic".to_string(),
            program: program(TriageProfile::Contest),
            gate: RegressionGateConfig::default(),
            chain: None,
            project: project(),
            cases: vec![
                case("admitted", RegressionExpectation::Admitted, |_| {}),
                case("cents", RegressionExpectation::Admitted, |s| {
                    s.magnitude.as_mut().map(|m| m.usd = Some(2.42));
                }),
                case("argued-live-state", RegressionExpectation::Admitted, live_state),
                case("unargued-live-state", refused("live-instance-missing"), |s| {
                    live_state(s);
                    s.inevitability = None;
                }),
                case("donation", refused("value-flow-direction"), |s| {
                    s.value_flow = vec![
                        row(ActorRole::Attacker, ValueDirection::Loses),
                        row(ActorRole::OtherHolders, ValueDirection::Gains),
                    ];
                }),
                case("guarded", refused("setup-guarded"), |s| {
                    s.setup_authority.as_mut().map(|a| a.calls = vec!["Vault.setFee".to_string()]);
                }),
                case("paraphrase", refused("intent-not-in-source"), |s| {
                    s.intent.as_mut().map(|i| i.quote = "redemption is always fair".to_string());
                }),
            ],
        };
        let report = set.evaluate().await;
        assert_eq!(report.failures(), 0, "{}", report.render());
    }
}
