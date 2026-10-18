//! Spec-builder agent tools + the draft state they mutate.
//!
//!
//! Split out of `spec/mod.rs`: the `#[llmy::agent::tool]` structs, the
//! shared `SpecDraft` they build up, and the commit-time `post_attack`
//! shape gate. `build_tool_box` assembles the full set so each tool's
//! fields stay private to this module — only `PlannedLinkWork::run_agent`
//! (in `spec::mod`) needs the resulting `ToolBox` plus the `DraftHandle` to
//! read the committed result back out.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;
use tokio::sync::Mutex;

use llmy::agent::tool::ToolBox;

use knowdit_repo_model::storage::FunctionStateVariable;

use crate::chain::{ChainQuery, ChainReader};
use crate::scope::ScopeGate;
use crate::types::{
    ActorDelta, AttackerModel, AttackerPrivilege, AuditSpecification, AuditStateSpecification,
    AuxStateVariable, AuxiliaryContract, Comparison, ImpactClass, IntentEvidence, LiveInstance,
    KnownIssueDistinction, LossMagnitude, OracleKind, OracleSpec, PropertyKind, SearchMode, SetupAuthority, SetupAuthorityKind, SpecProperty,
    SuqenceCallStep,
};

use super::economic_gate::{CommitGates, EconomicGate};
use super::patterns::PatternLibrary;
use super::{LinkSource, LinkSpecStatus, ProjectIndex};

// ---------------------------------------------------------------------------
// Spec draft + tools
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct SpecDraft {
    pub setup: AuditStateSpecification,
    pub pre_attack: AuditStateSpecification,
    pub post_attack: AuditStateSpecification,
    pub sequence: Vec<SuqenceCallStep>,
    /// What the exploit takes away, declared via `set_impact_class`. Carried
    /// into the committed spec; cleared on `reset_draft`.
    pub impact: Option<ImpactClass>,
    /// Who carries out the sequence, declared via `set_attacker_model`. Carried
    /// into the committed spec; cleared on `reset_draft`. `None` until the
    /// agent says who the caller is — the scope gate reads that as
    /// "not declared" and refuses the commit when the engagement requires an
    /// unprivileged actor.
    pub attacker: Option<AttackerModel>,
    /// Functional claims about computations, declared via `add_property`.
    /// Carried into the committed spec; cleared on `reset_draft`.
    pub properties: Vec<SpecProperty>,
    /// Auxiliary contracts the PoC must deploy itself, declared via
    /// `add_aux_contract` / `add_aux_contract_state`. Carried into the
    /// committed spec; cleared on `reset_draft`.
    pub auxiliary_contracts: BTreeMap<String, AuxiliaryContract>,
    /// The economic claim, declared via `set_value_flow`,
    /// `set_setup_authority`, `add_live_instance`, `set_inevitability`,
    /// `set_magnitude` and `set_intent_evidence`. Carried into the committed
    /// spec and checked by the economic gate; cleared on `reset_draft`.
    pub value_flow: Vec<ActorDelta>,
    pub setup_authority: Option<SetupAuthority>,
    pub live_instances: Vec<LiveInstance>,
    pub inevitability: Option<String>,
    pub magnitude: Option<LossMagnitude>,
    pub intent: Option<IntentEvidence>,
    /// The oracle, declared via `set_oracle`.
    pub oracle: Option<OracleSpec>,
    /// Known issues argued distinct, via `distinguish_from_known_issue`.
    pub distinct_from: Vec<KnownIssueDistinction>,
    /// Specs that have been committed already.
    pub completed: Vec<AuditSpecification>,
    /// Set on `finalize`. Until then the agent loop keeps spinning.
    pub final_status: Option<LinkSpecStatus>,
    pub abort_reason: Option<String>,
    /// How many `abandoned` finalizes were sent back for missing composition
    /// analysis (see [`FinalizeTool`]); capped so the gate never eats the
    /// whole step budget.
    pub composition_rejections: usize,
    pub final_summary: Option<String>,
}

impl SpecDraft {
    fn reset_draft(&mut self) {
        self.setup = AuditStateSpecification::default();
        self.pre_attack = AuditStateSpecification::default();
        self.post_attack = AuditStateSpecification::default();
        self.sequence.clear();
        self.impact = None;
        self.attacker = None;
        self.properties.clear();
        self.auxiliary_contracts.clear();
        self.value_flow.clear();
        self.setup_authority = None;
        self.live_instances.clear();
        self.inevitability = None;
        self.magnitude = None;
        self.intent = None;
        self.oracle = None;
        self.distinct_from.clear();
    }

    fn build_spec(&self, summary: String) -> AuditSpecification {
        AuditSpecification {
            summary,
            impact: self.impact,
            attacker: self.attacker.clone(),
            setup: self.setup.clone(),
            pre_attack: self.pre_attack.clone(),
            post_attack: self.post_attack.clone(),
            sequence: self.sequence.clone(),
            properties: self.properties.clone(),
            auxiliary_contracts: self.auxiliary_contracts.clone(),
            value_flow: self.value_flow.clone(),
            setup_authority: self.setup_authority.clone(),
            live_instances: self.live_instances.clone(),
            inevitability: self.inevitability.clone(),
            magnitude: self.magnitude.clone(),
            intent: self.intent.clone(),
            oracle: self.oracle.clone(),
            distinct_from: self.distinct_from.clone(),
        }
    }

    /// Case-insensitive lookup of a declared auxiliary contract.
    fn find_aux_contract(&self, name: &str) -> Option<&AuxiliaryContract> {
        let needle = name.trim().to_lowercase();
        self.auxiliary_contracts
            .iter()
            .find(|(k, _)| k.to_lowercase() == needle)
            .map(|(_, v)| v)
    }

    /// The stored (original-case) key of a declared auxiliary contract,
    /// matched case-insensitively.
    fn aux_contract_key(&self, name: &str) -> Option<String> {
        let needle = name.trim().to_lowercase();
        self.auxiliary_contracts
            .keys()
            .find(|k| k.to_lowercase() == needle)
            .cloned()
    }

    /// True when `key` is a `Contract.field` reference to a state
    /// variable declared on an already-registered auxiliary contract.
    fn aux_has_state_variable(&self, key: &str) -> bool {
        match key.trim().split_once('.') {
            Some((contract, field)) => self
                .find_aux_contract(contract)
                .map(|aux| aux.state_variables.contains_key(field.trim()))
                .unwrap_or(false),
            None => false,
        }
    }

    fn draft_summary(&self) -> String {
        format!(
            "impact={} attacker={} properties={} value_flow={} setup_authority={} live_instances={} inevitability={} magnitude={} intent={} oracle={} setup{{desc={} sv={} contracts={}}} pre{{desc={} sv={} contracts={}}} post{{desc={} sv={} contracts={}}} sequence={}",
            self.impact.map(ImpactClass::as_str).unwrap_or("unset"),
            self.attacker
                .as_ref()
                .map(|attacker| attacker.privilege.as_str())
                .unwrap_or("unset"),
            self.properties.len(),
            self.value_flow.len(),
            self.setup_authority
                .as_ref()
                .map(|setup| setup.kind.as_str())
                .unwrap_or("unset"),
            self.live_instances.len(),
            self.inevitability.is_some() as u8,
            self.magnitude.is_some() as u8,
            self.intent.is_some() as u8,
            self.oracle
                .as_ref()
                .map(|oracle| oracle.kind.as_str())
                .unwrap_or("state_invariant"),
            !self.setup.description.is_empty() as u8,
            self.setup.states_variables.len(),
            self.setup.contracts.len(),
            !self.pre_attack.description.is_empty() as u8,
            self.pre_attack.states_variables.len(),
            self.pre_attack.contracts.len(),
            !self.post_attack.description.is_empty() as u8,
            self.post_attack.states_variables.len(),
            self.post_attack.contracts.len(),
            self.sequence.len(),
        )
    }
}

#[derive(Clone)]
pub struct DraftHandle(pub Arc<Mutex<SpecDraft>>);

impl std::fmt::Debug for DraftHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DraftHandle").finish_non_exhaustive()
    }
}

impl DraftHandle {
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(SpecDraft::default())))
    }

    pub async fn snapshot(&self) -> SpecDraft {
        self.0.lock().await.clone()
    }

    /// Assemble the spec-builder agent's full tool box over this draft.
    /// Keeps each tool's fields private to this module; the caller only
    /// gets the `ToolBox` and keeps this `DraftHandle` to read the
    /// committed specs back.
    ///
    /// `scope_gate` is the engagement's program scope, when the run declared
    /// one. It rides on the commit tool alone: every other tool builds the
    /// draft, and the gate is about what a *finished* specification claims.
    pub fn build_tool_box(
        &self,
        project_index: &Arc<ProjectIndex>,
        max_specs_per_link: usize,
        scope_gate: Option<Arc<ScopeGate>>,
        link_source: LinkSource,
        economic_gate: Option<Arc<EconomicGate>>,
        chain: Option<Arc<ChainReader>>,
        patterns: Option<Arc<PatternLibrary>>,
        require_composition: bool,
    ) -> ToolBox {
        let mut tools = ToolBox::new();
        tools.add_tool(UpdateStateDescriptionTool {
            draft: self.clone(),
        });
        tools.add_tool(AddStateVariableConstraintTool {
            draft: self.clone(),
            project: project_index.clone(),
        });
        tools.add_tool(AddContractConstraintTool {
            draft: self.clone(),
            project: project_index.clone(),
        });
        tools.add_tool(SetCallSequenceTool {
            draft: self.clone(),
            project: project_index.clone(),
        });
        tools.add_tool(AddAuxContractTool {
            draft: self.clone(),
            project: project_index.clone(),
        });
        tools.add_tool(AddAuxContractStateTool {
            draft: self.clone(),
        });
        tools.add_tool(SetImpactClassTool {
            draft: self.clone(),
        });
        tools.add_tool(SetAttackerModelTool {
            draft: self.clone(),
        });
        tools.add_tool(AddPropertyTool {
            draft: self.clone(),
            project: project_index.clone(),
        });
        tools.add_tool(SetValueFlowTool {
            draft: self.clone(),
        });
        tools.add_tool(SetSetupAuthorityTool {
            draft: self.clone(),
            project: project_index.clone(),
        });
        tools.add_tool(AddLiveInstanceTool {
            draft: self.clone(),
            chain: chain.clone(),
        });
        tools.add_tool(SetInevitabilityTool {
            draft: self.clone(),
        });
        tools.add_tool(SetMagnitudeTool {
            draft: self.clone(),
        });
        tools.add_tool(SetIntentEvidenceTool {
            draft: self.clone(),
            project: project_index.clone(),
        });
        tools.add_tool(SetOracleTool {
            draft: self.clone(),
        });
        tools.add_tool(DistinguishFromKnownIssueTool {
            draft: self.clone(),
            known: scope_gate
                .as_ref()
                .map(|gate| {
                    let scope = gate.scope();
                    scope
                        .known_issues
                        .iter()
                        .map(|issue| issue.id.clone())
                        .chain(
                            scope
                                .out_of_scope
                                .iter()
                                .filter(|rule| rule.enforce)
                                .map(|rule| rule.id.clone()),
                        )
                        .collect()
                })
                .unwrap_or_default(),
        });
        tools.add_tool(CommitSpecificationTool {
            draft: self.clone(),
            project: project_index.clone(),
            max_specs: max_specs_per_link,
            scope_gate,
            link_source,
            economic_gate,
            chain: chain.clone(),
        });
        // Read-only chain access is offered only when the run reaches a chain:
        // a tool that can only answer "not configured" costs a turn per link.
        if let Some(chain) = chain {
            tools.add_tool(ChainCallTool {
                chain: chain.clone(),
            });
            tools.add_tool(ChainStorageTool {
                chain: chain.clone(),
            });
            tools.add_tool(ChainLogsTool {
                chain: chain.clone(),
            });
            tools.add_tool(ChainAccountTool {
                chain: chain.clone(),
            });
            tools.add_tool(ResolveAddressTool { chain });
        }
        if let Some(library) = patterns.filter(|library| !library.is_empty()) {
            tools.add_tool(SearchPatternsTool { library });
        }
        tools.add_tool(FinalizeTool {
            draft: self.clone(),
            require_composition,
        });
        tools.add_tool(LookupCallGraphTool {
            project: project_index.clone(),
        });
        tools.add_tool(LookupStateVariableXrefsTool {
            project: project_index.clone(),
        });
        // The source readers are only registered when the agent does not
        // already carry the source in its prompt.
        tools.add_tool(ReadContractSourceTool {
            project: project_index.clone(),
        });
        tools.add_tool(ReadFunctionSourceTool {
            project: project_index.clone(),
        });
        tools
    }
}

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum SpecStage {
    Setup,
    PreAttack,
    PostAttack,
}

impl SpecStage {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Setup => "setup",
            Self::PreAttack => "pre_attack",
            Self::PostAttack => "post_attack",
        }
    }
}

fn stage_field_mut(draft: &mut SpecDraft, stage: SpecStage) -> &mut AuditStateSpecification {
    match stage {
        SpecStage::Setup => &mut draft.setup,
        SpecStage::PreAttack => &mut draft.pre_attack,
        SpecStage::PostAttack => &mut draft.post_attack,
    }
}

// ---- update_state_description ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct UpdateStateDescriptionArgs {
    /// One of `setup`, `pre_attack`, `post_attack`.
    stage: SpecStage,
    /// Free-form description of the general business logic governing this stage.
    description: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = UpdateStateDescriptionArgs,
    invoke = invoke,
    name = "update_state_description",
    description = "Set or replace the general-business-logic description of one stage of the current spec draft (setup / pre_attack / post_attack). Each stage holds exactly one description string. For stage=post_attack, describe how the attacked state violates a project expectation and the concrete bad consequence (drain / loss / DoS / bypass); the actual proof of the vulnerability goes into post_attack's state-variable / contract invariants, which commit_specification requires.",
)]
struct UpdateStateDescriptionTool {
    draft: DraftHandle,
}

impl UpdateStateDescriptionTool {
    async fn invoke(
        &self,
        args: UpdateStateDescriptionArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let mut guard = self.draft.0.lock().await;
        let stage_label = args.stage.as_str();
        let field = stage_field_mut(&mut guard, args.stage);
        field.description = args.description.trim().to_string();
        Ok(format!(
            "ok: updated {stage_label} description ({} chars). Draft: {}",
            field.description.len(),
            guard.draft_summary()
        ))
    }
}

// ---- add_state_variable_constraint ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct AddStateVariableConstraintArgs {
    stage: SpecStage,
    /// Identifier for the state variable. Prefer `Contract.fieldName` form
    /// when the same name lives on multiple contracts. For a variable on
    /// an auxiliary contract you must use the `AuxContract.fieldName` form,
    /// and that contract + field must already be declared with
    /// `add_aux_contract` / `add_aux_contract_state`.
    key: String,
    /// Constraint text: either a value, an invariant, or "must be initialized".
    constraint: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = AddStateVariableConstraintArgs,
    invoke = invoke,
    name = "add_state_variable_constraint",
    description = "Insert or overwrite one state-variable constraint on the current draft's stage. The key must be a real project state variable (bare name, or `Contract.fieldName` to disambiguate) OR a `AuxContract.fieldName` declared via add_aux_contract_state. Unknown identifiers are rejected and nothing is added.",
)]
struct AddStateVariableConstraintTool {
    draft: DraftHandle,
    project: Arc<ProjectIndex>,
}

impl AddStateVariableConstraintTool {
    async fn invoke(
        &self,
        args: AddStateVariableConstraintArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let key = args.key.trim().to_string();
        let mut guard = self.draft.0.lock().await;
        if !guard.aux_has_state_variable(&key) && !self.project.has_state_variable(&key) {
            return Ok(format!(
                "error: `{key}` is not a project state variable and not a declared auxiliary-contract field. \
                 Look it up with lookup_state_variable_xrefs (use `Contract.fieldName` to disambiguate), \
                 or, if it belongs to a helper contract the PoC deploys, declare it first with \
                 add_aux_contract + add_aux_contract_state and reference it as `AuxContract.fieldName`. \
                 Draft unchanged — fix the name and retry."
            ));
        }
        let stage_label = args.stage.as_str();
        let field = stage_field_mut(&mut guard, args.stage);
        field
            .states_variables
            .insert(key, args.constraint.trim().to_string());
        Ok(format!(
            "ok: {stage_label} state-variable constraints now {}. Draft: {}",
            field.states_variables.len(),
            guard.draft_summary()
        ))
    }
}

// ---- add_contract_constraint ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct AddContractConstraintArgs {
    stage: SpecStage,
    /// Contract name. Must be a real project contract/interface, or an
    /// auxiliary contract already declared with `add_aux_contract`.
    contract: String,
    /// Constraint text: e.g. "must be deployed and own >0 of TokenA", "admin
    /// is the test EOA".
    constraint: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = AddContractConstraintArgs,
    invoke = invoke,
    name = "add_contract_constraint",
    description = "Insert or overwrite one contract-level constraint on the current draft's stage (e.g. deployment, role assignment). The contract must be a real project contract/interface OR an auxiliary contract declared via add_aux_contract. Unknown names are rejected and nothing is added.",
)]
struct AddContractConstraintTool {
    draft: DraftHandle,
    project: Arc<ProjectIndex>,
}

impl AddContractConstraintTool {
    async fn invoke(
        &self,
        args: AddContractConstraintArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let contract = args.contract.trim().to_string();
        let mut guard = self.draft.0.lock().await;
        if !self.project.has_contract(&contract) && guard.find_aux_contract(&contract).is_none() {
            return Ok(format!(
                "error: `{contract}` is not a project contract/interface and not a declared auxiliary contract. \
                 Use a real project name (check with lookup_call_graph), or, if the PoC deploys it, \
                 declare it first with add_aux_contract. Draft unchanged — fix the name and retry."
            ));
        }
        let stage_label = args.stage.as_str();
        let field = stage_field_mut(&mut guard, args.stage);
        field
            .contracts
            .insert(contract, args.constraint.trim().to_string());
        Ok(format!(
            "ok: {stage_label} contract constraints now {}. Draft: {}",
            field.contracts.len(),
            guard.draft_summary()
        ))
    }
}

// ---- set_call_sequence ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct CallStepArg {
    /// Contract whose function is being called.
    contract: String,
    /// Function name (no parameter list).
    function: String,
    /// One-sentence intention of the call in this scenario.
    intention: String,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct SetCallSequenceArgs {
    /// Replaces the entire call sequence on the current draft.
    steps: Vec<CallStepArg>,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = SetCallSequenceArgs,
    invoke = invoke,
    name = "set_call_sequence",
    description = "Replace the current draft's core call sequence with the provided ordered list of (contract, function, intention) steps. Every step's contract must be a real project contract/interface (its function is grounded) OR an auxiliary contract declared via add_aux_contract. If any step references an unknown contract or function the whole call is rejected and the existing sequence is left unchanged.",
)]
struct SetCallSequenceTool {
    draft: DraftHandle,
    project: Arc<ProjectIndex>,
}

impl SetCallSequenceTool {
    async fn invoke(
        &self,
        args: SetCallSequenceArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        if args.steps.is_empty() {
            return Ok("error: refusing to set an empty call sequence; commit_specification needs at least one core call".to_string());
        }
        let mut guard = self.draft.0.lock().await;
        // Validate every step before mutating anything.
        let mut problems: Vec<String> = Vec::new();
        for (i, step) in args.steps.iter().enumerate() {
            let contract = step.contract.trim();
            let function = step.function.trim();
            if guard.find_aux_contract(contract).is_some() {
                // Auxiliary contract: its functions are synthesized by the
                // harness, so accept any function name.
                continue;
            }
            if !self.project.has_contract(contract) {
                problems.push(format!(
                    "step {}: contract `{contract}` is not a project contract/interface and not a declared auxiliary contract",
                    i + 1
                ));
            } else if !self.project.contract_has_function(contract, function) {
                problems.push(format!(
                    "step {}: `{contract}` has no function `{function}` (own or inherited)",
                    i + 1
                ));
            }
        }
        if !problems.is_empty() {
            return Ok(format!(
                "error: call sequence rejected, existing sequence unchanged. \
                 Fix these and resend the full sequence:\n- {}\n\
                 Verify names with lookup_call_graph, or declare helper contracts with add_aux_contract first.",
                problems.join("\n- ")
            ));
        }
        guard.sequence = args
            .steps
            .into_iter()
            .map(|step| SuqenceCallStep {
                contract: step.contract.trim().to_string(),
                function: step.function.trim().to_string(),
                intention: step.intention.trim().to_string(),
            })
            .collect();
        Ok(format!(
            "ok: call sequence now {} step(s). Draft: {}",
            guard.sequence.len(),
            guard.draft_summary()
        ))
    }
}

// ---- add_aux_contract ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct AddAuxContractArgs {
    /// Name for the auxiliary contract the PoC will deploy (you choose it).
    /// Must NOT collide with a real project contract/interface.
    name: String,
    /// Why this contract exists in the attack, e.g. "reentrancy attacker that
    /// re-enters withdraw on receive()".
    purpose: String,
    /// When this contract impersonates an external protocol (a staking
    /// manager, an oracle, a bridge) rather than playing the attacker: the
    /// address or deployed name of the real one. Leave empty for attacker-side
    /// helpers.
    #[serde(default)]
    stands_in_for: Option<String>,
    /// Required with `stands_in_for`: how you checked the stand-in behaves
    /// like the real deployment — address, selectors, and the exact state
    /// transitions your sequence relies on.
    #[serde(default)]
    fidelity: Option<String>,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = AddAuxContractArgs,
    invoke = invoke,
    name = "add_aux_contract",
    description = "Declare a helper contract the PoC must deploy itself (attacker contract, malicious callback, mock token) that does NOT exist in the project source. Required before you can reference it (or its state) in constraints or the call sequence. The name must not collide with a real project contract/interface. If it impersonates an external protocol, set `stands_in_for` and `fidelity` — the commit gate refuses a stand-in for a dependency that is deployed on the fork (use the real one) and a stand-in with no fidelity note.",
)]
struct AddAuxContractTool {
    draft: DraftHandle,
    project: Arc<ProjectIndex>,
}

impl AddAuxContractTool {
    async fn invoke(
        &self,
        args: AddAuxContractArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let name = args.name.trim().to_string();
        if name.is_empty() {
            return Ok("error: auxiliary contract name is empty. Draft unchanged.".to_string());
        }
        if self.project.has_contract(&name) {
            return Ok(format!(
                "error: `{name}` is a real project contract/interface — don't redeclare it as auxiliary. \
                 Reference it directly, or pick a different name for your helper contract. Draft unchanged."
            ));
        }
        let mut guard = self.draft.0.lock().await;
        let entry = guard.auxiliary_contracts.entry(name.clone()).or_default();
        entry.purpose = args.purpose.trim().to_string();
        entry.stands_in_for = args
            .stands_in_for
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        entry.fidelity = args
            .fidelity
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        Ok(format!(
            "ok: auxiliary contract `{name}` declared ({} total). Add its state with add_aux_contract_state.",
            guard.auxiliary_contracts.len()
        ))
    }
}

// ---- add_aux_contract_state ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct AddAuxContractStateArgs {
    /// Name of an auxiliary contract already declared with add_aux_contract.
    contract: String,
    /// State variable name on that auxiliary contract.
    name: String,
    /// Solidity type, e.g. `uint256`, `address`, `mapping(address => uint256)`.
    solidity_type: String,
    /// What this variable is for in the attack scenario.
    purpose: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = AddAuxContractStateArgs,
    invoke = invoke,
    name = "add_aux_contract_state",
    description = "Declare one state variable (name + Solidity type + purpose) on an auxiliary contract previously declared with add_aux_contract. Reference it later as `AuxContract.fieldName`. Fails if the auxiliary contract was not declared first.",
)]
struct AddAuxContractStateTool {
    draft: DraftHandle,
}

impl AddAuxContractStateTool {
    async fn invoke(
        &self,
        args: AddAuxContractStateArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let var_name = args.name.trim().to_string();
        let solidity_type = args.solidity_type.trim().to_string();
        if var_name.is_empty() || solidity_type.is_empty() {
            return Ok(
                "error: state variable name and solidity_type are both required. Draft unchanged."
                    .to_string(),
            );
        }
        let mut guard = self.draft.0.lock().await;
        let Some(key) = guard.aux_contract_key(&args.contract) else {
            return Ok(format!(
                "error: auxiliary contract `{}` is not declared yet; call add_aux_contract first. Draft unchanged.",
                args.contract.trim()
            ));
        };
        let aux = guard.auxiliary_contracts.get_mut(&key).expect("key exists");
        aux.state_variables.insert(
            var_name.clone(),
            AuxStateVariable {
                solidity_type,
                purpose: args.purpose.trim().to_string(),
            },
        );
        Ok(format!(
            "ok: auxiliary contract `{key}` now has {} state variable(s). Reference as `{key}.{var_name}`.",
            aux.state_variables.len()
        ))
    }
}

// ---- set_impact_class ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct SetImpactClassArgs {
    /// What the exploit takes away, as one of: direct_theft_of_funds |
    /// permanent_freeze | temporary_freeze | protocol_insolvency |
    /// protocol_dos | griefing | no_value_at_risk.
    class: ImpactClass,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = SetImpactClassArgs,
    invoke = invoke,
    description = "Declare what this exploit takes away, before committing the specification. Required: a spec without a declared class is refused, and `no_value_at_risk` is refused too — a behavioural divergence with nothing of value at stake is declined by every program, so a proof for it costs a full harness run and can never be submitted. Pick the class from the loss `post_attack` describes, and make `post_attack.description` state that loss in the class's own terms.",
    name = "set_impact_class",
)]
struct SetImpactClassTool {
    draft: DraftHandle,
}

impl SetImpactClassTool {
    async fn invoke(
        &self,
        args: SetImpactClassArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        if !args.class.is_reportable() {
            return Ok(format!(
                "error: `{}` is not reportable — a real divergence with nothing of value at stake is \
                 declined by every program that pays for losses, so a proof for it can only be \
                 discarded. Either find the loss that makes this direction matter (which contract or \
                 user ends up short, and by how much) and declare THAT class, or abandon this \
                 direction: call finalize with an abandoned status and say what you ruled out. Draft \
                 unchanged.",
                args.class.as_str()
            ));
        }
        let mut guard = self.draft.0.lock().await;
        guard.impact = Some(args.class);
        Ok(format!(
            "ok: impact class is `{}`. Draft: {}",
            args.class.as_str(),
            guard.draft_summary()
        ))
    }
}

// ---- set_attacker_model ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct SetAttackerModelArgs {
    /// Who drives the sequence: `unprivileged` | `privileged` |
    /// `privilege_escalation`.
    privilege: AttackerPrivilege,
    /// The authority the caller already holds, when `privilege` is
    /// `privileged` (e.g. `owner()`, `GUARDIAN_ROLE`, `onlyOperator`). Required
    /// for that one value, ignored for the others.
    #[serde(default)]
    role: Option<String>,
    /// One sentence on how the caller comes to be in that position — "any
    /// holder of the token can call", "the sequence deploys its own contract
    /// and calls it", "the guardian role is granted by the protocol at
    /// deployment".
    note: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = SetAttackerModelArgs,
    invoke = invoke,
    description = "Declare who carries out this exploit, before committing the specification. Required whenever the engagement declares an audit-program scope: a program pays for impacts an unprivileged actor can reach, and an exploit that needs a role the protocol already grants is not a finding no matter how clean the proof is. Declare `privileged` only when the sequence genuinely needs that authority, and name it in `role`.",
    name = "set_attacker_model",
)]
struct SetAttackerModelTool {
    draft: DraftHandle,
}

impl SetAttackerModelTool {
    async fn invoke(
        &self,
        args: SetAttackerModelArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let note = args.note.trim().to_string();
        if note.is_empty() {
            return Ok(
                "error: `note` is required — say how the caller comes to be in that position \
                 (a holder of the token, a permissionless entry point, an account the sequence \
                 grants itself). Draft unchanged."
                    .to_string(),
            );
        }
        let role = args
            .role
            .map(|role| role.trim().to_string())
            .filter(|role| !role.is_empty());
        if args.privilege == AttackerPrivilege::Privileged && role.is_none() {
            return Ok(
                "error: `privilege=privileged` needs `role` — name the authority the exploit \
                 relies on (e.g. `owner()`, `GUARDIAN_ROLE`). If nothing already holds it, the \
                 model is `unprivileged`; if the sequence takes it, it is `privilege_escalation`. \
                 Draft unchanged."
                    .to_string(),
            );
        }
        let mut guard = self.draft.0.lock().await;
        guard.attacker = Some(AttackerModel {
            privilege: args.privilege,
            role,
            note,
        });
        Ok(format!(
            "ok: attacker model is `{}`. Draft: {}",
            args.privilege.as_str(),
            guard.draft_summary()
        ))
    }
}

// ---- add_property ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct AddPropertyArgs {
    /// `agrees_with_actual` | `branch_expected` | `monotone` | `round_trip`.
    kind: PropertyKind,
    /// The project function the claim is about, as `Contract.function`
    /// (e.g. `LamboRebalanceOnUniwap._getTokenInOut`). Grounded against the
    /// project: an unknown contract or function is rejected.
    target: String,
    /// The claim in one sentence, in project terms: what must hold, and for
    /// which inputs. This is what the harness turns into an assertion, so state
    /// it as something a test can check (`the preview's tokenIn/amountIn must
    /// equal what the swap actually does for the same pool state`).
    claim: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = AddPropertyArgs,
    invoke = invoke,
    description = "Add a functional claim about a computation to the current spec draft. Use this instead of a post_attack state invariant when the defect is arithmetic or control-flow rather than a drained balance: a formula that assumes the wrong curve, a ratio or rounding step, a branch/selector that picks the wrong side. A specification with at least one property may commit without any post_attack invariant. Repeatable: each call appends one claim.",
    name = "add_property",
)]
struct AddPropertyTool {
    draft: DraftHandle,
    project: Arc<ProjectIndex>,
}

impl AddPropertyTool {
    async fn invoke(
        &self,
        args: AddPropertyArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let claim = args.claim.trim().to_string();
        if claim.is_empty() {
            return Ok("error: claim is required — state, in one sentence, what must hold and for                        which inputs. Draft unchanged."
                .to_string());
        }
        let target = args.target.trim().to_string();
        let Some((contract, function)) = target.split_once('.') else {
            return Ok(format!(
                "error: target must be `Contract.function`, got `{target}`. Draft unchanged."
            ));
        };
        let (contract, function) = (contract.trim(), function.trim());
        if !self.project.has_contract(contract) {
            return Ok(format!(
                "error: `{contract}` is not a project contract or interface — a property must be                  about the project's own code. Find the real name with `lookup_call_graph`. Draft                  unchanged."
            ));
        }
        if !self.project.contract_has_function(contract, function) {
            return Ok(format!(
                "error: `{contract}` has no function `{function}` — check the signature with                  `lookup_call_graph` or `read_function_source`. Draft unchanged."
            ));
        }
        let mut guard = self.draft.0.lock().await;
        guard.properties.push(SpecProperty {
            kind: args.kind,
            target,
            claim,
        });
        Ok(format!(
            "ok: property `{}` recorded ({} in draft). Draft: {}",
            args.kind.as_str(),
            guard.properties.len(),
            guard.draft_summary()
        ))
    }
}

// ---- set_value_flow ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct SetValueFlowArgs {
    /// One row per party: at least the attacker and whoever ends up short.
    /// Replaces any earlier table.
    rows: Vec<ActorDelta>,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = SetValueFlowArgs,
    invoke = invoke,
    name = "set_value_flow",
    description = "Declare who gains and who loses, per party (attacker, victim, other_holders, protocol, fee_recipient, third_party), in which asset, and through which code path. Required before commit. The gate refuses a table where nobody but the attacker is harmed (a donation or self-inflicted forfeit is not a finding), and a `direct_theft_of_funds` claim with no attacker row that gains.",
)]
struct SetValueFlowTool {
    draft: DraftHandle,
}

impl SetValueFlowTool {
    async fn invoke(
        &self,
        args: SetValueFlowArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        if args.rows.is_empty() {
            return Ok("error: the table is empty — add one row per party. Draft unchanged.".to_string());
        }
        if let Some(row) = args.rows.iter().find(|row| row.mechanism.trim().is_empty()) {
            return Ok(format!(
                "error: the `{}` row has no mechanism — name the code path that moves the value. \
                 Draft unchanged.",
                row.role.as_str()
            ));
        }
        let mut guard = self.draft.0.lock().await;
        guard.value_flow = args.rows;
        let rendered: Vec<String> = guard
            .value_flow
            .iter()
            .map(|row| format!("{} {} {}", row.role.as_str(), row.direction.as_str(), row.asset))
            .collect();
        Ok(format!(
            "ok: value flow [{}]. Draft: {}",
            rendered.join("; "),
            guard.draft_summary()
        ))
    }
}

// ---- set_setup_authority ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct SetSetupAuthorityArgs {
    /// `permissionless` (any participant creates it through public entry
    /// points), `live_state` (it already exists on the deployed system — add
    /// a live instance), or `privileged` (it needs owner / deployer / manager
    /// / timelock authority).
    kind: SetupAuthorityKind,
    /// The project functions that create the setup state, as
    /// `Contract.function`. Their source is checked for access guards.
    #[serde(default)]
    calls: Vec<String>,
    /// The authority needed, when `kind` is `privileged`.
    #[serde(default)]
    role: Option<String>,
    /// One sentence on how the setup state comes about.
    note: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = SetSetupAuthorityArgs,
    invoke = invoke,
    name = "set_setup_authority",
    description = "Declare who can create the `setup` state your sequence starts from — separate from who drives the sequence. A configuration only the deployer / owner / a timelock can install makes the direction privileged even if every sequence step is public. Required before commit. `permissionless` calls are checked against the project source: a function behind an `only*` modifier or a sender check is refused as permissionless.",
)]
struct SetSetupAuthorityTool {
    draft: DraftHandle,
    project: Arc<ProjectIndex>,
}

impl SetSetupAuthorityTool {
    async fn invoke(
        &self,
        args: SetSetupAuthorityArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let note = args.note.trim().to_string();
        if note.is_empty() {
            return Ok("error: `note` is required — say how the setup state comes about. Draft unchanged.".to_string());
        }
        let calls: Vec<String> = args
            .calls
            .iter()
            .map(|c| c.trim().to_string())
            .filter(|c| !c.is_empty())
            .collect();
        for call in &calls {
            let known = call
                .split_once('.')
                .is_some_and(|(contract, function)| {
                    self.project.contract_has_function(contract.trim(), function.trim())
                });
            if !known {
                return Ok(format!(
                    "error: `{call}` is not a project function — use `Contract.function` from \
                     `lookup_call_graph`. Draft unchanged."
                ));
            }
        }
        let role = args.role.map(|r| r.trim().to_string()).filter(|r| !r.is_empty());
        if args.kind == SetupAuthorityKind::Privileged && role.is_none() {
            return Ok("error: `kind=privileged` needs `role`. Draft unchanged.".to_string());
        }
        let mut guard = self.draft.0.lock().await;
        guard.setup_authority = Some(SetupAuthority {
            kind: args.kind,
            calls,
            role,
            note,
        });
        Ok(format!(
            "ok: setup authority is `{}`. Draft: {}",
            args.kind.as_str(),
            guard.draft_summary()
        ))
    }
}

// ---- add_live_instance ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct AddLiveInstanceArgs {
    /// Chain id; omit for the run's default chain.
    #[serde(default)]
    chain_id: Option<u64>,
    /// Address, or a contract name the scope / deployment artifacts resolve.
    target: String,
    /// `cast`-style signature with return types, e.g.
    /// `balanceOf(address)(uint256)` or `getFeeManagerConfigData(address)(bytes)`.
    signature: String,
    #[serde(default)]
    args: Vec<String>,
    /// `eq` | `ne` | `gt` | `gte` | `lt` | `lte` | `nonzero` | `contains`.
    comparison: Comparison,
    /// The value the read must compare against (ignored for `nonzero`).
    #[serde(default)]
    expected: String,
    /// Why this read shows the setup state exists.
    note: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = AddLiveInstanceArgs,
    invoke = invoke,
    name = "add_live_instance",
    description = "Record a deployed instance that is in your `setup` state, as a read against the fork: call `signature` on `target` and compare. When the run has a chain, the read executes now and its result is returned — a read that does not hold means the precondition is a state nobody is in, and commit will refuse it. Repeatable.",
)]
struct AddLiveInstanceTool {
    draft: DraftHandle,
    chain: Option<Arc<ChainReader>>,
}

impl AddLiveInstanceTool {
    async fn invoke(
        &self,
        args: AddLiveInstanceArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let note = args.note.trim().to_string();
        if note.is_empty() || args.target.trim().is_empty() || args.signature.trim().is_empty() {
            return Ok("error: `target`, `signature` and `note` are required. Draft unchanged.".to_string());
        }
        let mut instance = LiveInstance {
            chain_id: args.chain_id,
            target: args.target.trim().to_string(),
            signature: args.signature.trim().to_string(),
            args: args.args,
            comparison: args.comparison,
            expected: args.expected.trim().to_string(),
            note,
            check: None,
        };
        let observation = match &self.chain {
            Some(chain) => match chain.check(&instance).await {
                Ok(check) => {
                    let line = format!(
                        "read returned `{}` at block {} → {} {} `{}` {}",
                        check.observed,
                        check.block.map(|b| b.to_string()).unwrap_or_else(|| "head".to_string()),
                        instance.signature,
                        instance.comparison.as_str(),
                        instance.expected,
                        if check.holds { "HOLDS" } else { "DOES NOT HOLD" }
                    );
                    instance.check = Some(check);
                    line
                }
                Err(reason) => {
                    return Ok(format!(
                        "error: the read failed ({reason}). Fix target/signature/args. Draft unchanged."
                    ));
                }
            },
            None => "no chain in this run — recorded unverified".to_string(),
        };
        let mut guard = self.draft.0.lock().await;
        guard.live_instances.push(instance);
        Ok(format!(
            "ok: live instance #{} recorded; {observation}. Draft: {}",
            guard.live_instances.len(),
            guard.draft_summary()
        ))
    }
}

// ---- set_inevitability ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct SetInevitabilityArgs {
    /// Why the setup state must arise without anyone's permission — e.g. "any
    /// depositor creates a pending request in one call to `requestWithdraw`".
    argument: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = SetInevitabilityArgs,
    invoke = invoke,
    name = "set_inevitability",
    description = "For a `permissionless` setup with no live instance: argue why any participant can bring the setup state about (name the public entry point and why nothing blocks it). The alternative to `add_live_instance`; a plausible-but-absent state with neither is refused at commit.",
)]
struct SetInevitabilityTool {
    draft: DraftHandle,
}

impl SetInevitabilityTool {
    async fn invoke(
        &self,
        args: SetInevitabilityArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let argument = args.argument.trim().to_string();
        if argument.len() < 20 {
            return Ok("error: the argument is too short to check — name the entry point and why nothing blocks it. Draft unchanged.".to_string());
        }
        let mut guard = self.draft.0.lock().await;
        guard.inevitability = Some(argument);
        Ok(format!("ok: inevitability recorded. Draft: {}", guard.draft_summary()))
    }
}

// ---- set_magnitude ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct SetMagnitudeArgs {
    /// Amount at risk in the asset's units, as a decimal string.
    amount: String,
    asset: String,
    /// USD value, when a price exists (read it on the fork when you can).
    #[serde(default)]
    usd: Option<f64>,
    /// Where the number comes from: a live balance, TVL, a bound.
    basis: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = SetMagnitudeArgs,
    invoke = invoke,
    name = "set_magnitude",
    description = "Declare how much the exploit moves. Required for theft, insolvency and permanent freeze. A USD value below the run's loss floor is refused at commit: a bounded, cent-sized payout is informational however real the mechanism is.",
)]
struct SetMagnitudeTool {
    draft: DraftHandle,
}

impl SetMagnitudeTool {
    async fn invoke(
        &self,
        args: SetMagnitudeArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        if args.amount.trim().is_empty() || args.asset.trim().is_empty() || args.basis.trim().is_empty() {
            return Ok("error: `amount`, `asset` and `basis` are required. Draft unchanged.".to_string());
        }
        if args.usd.is_some_and(|usd| !usd.is_finite() || usd < 0.0) {
            return Ok("error: `usd` must be a non-negative number. Draft unchanged.".to_string());
        }
        let mut guard = self.draft.0.lock().await;
        guard.magnitude = Some(LossMagnitude {
            amount: args.amount.trim().to_string(),
            asset: args.asset.trim().to_string(),
            usd: args.usd,
            basis: args.basis.trim().to_string(),
        });
        Ok(format!("ok: magnitude recorded. Draft: {}", guard.draft_summary()))
    }
}

// ---- set_intent_evidence ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct SetIntentEvidenceArgs {
    /// `Contract.function` whose code or comments carry the promise.
    function: String,
    /// The comment, NatSpec line or `require` the promise rests on — copied
    /// verbatim from the source.
    quote: String,
    /// Why the nearest documentation does not already bless the state your
    /// sequence reaches.
    why_not_by_design: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = SetIntentEvidenceArgs,
    invoke = invoke,
    name = "set_intent_evidence",
    description = "Quote, verbatim, the comment / NatSpec / require that establishes the promise your post_attack breaks, and say why the nearest documentation does not already describe the reached state as intended. The quote is checked against the contract's source now; a paraphrase is refused. Required before commit.",
)]
struct SetIntentEvidenceTool {
    draft: DraftHandle,
    project: Arc<ProjectIndex>,
}

impl SetIntentEvidenceTool {
    async fn invoke(
        &self,
        args: SetIntentEvidenceArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let function = args.function.trim().to_string();
        let quote = args.quote.trim().to_string();
        let why = args.why_not_by_design.trim().to_string();
        if quote.is_empty() || why.is_empty() {
            return Ok("error: `quote` and `why_not_by_design` are required. Draft unchanged.".to_string());
        }
        let contract = function.split_once('.').map(|(c, _)| c).unwrap_or(&function).trim();
        if !self.project.has_contract(contract) {
            return Ok(format!("error: `{contract}` is not a project contract. Draft unchanged."));
        }
        if !self.project.source_contains(contract, &quote) {
            return Ok(format!(
                "error: that text is not in `{contract}`'s source (or its ancestors'). Copy it \
                 verbatim — read it with `read_contract_source`. Draft unchanged."
            ));
        }
        let mut guard = self.draft.0.lock().await;
        guard.intent = Some(IntentEvidence {
            function,
            quote,
            why_not_by_design: why,
        });
        Ok(format!("ok: intent evidence recorded (quote found in source). Draft: {}", guard.draft_summary()))
    }
}

// ---- chain_call ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct ChainCallArgs {
    /// Chain id; omit for the run's default chain.
    #[serde(default)]
    chain_id: Option<u64>,
    /// Address, or a contract name the scope / deployment artifacts resolve.
    target: String,
    /// `cast`-style signature with return types, e.g. `totalAssets()(uint256)`.
    signature: String,
    #[serde(default)]
    args: Vec<String>,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = ChainCallArgs,
    invoke = invoke,
    name = "chain_call",
    description = "Read-only `eth_call` against the deployed system at the run's pinned block. Use it to check what live positions actually hold, how a deployed fund is configured, what an external dependency returns — before you write a setup the chain is not in.",
)]
struct ChainCallTool {
    chain: Arc<ChainReader>,
}

impl ChainCallTool {
    async fn invoke(
        &self,
        args: ChainCallArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let (chain_id, address) = match self.chain.resolve(args.chain_id, &args.target) {
            Ok(resolved) => resolved,
            Err(reason) => return Ok(format!("error: {reason}")),
        };
        let query = ChainQuery::Call {
            address: address.clone(),
            signature: args.signature.trim().to_string(),
            args: args.args,
        };
        Ok(match self.chain.run(chain_id, &query).await {
            Ok(out) => format!("chain {chain_id} {address} `{}` → {out}", args.signature.trim()),
            Err(reason) => format!("error: {reason}"),
        })
    }
}

// ---- chain_storage ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct ChainStorageArgs {
    #[serde(default)]
    chain_id: Option<u64>,
    target: String,
    /// Slot number, decimal or 0x-hex.
    slot: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = ChainStorageArgs,
    invoke = invoke,
    name = "chain_storage",
    description = "Read one raw storage word at the pinned block. For private state with no getter; prefer `chain_call` when a getter exists.",
)]
struct ChainStorageTool {
    chain: Arc<ChainReader>,
}

impl ChainStorageTool {
    async fn invoke(
        &self,
        args: ChainStorageArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let (chain_id, address) = match self.chain.resolve(args.chain_id, &args.target) {
            Ok(resolved) => resolved,
            Err(reason) => return Ok(format!("error: {reason}")),
        };
        let query = ChainQuery::Storage {
            address: address.clone(),
            slot: args.slot.trim().to_string(),
        };
        Ok(match self.chain.run(chain_id, &query).await {
            Ok(out) => format!("chain {chain_id} {address} slot {} → {out}", args.slot.trim()),
            Err(reason) => format!("error: {reason}"),
        })
    }
}

// ---- chain_logs ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct ChainLogsArgs {
    #[serde(default)]
    chain_id: Option<u64>,
    target: String,
    /// Event signature, e.g. `FeeSettledForFund(address,uint8,address,uint256)`.
    event: String,
    #[serde(default)]
    from_block: Option<u64>,
    #[serde(default)]
    to_block: Option<u64>,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = ChainLogsArgs,
    invoke = invoke,
    name = "chain_logs",
    description = "Recent logs of one event on one address, bounded in block range and count (latest first). Use it to find live instances — which funds enabled a fee, which positions opened, whether anyone ever called a path.",
)]
struct ChainLogsTool {
    chain: Arc<ChainReader>,
}

impl ChainLogsTool {
    async fn invoke(
        &self,
        args: ChainLogsArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let (chain_id, address) = match self.chain.resolve(args.chain_id, &args.target) {
            Ok(resolved) => resolved,
            Err(reason) => return Ok(format!("error: {reason}")),
        };
        Ok(match self
            .chain
            .logs(chain_id, &address, args.event.trim(), args.from_block, args.to_block)
            .await
        {
            Ok(out) => out,
            Err(reason) => format!("error: {reason}"),
        })
    }
}

// ---- chain_account ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct ChainAccountArgs {
    #[serde(default)]
    chain_id: Option<u64>,
    target: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = ChainAccountArgs,
    invoke = invoke,
    name = "chain_account",
    description = "What lives at an address at the pinned block: code size, native balance, and the EIP-1967 implementation / admin / beacon it points at. Use it before reasoning about a proxy or an external dependency.",
)]
struct ChainAccountTool {
    chain: Arc<ChainReader>,
}

impl ChainAccountTool {
    async fn invoke(
        &self,
        args: ChainAccountArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let (chain_id, address) = match self.chain.resolve(args.chain_id, &args.target) {
            Ok(resolved) => resolved,
            Err(reason) => return Ok(format!("error: {reason}")),
        };
        Ok(match self.chain.account(chain_id, &address).await {
            Ok(out) => out,
            Err(reason) => format!("error: {reason}"),
        })
    }
}

// ---- resolve_address ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct ResolveAddressArgs {
    #[serde(default)]
    chain_id: Option<u64>,
    /// Contract name as the project or the scope names it.
    name: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = ResolveAddressArgs,
    invoke = invoke,
    name = "resolve_address",
    description = "List the deployed addresses a contract name resolves to on a chain — from the engagement's asset list first, then the repository's deployment artifacts. An address that appears in neither is outside what the program insures.",
)]
struct ResolveAddressTool {
    chain: Arc<ChainReader>,
}

impl ResolveAddressTool {
    async fn invoke(
        &self,
        args: ResolveAddressArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let chain_id = match self.chain.chain_or_default(args.chain_id) {
            Ok(chain_id) => chain_id,
            Err(reason) => return Ok(format!("error: {reason}")),
        };
        let found = self.chain.candidates(chain_id, args.name.trim());
        if found.is_empty() {
            return Ok(format!(
                "`{}` resolves to nothing on chain {chain_id}: neither the scope's asset list nor \
                 the deployment artifacts name it",
                args.name.trim()
            ));
        }
        Ok(found
            .into_iter()
            .map(|(name, address, source)| format!("- {name} → {address} ({source})"))
            .collect::<Vec<_>>()
            .join("\n"))
    }
}

// ---- set_oracle ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct SetOracleArgs {
    /// `state_invariant` | `value_delta` | `differential` | `bounds` | `liveness`.
    kind: OracleKind,
    /// What to measure: the assets making up the actor's value
    /// (`value_delta`), the functions compared (`differential` / `bounds`),
    /// the operations that must keep succeeding (`liveness`).
    #[serde(default)]
    measured: Vec<String>,
    /// `value_delta`: the victim metric. `differential`: the reference — a
    /// closed form, a prior deployed version's address, a sibling function.
    #[serde(default)]
    reference: Option<String>,
    /// Allowed error (`bounds`, `differential`) or profit threshold
    /// (`value_delta`).
    #[serde(default)]
    tolerance: Option<String>,
    /// Starting capital the actor may use (e.g. flash-loanable liquidity).
    #[serde(default)]
    capital: Option<String>,
    /// `fixed` (replay the sequence) or `fuzz` (search sequences and inputs
    /// against the oracle — use it when you know the identity that breaks but
    /// not the exact path).
    #[serde(default)]
    search: SearchMode,
    #[serde(default)]
    note: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = SetOracleArgs,
    invoke = invoke,
    name = "set_oracle",
    description = "Declare what the harness must measure to prove this specification. `value_delta`: the actor ends richer across `measured` assets while `reference` (the victim metric) falls — required for `direct_theft_of_funds`. `differential`: the target disagrees with `reference` beyond `tolerance`. `bounds`: a rounding / round-trip / monotonicity error exceeds `tolerance` or favours the caller. `liveness`: a legitimate call to a `measured` operation stops succeeding. `state_invariant` (default): post_attack fails. Set `search=fuzz` when the path must be searched rather than replayed — the harness then fuzzes entry points against the oracle.",
)]
struct SetOracleTool {
    draft: DraftHandle,
}

impl SetOracleTool {
    async fn invoke(
        &self,
        args: SetOracleArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let measured: Vec<String> = args
            .measured
            .iter()
            .map(|m| m.trim().to_string())
            .filter(|m| !m.is_empty())
            .collect();
        let clean = |value: Option<String>| value.map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        let reference = clean(args.reference);
        let tolerance = clean(args.tolerance);
        let missing = match args.kind {
            OracleKind::ValueDelta if measured.is_empty() => {
                Some("`measured` (the assets whose balances make up the actor's value)")
            }
            OracleKind::ValueDelta if reference.is_none() => {
                Some("`reference` (the victim metric that has to fall)")
            }
            OracleKind::Differential if reference.is_none() => {
                Some("`reference` (what the target is compared against)")
            }
            OracleKind::Differential | OracleKind::Bounds if tolerance.is_none() => {
                Some("`tolerance` (the error that counts as a violation)")
            }
            OracleKind::Liveness if measured.is_empty() => {
                Some("`measured` (the operations a legitimate caller must still complete)")
            }
            _ => None,
        };
        if let Some(missing) = missing {
            return Ok(format!(
                "error: `{}` needs {missing}. Draft unchanged.",
                args.kind.as_str()
            ));
        }
        let mut guard = self.draft.0.lock().await;
        guard.oracle = Some(OracleSpec {
            kind: args.kind,
            measured,
            reference,
            tolerance,
            capital: clean(args.capital),
            search: args.search,
            note: args.note.trim().to_string(),
        });
        Ok(format!(
            "ok: oracle `{}` ({}). Draft: {}",
            args.kind.as_str(),
            args.search.as_str(),
            guard.draft_summary()
        ))
    }
}

// ---- distinguish_from_known_issue ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct DistinguishFromKnownIssueArgs {
    /// The known issue's id, exactly as the refusal named it.
    id: String,
    /// The root cause that makes this defect a different one — not a
    /// different entry point or impact wording for the same mistake.
    difference: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = DistinguishFromKnownIssueArgs,
    invoke = invoke,
    name = "distinguish_from_known_issue",
    description = "When commit is refused because the draft matches a known issue (a prior audit's finding or a program's rejection) but your defect has a DIFFERENT root cause, record why. The gate then admits the draft and the reviewer reads your distinction. A different call path, a different impact label or a different amount for the same underlying mistake is NOT different — that is a duplicate; abandon it.",
)]
struct DistinguishFromKnownIssueTool {
    draft: DraftHandle,
    /// Ids the engagement's scope knows, so a typo is caught here.
    known: Vec<String>,
}

impl DistinguishFromKnownIssueTool {
    async fn invoke(
        &self,
        args: DistinguishFromKnownIssueArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let id = args.id.trim().to_string();
        let difference = args.difference.trim().to_string();
        if !self.known.iter().any(|known| known == &id) {
            return Ok(format!(
                "error: `{id}` is not a known issue of this engagement — use the id the refusal \
                 named. Draft unchanged."
            ));
        }
        if difference.len() < 40 {
            return Ok(
                "error: state the different root cause in a sentence or two (at least 40 \
                 characters). Draft unchanged."
                    .to_string(),
            );
        }
        let mut guard = self.draft.0.lock().await;
        guard.distinct_from.retain(|d| d.id != id);
        guard.distinct_from.push(KnownIssueDistinction { id: id.clone(), difference });
        Ok(format!(
            "ok: recorded why this is not `{id}`; the reviewer will read it. Commit again."
        ))
    }
}

// ---- search_patterns ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct SearchPatternsArgs {
    /// Words describing the mechanism you want a precedent for, e.g.
    /// `share price donation first depositor` or `mirror stale callback`.
    query: String,
    /// How many findings to return (default 5, at most 10).
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = SearchPatternsArgs,
    invoke = invoke,
    name = "search_patterns",
    description = "Search the historical findings this project's knowledge-graph mirror holds for precedents of a mechanism: how a similar identity was broken elsewhere, which failure paths were used. A pattern library to consult when a hypothesis needs one — not a list of things to reproduce. No match is itself a signal: a failure mode with no precedent is what this audit is after.",
)]
struct SearchPatternsTool {
    library: Arc<PatternLibrary>,
}

impl SearchPatternsTool {
    async fn invoke(
        &self,
        args: SearchPatternsArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        Ok(self
            .library
            .search(args.query.trim(), args.limit.unwrap_or(5).clamp(1, 10)))
    }
}

// ---- commit_specification ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct CommitSpecificationArgs {
    /// One- or two-sentence summary of the overall vulnerable behavior this
    /// spec captures (what the attacker does and what breaks). Required.
    summary: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = CommitSpecificationArgs,
    invoke = invoke,
    name = "commit_specification",
    description = "Snapshot the current draft as one finalized AuditSpecification and reset the draft for the next one. Requires a non-empty summary, at least one post_attack invariant (state-variable or contract constraint), and at least one call-sequence step.",
)]
struct CommitSpecificationTool {
    draft: DraftHandle,
    /// The project index, so the scope gate can tell the project's own
    /// contracts from the specification's own helpers.
    project: Arc<ProjectIndex>,
    max_specs: usize,
    /// The engagement's audit-program scope, when this run declared one. The
    /// gate runs here and nowhere else: this is the last moment before the
    /// specification costs a harness agent, a fuzzer and three graders, and the
    /// first moment its claims (impact, attacker, assets) exist to be checked.
    scope_gate: Option<Arc<ScopeGate>>,
    /// What kind of link produced this draft. It only changes how a refusal is
    /// phrased: an externally-supplied finding is a question someone asked, so
    /// the refusal has to say what the answer is, not just that the commit was
    /// rejected.
    link_source: LinkSource,
    /// The economic gate (value flow, setup authority, live instance,
    /// magnitude, stand-in fidelity, intent). Runs after the scope gate.
    economic_gate: Option<Arc<EconomicGate>>,
    /// The run's chain access, for the live-instance and stand-in checks.
    chain: Option<Arc<ChainReader>>,
}

impl CommitSpecificationTool {
    async fn invoke(
        &self,
        args: CommitSpecificationArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let summary = args.summary.trim().to_string();
        let mut guard = self.draft.0.lock().await;
        if guard.completed.len() >= self.max_specs {
            return Ok(format!(
                "error: max_specs_per_link={} reached; call finalize next",
                self.max_specs
            ));
        }
        if summary.is_empty() {
            return Ok(
                "error: summary is required — give a one- or two-sentence description of the overall vulnerable behavior this spec captures. Draft unchanged."
                    .to_string(),
            );
        }
        if guard.post_attack.states_variables.is_empty()
            && guard.post_attack.contracts.is_empty()
            && guard.properties.is_empty()
        {
            return Ok(
                "error: the specification proves nothing yet. Add a post_attack invariant \
                 (add_state_variable_constraint / add_contract_constraint with stage=post_attack) \
                 when the defect is a broken state, or a functional claim (add_property) when it \
                 is a miscomputation — a formula, a ratio, a rounding step, a wrong branch. Draft \
                 unchanged."
                    .to_string(),
            );
        }
        if guard.sequence.is_empty() {
            return Ok(
                "error: call sequence is empty; describe at least one core call before committing. Draft unchanged."
                    .to_string(),
            );
        }
        // Impact class, program scope, economic claim — applied
        // deterministically, in that order, by the same gates the regression
        // suite runs. A refusal here costs one turn; the same refusal
        // discovered at review time costs the whole pipeline and still lands
        // in the report.
        let mut spec = guard.build_spec(summary);
        let gates = CommitGates {
            scope: self.scope_gate.clone(),
            economic: self.economic_gate.clone(),
            chain: self.chain.clone(),
        };
        let economic_notes = match gates.admit(&mut spec, &self.project).await {
            Ok(notes) => notes,
            Err(refusal) => {
                // Keep what the fork said on the draft, so a retry after an
                // unrelated refusal does not re-read it.
                guard.live_instances = spec.live_instances.clone();
                let validation_note = match (self.link_source, refusal.id.starts_with("scope-")) {
                    (LinkSource::External, true) => {
                        " This link was supplied for validation, so the outcome to record is that \
                         the reported issue is not payable under this program's rules — a real \
                         defect it does not buy is still not a finding. Abandon the link with that \
                         as the reason rather than re-cutting the same claim."
                    }
                    _ => "",
                };
                return Ok(format!(
                    "error: {} [gate: {}]. Draft unchanged — fix it and commit again, or abandon \
                     the direction with finalize if no fix exists.{validation_note}",
                    refusal.clause, refusal.id
                ));
            }
        };
        guard.completed.push(spec);
        guard.reset_draft();
        let notes = if economic_notes.is_empty() {
            String::new()
        } else {
            format!(" Notes: {}", economic_notes.join("; "))
        };
        Ok(format!(
            "ok: committed spec #{} for this link. Draft has been reset.{notes}",
            guard.completed.len()
        ))
    }
}

// ---- finalize ----

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum FinalizeStatus {
    Completed,
    Abandoned,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct FinalizeArgs {
    /// `completed` if at least one spec was committed; `abandoned` otherwise.
    status: FinalizeStatus,
    /// Free-form summary of what you produced or why you abandoned. Required.
    summary: String,
    /// When abandoned, a one-sentence reason (e.g. "project has no AMM
    /// pool — historical pattern doesn't apply").
    #[serde(default)]
    abort_reason: Option<String>,
    /// When abandoning a target-derived (invariant) hypothesis: the multi-call
    /// sequences you checked. Each must chain at least two different
    /// functions in order (possibly across transactions or users) and say
    /// concretely why that ordering still preserves the identity. Showing
    /// that each writer preserves the identity on its own does not cover
    /// orderings, interleavings with other users, rounding accumulated over
    /// repeated calls, or callbacks between steps.
    #[serde(default, deserialize_with = "lenient_probes")]
    #[schemars(with = "Vec<CompositionProbe>")]
    compositions_checked: Vec<CompositionProbe>,
}

/// One multi-call sequence the agent checked before abandoning.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct CompositionProbe {
    /// The calls in order, as `Contract.function` (at least two distinct).
    #[serde(default, deserialize_with = "lenient_calls")]
    #[schemars(with = "Vec<String>")]
    calls: Vec<String>,
    /// Why this sequence does not break the identity, citing the state each
    /// step reads and writes.
    why_it_holds: String,
}

/// The call list as models actually send it: an array, a JSON-encoded array,
/// `{"item": [...]}`, or one string joined by commas or `->`. A shape mismatch
/// here used to fail the whole `finalize` and burn the link's step budget on
/// retries.
fn lenient_calls<'de, D>(deserializer: D) -> std::result::Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = super::invariant_llm::lenient_strings(deserializer)?;
    Ok(raw
        .iter()
        .flat_map(|entry| entry.split("->").flat_map(|part| part.split(',')))
        .map(|call| call.trim().to_string())
        .filter(|call| !call.is_empty())
        .collect())
}

/// The probe list, accepted as an array, a JSON-encoded array, a single
/// object, or `{"item": [...]}`. Anything else reads as no probes, which the
/// composition gate then asks for again instead of rejecting the call.
fn lenient_probes<'de, D>(deserializer: D) -> std::result::Result<Vec<CompositionProbe>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let mut value = <serde_json::Value as serde::Deserialize>::deserialize(deserializer)?;
    if let serde_json::Value::String(text) = &value
        && let Ok(inner) = serde_json::from_str::<serde_json::Value>(text.trim())
    {
        value = inner;
    }
    if let serde_json::Value::Object(map) = &value
        && map.len() == 1
        && let Some(inner) = map.get("item")
    {
        value = inner.clone();
    }
    if value.is_object() {
        value = serde_json::Value::Array(vec![value]);
    }
    Ok(serde_json::from_value(value).unwrap_or_default())
}

/// Most times an `abandoned` finalize is sent back for missing composition
/// analysis before it is accepted anyway (and marked as such).
const MAX_COMPOSITION_REJECTIONS: usize = 2;

impl CompositionProbe {
    /// A probe counts when it chains at least two distinct functions and
    /// gives a non-trivial reason.
    fn is_substantive(&self) -> bool {
        let distinct: BTreeSet<String> = self
            .calls
            .iter()
            .map(|call| call.trim().to_ascii_lowercase())
            .filter(|call| !call.is_empty())
            .collect();
        distinct.len() >= 2 && self.why_it_holds.trim().len() >= 40
    }

    fn render(&self) -> String {
        format!("[{}] {}", self.calls.join(" -> "), self.why_it_holds.trim())
    }
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = FinalizeArgs,
    invoke = invoke,
    name = "finalize",
    description = "End the spec-generation run for this link. Use `completed` when at least one spec has been committed; otherwise `abandoned`. Once called, no further tool calls are honoured.",
)]
struct FinalizeTool {
    draft: DraftHandle,
    /// Require composition analysis before `abandoned` (target-derived links).
    require_composition: bool,
}

impl FinalizeTool {
    async fn invoke(
        &self,
        args: FinalizeArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let mut guard = self.draft.0.lock().await;
        let status = match args.status {
            FinalizeStatus::Completed => {
                if guard.completed.is_empty() {
                    return Ok(
                        "error: cannot finalize as completed without at least one committed spec; use status=abandoned"
                            .to_string(),
                    );
                }
                LinkSpecStatus::Built
            }
            FinalizeStatus::Abandoned => LinkSpecStatus::Abandoned,
        };
        let substantive: Vec<&CompositionProbe> = args
            .compositions_checked
            .iter()
            .filter(|probe| probe.is_substantive())
            .collect();
        let mut composition_note: Option<String> = None;
        if status == LinkSpecStatus::Abandoned && self.require_composition {
            if substantive.is_empty() {
                if guard.composition_rejections < MAX_COMPOSITION_REJECTIONS {
                    guard.composition_rejections += 1;
                    return Ok(format!(
                        "error: abandon refused ({}/{}): this hypothesis is an identity over the \
                         target's own state, and checking that each writer preserves it on its \
                         own does not cover how writers combine. Before abandoning, check at \
                         least one sequence that chains two or more different functions in order \
                         — e.g. writer A then writer B by another user, the same writer repeated \
                         so rounding accumulates, a call made while another is mid-flight \
                         (callback/hook), or one transaction setting up state the next one \
                         exploits — and pass it in `compositions_checked` as {{calls: \
                         [\"Contract.fnA\", \"Contract.fnB\", ...], why_it_holds: \"...\"}}. If a \
                         sequence looks plausible but you cannot settle it by reading, do not \
                         abandon: commit with `set_oracle(..., search=\"fuzz\")` so the harness \
                         searches it.",
                        guard.composition_rejections, MAX_COMPOSITION_REJECTIONS
                    ));
                }
                composition_note = Some(
                    "composition not analysed (accepted after repeated refusals)".to_string(),
                );
            } else {
                composition_note = Some(format!(
                    "compositions checked: {}",
                    substantive
                        .iter()
                        .map(|probe| probe.render())
                        .collect::<Vec<_>>()
                        .join(" | ")
                ));
            }
        }
        guard.final_status = Some(status);
        guard.final_summary = Some(args.summary);
        guard.abort_reason = args.abort_reason.or_else(|| {
            if status == LinkSpecStatus::Abandoned {
                Some("agent abandoned without explicit reason".to_string())
            } else {
                None
            }
        });
        if let (Some(reason), Some(note)) = (guard.abort_reason.as_mut(), composition_note) {
            reason.push_str(" — ");
            reason.push_str(&note);
        }
        Ok(format!(
            "ok: finalized as {} with {} committed spec(s). Stop now.",
            match status {
                LinkSpecStatus::Built => "completed",
                LinkSpecStatus::Abandoned => "abandoned",
            },
            guard.completed.len()
        ))
    }
}

// ---- lookup_call_graph ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub(crate) struct LookupCallGraphArgs {
    /// Contract or interface name (case-insensitive). Required.
    contract: String,
    /// Optional function name to focus on. When omitted, all functions of
    /// the contract are listed (no edges).
    #[serde(default)]
    function: Option<String>,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = LookupCallGraphArgs,
    invoke = invoke,
    name = "lookup_call_graph",
    description = "Return the call-graph context for a contract or function: outgoing edges, incoming edges, and source content. Pass just the contract to enumerate its functions; pass contract+function for full edge listing.",
)]
pub(crate) struct LookupCallGraphTool {
    pub(crate) project: Arc<ProjectIndex>,
}

impl LookupCallGraphTool {
    async fn invoke(
        &self,
        args: LookupCallGraphArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let contract_ids = self.project.lookup_contract(&args.contract);
        let interface_ids = self.project.lookup_interface(&args.contract);
        if contract_ids.is_empty() && interface_ids.is_empty() {
            return Ok(format!(
                "no contract or interface named `{}` found in this project",
                args.contract
            ));
        }
        let mut out = String::new();
        for id in contract_ids {
            if let Some(contract) = self.project.contract(id) {
                out.push_str(
                    &self
                        .project
                        .render_contract_lookup(contract, args.function.as_deref()),
                );
                out.push_str("\n---\n");
            }
        }
        for id in interface_ids {
            if let Some(interface) = self.project.interface(id) {
                out.push_str(
                    &self
                        .project
                        .render_interface_lookup(interface, args.function.as_deref()),
                );
                out.push_str("\n---\n");
            }
        }
        Ok(out)
    }
}

// ---- read_contract_source ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub(crate) struct ReadContractSourceArgs {
    /// Contract or interface name (case-insensitive).
    contract: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = ReadContractSourceArgs,
    invoke = invoke,
    name = "read_contract_source",
    description = "Return the full source text of a contract or interface. Expensive — prefer `read_function_source` for a single function body when you only need to inspect one function. Use this when you need surrounding file context (modifiers, imports, struct layouts).",
)]
pub(crate) struct ReadContractSourceTool {
    pub(crate) project: Arc<ProjectIndex>,
}

impl ReadContractSourceTool {
    async fn invoke(
        &self,
        args: ReadContractSourceArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let contract_ids = self.project.lookup_contract(&args.contract);
        let interface_ids = self.project.lookup_interface(&args.contract);
        if contract_ids.is_empty() && interface_ids.is_empty() {
            return Ok(format!(
                "no contract or interface named `{}` found in this project",
                args.contract
            ));
        }
        let mut out = String::new();
        for id in contract_ids {
            if let Some(contract) = self.project.contract(id) {
                self.project.push_contract_source(&mut out, contract);
            }
        }
        for id in interface_ids {
            if let Some(interface) = self.project.interface(id) {
                self.project.push_interface_source(&mut out, interface);
            }
        }
        Ok(out)
    }
}

// ---- read_function_source ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub(crate) struct ReadFunctionSourceArgs {
    /// Contract or interface name (case-insensitive).
    contract: String,
    /// Function name (case-insensitive). For overloads, all matching bodies are returned.
    function: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = ReadFunctionSourceArgs,
    invoke = invoke,
    name = "read_function_source",
    description = "Return the source text of a single function (preferred over `read_contract_source` for body inspection). Returns all overloads matching the name within the contract. For interfaces this returns the declaration line.",
)]
pub(crate) struct ReadFunctionSourceTool {
    pub(crate) project: Arc<ProjectIndex>,
}

impl ReadFunctionSourceTool {
    async fn invoke(
        &self,
        args: ReadFunctionSourceArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let target = args.function.to_lowercase();
        let mut out = String::new();
        let mut found = false;

        for cid in self.project.lookup_contract(&args.contract) {
            if let Some(contract) = self.project.contract(cid) {
                for f in contract
                    .functions
                    .iter()
                    .filter(|f| f.name.to_lowercase() == target)
                {
                    found = true;
                    out.push_str(&format!(
                        "## `{}::{}({})` @ {}:{} (id={})\n",
                        contract.name,
                        f.name,
                        f.args,
                        f.relative_file_path.display(),
                        f.loc.start_line,
                        f.id,
                    ));
                    if let Some(body) = &f.content {
                        out.push_str("```solidity\n");
                        out.push_str(body.trim_end());
                        out.push_str("\n```\n");
                    } else {
                        out.push_str("(no body — this function has no recorded source)\n");
                    }
                    if !f.calls.is_empty() {
                        out.push_str("outgoing calls:\n");
                        for call in &f.calls {
                            out.push_str(&format!(
                                "- -> {}\n",
                                self.project.describe_endpoint(call.to_id, call)
                            ));
                        }
                    }
                    out.push_str("---\n");
                }
            }
        }
        for iid in self.project.lookup_interface(&args.contract) {
            if let Some(interface) = self.project.interface(iid) {
                for f in interface
                    .functions
                    .iter()
                    .filter(|f| f.name.to_lowercase() == target)
                {
                    found = true;
                    out.push_str(&format!(
                        "## (interface) `{}::{}({})` @ {}:{} (id={})\n(declaration only)\n---\n",
                        interface.name,
                        f.name,
                        f.args,
                        f.relative_file_path.display(),
                        f.loc.start_line,
                        f.id,
                    ));
                }
            }
        }

        if !found {
            return Ok(format!(
                "no function named `{}` in contract or interface `{}`. Use `read_memory` to see available signatures.",
                args.function, args.contract
            ));
        }
        Ok(out)
    }
}

// ---- lookup_state_variable_xrefs ----

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub(crate) struct LookupStateVariableXrefsArgs {
    /// State-variable name (case-insensitive).
    state_variable: String,
    /// Optional contract scope (case-insensitive). When provided, only
    /// matches declared in or visible to this contract are returned.
    #[serde(default)]
    contract: Option<String>,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = LookupStateVariableXrefsArgs,
    invoke = invoke,
    name = "lookup_state_variable_xrefs",
    description = "Return cross-references for a state variable: declaring contract, all functions reading it, all functions writing it. Optionally constrain the search to a single contract scope.",
)]
pub(crate) struct LookupStateVariableXrefsTool {
    pub(crate) project: Arc<ProjectIndex>,
}

impl LookupStateVariableXrefsTool {
    async fn invoke(
        &self,
        args: LookupStateVariableXrefsArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let candidate_ids = self.project.lookup_state_var(&args.state_variable);
        if candidate_ids.is_empty() {
            return Ok(format!(
                "no state variable named `{}` found in this project",
                args.state_variable
            ));
        }
        let scope_contract_ids: Option<Vec<i32>> = args
            .contract
            .as_deref()
            .map(|name| self.project.lookup_contract(name));

        let mut out = String::new();
        for var_id in &candidate_ids {
            let Some(var) = self.project.state_variable(*var_id) else {
                continue;
            };
            let declaring_contract_id =
                self.project.declaring_contract_for_var.get(var_id).copied();
            // Apply scope filter: keep only when scope contract sees this var.
            if let Some(scope_ids) = &scope_contract_ids {
                let visible = scope_ids.iter().any(|cid| {
                    self.project
                        .visible_state_vars
                        .get(cid)
                        .map(|s| s.contains(var_id))
                        .unwrap_or(false)
                });
                if !visible {
                    continue;
                }
            }
            let declaring_contract_name = declaring_contract_id
                .and_then(|id| self.project.contract(id))
                .map(|c| c.name.as_str())
                .unwrap_or("?");
            out.push_str(&format!(
                "## {} {} (id={}) declared in `{}` @ {}:{}\n",
                var.type_name.trim(),
                var.name,
                var.id,
                declaring_contract_name,
                var.relative_file_path.display(),
                var.loc.start_line
            ));

            let xrefs: Vec<&FunctionStateVariable> = self
                .project
                .storage_graph
                .function_state_variables
                .iter()
                .filter(|fsv| fsv.state_variable_id == var.id)
                .collect();

            let (writes, reads): (Vec<&&FunctionStateVariable>, Vec<&&FunctionStateVariable>) =
                xrefs.iter().partition(|fsv| fsv.is_write);
            out.push_str(&format!(
                "writers ({}) and readers ({})\n",
                writes.len(),
                reads.len()
            ));
            if !writes.is_empty() {
                out.push_str("\nwriters:\n");
                for fsv in writes {
                    out.push_str(&format!(
                        "- {}\n",
                        self.project
                            .describe_function_xref(fsv.function_id, fsv.description.as_deref())
                    ));
                }
            }
            if !reads.is_empty() {
                out.push_str("\nreaders:\n");
                for fsv in reads {
                    out.push_str(&format!(
                        "- {}\n",
                        self.project
                            .describe_function_xref(fsv.function_id, fsv.description.as_deref())
                    ));
                }
            }
            out.push_str("\n---\n");
        }
        if out.is_empty() {
            return Ok(format!(
                "no state variable named `{}` matches the given contract scope",
                args.state_variable
            ));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod composition_tests {
    use super::*;

    fn abandon(compositions: &str) -> FinalizeArgs {
        serde_json::from_str(&format!(
            r#"{{"status":"abandoned","summary":"s","abort_reason":"every writer preserves it","compositions_checked":{compositions}}}"#
        ))
        .expect("args")
    }

    const PROBE: &str = r#"[{"calls":["Engine.lock","Engine.free"],"why_it_holds":"lock adds wad to ink and supply; free subtracts the same wad from both in one call"}]"#;

    #[test]
    fn composition_args_accept_the_shapes_models_actually_send() {
        let why = "lock adds wad to ink and supply; free subtracts the same wad from both";
        for calls in [
            r#""Engine.lock, Engine.free""#,
            r#""Engine.lock -> Engine.free""#,
            r#"{"item":["Engine.lock","Engine.free"]}"#,
            r#""[\"Engine.lock\",\"Engine.free\"]""#,
        ] {
            let args = abandon(&format!(r#"[{{"calls":{calls},"why_it_holds":"{why}"}}]"#));
            let probe = &args.compositions_checked[0];
            assert_eq!(probe.calls, ["Engine.lock", "Engine.free"], "{calls}");
            assert!(probe.is_substantive());
        }
        let wrapped = abandon(&format!(
            r#"{{"item":[{{"calls":["Engine.lock","Engine.free"],"why_it_holds":"{why}"}}]}}"#
        ));
        assert_eq!(wrapped.compositions_checked.len(), 1);
        assert!(abandon(r#""not json""#).compositions_checked.is_empty());
    }

    #[tokio::test]
    async fn abandon_without_composition_is_refused_then_accepted_after_cap() {
        let draft = DraftHandle::new();
        let tool = FinalizeTool { draft: draft.clone(), require_composition: true };
        for round in 1..=MAX_COMPOSITION_REJECTIONS {
            let out = tool.invoke(abandon("[]")).await.expect("invoke");
            assert!(out.starts_with("error: abandon refused"), "round {round}: {out}");
            assert!(draft.snapshot().await.final_status.is_none());
        }
        let out = tool.invoke(abandon("[]")).await.expect("invoke");
        assert!(out.starts_with("ok"), "{out}");
        let snap = draft.snapshot().await;
        assert!(snap.abort_reason.unwrap().contains("composition not analysed"));
    }

    #[tokio::test]
    async fn single_function_or_thin_probe_does_not_count() {
        let draft = DraftHandle::new();
        let tool = FinalizeTool { draft: draft.clone(), require_composition: true };
        let same_fn = r#"[{"calls":["Engine.lock","Engine.lock"],"why_it_holds":"lock repeated twice still adds the same wad to both sides"}]"#;
        assert!(tool.invoke(abandon(same_fn)).await.unwrap().starts_with("error"));
        let thin = r#"[{"calls":["Engine.lock","Engine.free"],"why_it_holds":"fine"}]"#;
        assert!(tool.invoke(abandon(thin)).await.unwrap().starts_with("error"));
    }

    #[tokio::test]
    async fn substantive_probe_is_accepted_and_recorded() {
        let draft = DraftHandle::new();
        let tool = FinalizeTool { draft: draft.clone(), require_composition: true };
        assert!(tool.invoke(abandon(PROBE)).await.unwrap().starts_with("ok"));
        let reason = draft.snapshot().await.abort_reason.unwrap();
        assert!(reason.contains("Engine.lock -> Engine.free"), "{reason}");
    }

    #[tokio::test]
    async fn not_required_on_mapper_links() {
        let draft = DraftHandle::new();
        let tool = FinalizeTool { draft: draft.clone(), require_composition: false };
        assert!(tool.invoke(abandon("[]")).await.unwrap().starts_with("ok"));
    }
}

#[cfg(test)]
mod grounding_tests {
    use super::*;
    use crate::scope::{ProgramScope, ScopeGate};
    use knowdit_repo_model::cg::{
        CallGraph, Contract, FileChunk, FileLocation, Function, Interface,
    };
    use knowdit_repo_model::inheritance::{ContractInherit, InheritanceGraph};
    use knowdit_repo_model::storage::{ContractVariable, StateVariable, StorageGraph};
    use std::path::{Path, PathBuf};

    fn loc() -> FileLocation {
        FileLocation {
            start_line: 1,
            start_column: 0,
            end_line: 2,
            end_column: 0,
        }
    }

    fn chunk() -> FileChunk {
        FileChunk {
            loc: loc(),
            content: String::new(),
        }
    }

    fn func(id: i32, name: &str) -> Function {
        Function {
            id,
            name: name.to_string(),
            args: String::new(),
            relative_file_path: PathBuf::from("src/X.sol"),
            loc: loc(),
            content: None,
            calls: vec![],
            description: None,
        }
    }

    fn contract(id: i32, name: &str, functions: Vec<Function>) -> Contract {
        Contract {
            id,
            name: name.to_string(),
            relative_file_path: PathBuf::from("src/X.sol"),
            chunk: chunk(),
            functions,
            description: None,
        }
    }

    fn state_var(id: i32, name: &str, ty: &str) -> StateVariable {
        StateVariable {
            id,
            name: name.to_string(),
            type_name: ty.to_string(),
            relative_file_path: PathBuf::from("src/X.sol"),
            loc: loc(),
            content: String::new(),
        }
    }

    /// Vault (own var `balances`, own fn `withdraw`) inherits Base (var
    /// `paused`, fn `pause`); plus interface IERC20 with `transfer`.
    fn project_index() -> Arc<ProjectIndex> {
        let mut contracts = BTreeMap::new();
        contracts.insert(1, contract(1, "Vault", vec![func(100, "withdraw")]));
        contracts.insert(2, contract(2, "Base", vec![func(200, "pause")]));
        let mut interfaces = BTreeMap::new();
        interfaces.insert(
            3,
            Interface {
                id: 3,
                name: "IERC20".to_string(),
                relative_file_path: PathBuf::from("src/I.sol"),
                chunk: chunk(),
                functions: vec![func(300, "transfer")],
                description: None,
            },
        );
        let call_graph = CallGraph {
            contracts,
            interfaces,
        };

        let mut state_variables = BTreeMap::new();
        state_variables.insert(10, state_var(10, "balances", "mapping(address => uint256)"));
        state_variables.insert(11, state_var(11, "paused", "bool"));
        let storage_graph = StorageGraph {
            state_variables,
            contract_variables: vec![
                ContractVariable {
                    contract_id: 1,
                    state_variable_id: 10,
                    description: None,
                },
                ContractVariable {
                    contract_id: 2,
                    state_variable_id: 11,
                    description: None,
                },
            ],
            function_state_variables: vec![],
        };
        let inheritance_graph = InheritanceGraph::new(vec![ContractInherit {
            contract_id: 1,
            inherited_id: 2,
        }]);
        Arc::new(ProjectIndex::build(
            call_graph,
            storage_graph,
            inheritance_graph,
        ))
    }

    #[test]
    fn project_index_grounds_contracts_vars_and_functions() {
        let idx = project_index();
        assert!(idx.has_contract("Vault"));
        assert!(idx.has_contract("vault")); // index is lower-cased
        assert!(idx.has_contract("IERC20"));
        assert!(!idx.has_contract("Ghost"));

        assert!(idx.has_state_variable("balances")); // bare
        assert!(idx.has_state_variable("Vault.balances")); // qualified, own
        assert!(idx.has_state_variable("Vault.paused")); // qualified, inherited
        assert!(!idx.has_state_variable("Base.balances")); // not visible to Base
        assert!(!idx.has_state_variable("Vault.ghost"));
        assert!(!idx.has_state_variable("ghost"));

        assert!(idx.contract_has_function("Vault", "withdraw")); // own
        assert!(idx.contract_has_function("Vault", "pause")); // inherited
        assert!(idx.contract_has_function("IERC20", "transfer")); // interface
        assert!(!idx.contract_has_function("Vault", "ghost"));
        assert!(!idx.contract_has_function("Ghost", "x"));
    }

    #[tokio::test]
    async fn state_var_constraint_rejects_unknown_without_mutating() {
        let idx = project_index();
        let draft = DraftHandle::new();
        let tool = AddStateVariableConstraintTool {
            draft: draft.clone(),
            project: idx.clone(),
        };
        let res = tool
            .invoke(AddStateVariableConstraintArgs {
                stage: SpecStage::PostAttack,
                key: "ghost".to_string(),
                constraint: "== 0".to_string(),
            })
            .await
            .unwrap();
        assert!(res.starts_with("error:"), "{res}");
        assert!(res.contains("Draft unchanged"), "{res}");
        assert!(
            draft
                .snapshot()
                .await
                .post_attack
                .states_variables
                .is_empty()
        );

        // a real project var is accepted
        let res = tool
            .invoke(AddStateVariableConstraintArgs {
                stage: SpecStage::PostAttack,
                key: "Vault.balances".to_string(),
                constraint: "drained".to_string(),
            })
            .await
            .unwrap();
        assert!(res.starts_with("ok:"), "{res}");
        assert_eq!(draft.snapshot().await.post_attack.states_variables.len(), 1);
    }

    #[tokio::test]
    async fn aux_contract_workflow_grounds_against_registry() {
        let idx = project_index();
        let draft = DraftHandle::new();

        // cannot declare aux state before the aux contract exists
        let r = AddAuxContractStateTool {
            draft: draft.clone(),
        }
        .invoke(AddAuxContractStateArgs {
            contract: "Attacker".to_string(),
            name: "callCount".to_string(),
            solidity_type: "uint256".to_string(),
            purpose: "reentry counter".to_string(),
        })
        .await
        .unwrap();
        assert!(r.starts_with("error:"), "{r}");

        // cannot shadow a real project contract
        let r = AddAuxContractTool {
            draft: draft.clone(),
            project: idx.clone(),
        }
        .invoke(AddAuxContractArgs {
            name: "Vault".to_string(),
            purpose: "x".to_string(),
            stands_in_for: None,
            fidelity: None,
        })
        .await
        .unwrap();
        assert!(r.starts_with("error:"), "{r}");

        // declare aux contract + state
        AddAuxContractTool {
            draft: draft.clone(),
            project: idx.clone(),
        }
        .invoke(AddAuxContractArgs {
            name: "Attacker".to_string(),
            purpose: "reentrancy".to_string(),
            stands_in_for: None,
            fidelity: None,
        })
        .await
        .unwrap();
        AddAuxContractStateTool {
            draft: draft.clone(),
        }
        .invoke(AddAuxContractStateArgs {
            contract: "Attacker".to_string(),
            name: "callCount".to_string(),
            solidity_type: "uint256".to_string(),
            purpose: "reentry counter".to_string(),
        })
        .await
        .unwrap();

        // aux state var grounds via the registry (case-insensitive contract)
        let r = AddStateVariableConstraintTool {
            draft: draft.clone(),
            project: idx.clone(),
        }
        .invoke(AddStateVariableConstraintArgs {
            stage: SpecStage::PostAttack,
            key: "attacker.callCount".to_string(),
            constraint: "== 2".to_string(),
        })
        .await
        .unwrap();
        assert!(r.starts_with("ok:"), "{r}");

        // contract constraint accepts the aux contract
        let r = AddContractConstraintTool {
            draft: draft.clone(),
            project: idx.clone(),
        }
        .invoke(AddContractConstraintArgs {
            stage: SpecStage::Setup,
            contract: "Attacker".to_string(),
            constraint: "deployed".to_string(),
        })
        .await
        .unwrap();
        assert!(r.starts_with("ok:"), "{r}");
    }

    #[tokio::test]
    async fn set_call_sequence_is_atomic_and_allows_aux_functions() {
        let idx = project_index();
        let draft = DraftHandle::new();
        AddAuxContractTool {
            draft: draft.clone(),
            project: idx.clone(),
        }
        .invoke(AddAuxContractArgs {
            name: "Attacker".to_string(),
            purpose: "reentrancy".to_string(),
            stands_in_for: None,
            fidelity: None,
        })
        .await
        .unwrap();

        let seq = SetCallSequenceTool {
            draft: draft.clone(),
            project: idx.clone(),
        };

        // one phantom function => whole call rejected, sequence untouched
        let r = seq
            .invoke(SetCallSequenceArgs {
                steps: vec![
                    CallStepArg {
                        contract: "Vault".to_string(),
                        function: "withdraw".to_string(),
                        intention: "a".to_string(),
                    },
                    CallStepArg {
                        contract: "Vault".to_string(),
                        function: "ghost".to_string(),
                        intention: "b".to_string(),
                    },
                ],
            })
            .await
            .unwrap();
        assert!(r.starts_with("error:"), "{r}");
        assert!(draft.snapshot().await.sequence.is_empty());

        // project fn grounded + aux fn accepted (synthesized by harness)
        let r = seq
            .invoke(SetCallSequenceArgs {
                steps: vec![
                    CallStepArg {
                        contract: "Attacker".to_string(),
                        function: "attack".to_string(),
                        intention: "kick".to_string(),
                    },
                    CallStepArg {
                        contract: "Vault".to_string(),
                        function: "withdraw".to_string(),
                        intention: "reenter".to_string(),
                    },
                ],
            })
            .await
            .unwrap();
        assert!(r.starts_with("ok:"), "{r}");
        assert_eq!(draft.snapshot().await.sequence.len(), 2);
    }

    #[tokio::test]
    async fn commit_requires_summary_invariant_and_sequence() {
        let idx = project_index();
        let draft = DraftHandle::new();
        let commit = CommitSpecificationTool {
            draft: draft.clone(),
            project: idx.clone(),
            max_specs: 4,
            scope_gate: None,
            link_source: LinkSource::Mapper,
            economic_gate: None,
            chain: None,
        };

        let r = commit
            .invoke(CommitSpecificationArgs {
                summary: "   ".to_string(),
            })
            .await
            .unwrap();
        assert!(r.contains("summary is required"), "{r}");

        let r = commit
            .invoke(CommitSpecificationArgs {
                summary: "drain".to_string(),
            })
            .await
            .unwrap();
        // A specification with neither a state invariant nor a functional claim
        // proves nothing; the refusal names both routes out.
        assert!(r.contains("proves nothing yet"), "{r}");
        assert!(r.contains("add_property"), "{r}");

        AddStateVariableConstraintTool {
            draft: draft.clone(),
            project: idx.clone(),
        }
        .invoke(AddStateVariableConstraintArgs {
            stage: SpecStage::PostAttack,
            key: "balances".to_string(),
            constraint: "drained".to_string(),
        })
        .await
        .unwrap();

        let r = commit
            .invoke(CommitSpecificationArgs {
                summary: "drain".to_string(),
            })
            .await
            .unwrap();
        assert!(r.contains("call sequence is empty"), "{r}");

        SetCallSequenceTool {
            draft: draft.clone(),
            project: idx.clone(),
        }
        .invoke(SetCallSequenceArgs {
            steps: vec![CallStepArg {
                contract: "Vault".to_string(),
                function: "withdraw".to_string(),
                intention: "x".to_string(),
            }],
        })
        .await
        .unwrap();

        let r = commit
            .invoke(CommitSpecificationArgs {
                summary: "attacker drains vault".to_string(),
            })
            .await
            .unwrap();
        assert!(r.contains("no impact class declared"), "{r}");

        SetImpactClassTool {
            draft: draft.clone(),
        }
        .invoke(SetImpactClassArgs {
            class: ImpactClass::DirectTheftOfFunds,
        })
        .await
        .unwrap();

        let r = commit
            .invoke(CommitSpecificationArgs {
                summary: "attacker drains vault".to_string(),
            })
            .await
            .unwrap();
        assert!(r.starts_with("ok:"), "{r}");
        let snap = draft.snapshot().await;
        assert_eq!(snap.completed.len(), 1);
        assert_eq!(snap.completed[0].summary, "attacker drains vault");
        assert_eq!(
            snap.completed[0].impact,
            Some(ImpactClass::DirectTheftOfFunds),
            "the declared class travels into the committed spec"
        );
        // draft reset after commit
        assert!(snap.post_attack.states_variables.is_empty());
        assert!(snap.sequence.is_empty());
        assert_eq!(snap.impact, None, "the class belongs to the spec, not the draft");
    }

    /// A divergence with nothing of value at stake is refused at declaration
    /// time, so the agent learns it before it has built anything.
    #[tokio::test]
    async fn impact_class_must_be_reportable() {
        let draft = DraftHandle::new();

        let r = SetImpactClassTool {
            draft: draft.clone(),
        }
        .invoke(SetImpactClassArgs {
            class: ImpactClass::NoValueAtRisk,
        })
        .await
        .unwrap();
        assert!(r.contains("not reportable"), "{r}");
        assert!(
            r.contains("abandon this direction"),
            "the agent must be told what to do instead: {r}"
        );
        assert_eq!(
            draft.snapshot().await.impact,
            None,
            "a refused class must not reach the draft"
        );

        let r = SetImpactClassTool {
            draft: draft.clone(),
        }
        .invoke(SetImpactClassArgs {
            class: ImpactClass::ProtocolDos,
        })
        .await
        .unwrap();
        assert!(r.starts_with("ok:"), "{r}");
        assert_eq!(draft.snapshot().await.impact, Some(ImpactClass::ProtocolDos));
    }

    /// An engagement scope whose only insured asset is the fixture's `Vault`,
    /// whose only payable impact is theft, and which keeps the standing
    /// unprivileged-actor rule.
    const SCOPE_FIXTURE: &str = r#"{
        "program": "Immunefi: Fixture",
        "primacy": "rules",
        "assets": [{ "name": "Vault", "chain_id": 1, "value_usd": 1000000 }],
        "impacts_in_scope": ["direct_theft_of_funds"]
    }"#;

    /// Fill the draft with one complete, otherwise-valid specification: impact,
    /// actor, a one-step sequence on `contract.function`, and a post_attack
    /// invariant. Everything the commit gate requires, so a refusal in the test
    /// can only come from the engagement scope.
    async fn build_scope_fixture_draft(
        draft: &DraftHandle,
        idx: &Arc<ProjectIndex>,
        contract: &str,
        function: &str,
        attacker: AttackerModel,
        impact: ImpactClass,
    ) {
        SetImpactClassTool {
            draft: draft.clone(),
        }
        .invoke(SetImpactClassArgs { class: impact })
        .await
        .unwrap();
        SetAttackerModelTool {
            draft: draft.clone(),
        }
        .invoke(SetAttackerModelArgs {
            privilege: attacker.privilege,
            role: attacker.role.clone(),
            note: attacker.note.clone(),
        })
        .await
        .unwrap();
        SetCallSequenceTool {
            draft: draft.clone(),
            project: idx.clone(),
        }
        .invoke(SetCallSequenceArgs {
            steps: vec![CallStepArg {
                contract: contract.to_string(),
                function: function.to_string(),
                intention: "drive the state the spec describes".to_string(),
            }],
        })
        .await
        .unwrap();
        AddContractConstraintTool {
            draft: draft.clone(),
            project: idx.clone(),
        }
        .invoke(AddContractConstraintArgs {
            stage: SpecStage::PostAttack,
            contract: contract.to_string(),
            constraint: "value has left the protocol".to_string(),
        })
        .await
        .unwrap();
    }

    fn unprivileged() -> AttackerModel {
        AttackerModel {
            privilege: AttackerPrivilege::Unprivileged,
            role: None,
            note: "any holder can call it".to_string(),
        }
    }

    fn privileged(role: &str) -> AttackerModel {
        AttackerModel {
            privilege: AttackerPrivilege::Privileged,
            role: Some(role.to_string()),
            note: "the sequence runs as the role holder".to_string(),
        }
    }

    /// Commit one draft through the gate and hand back the tool's own reply —
    /// the same string the agent would read.
    async fn commit_through_gate(
        draft: &DraftHandle,
        idx: &Arc<ProjectIndex>,
        gate: Option<Arc<ScopeGate>>,
        summary: &str,
    ) -> String {
        let tool = CommitSpecificationTool {
            draft: draft.clone(),
            project: idx.clone(),
            max_specs: 8,
            scope_gate: gate,
            link_source: LinkSource::Mapper,
            economic_gate: None,
            chain: None,
        };
        tool.invoke(CommitSpecificationArgs {
            summary: summary.to_string(),
        })
        .await
        .unwrap()
    }

    /// The commit gate, end to end: what the program's own rules exclude is
    /// refused before the specification costs a harness run, and what they
    /// admit still commits.
    #[tokio::test]
    async fn commit_is_refused_by_the_engagement_scope() {
        let idx = project_index();
        let draft = DraftHandle::new();
        let scope = ProgramScope::parse(SCOPE_FIXTURE, Path::new("knowdit-scope.json"))
            .expect("scope fixture parses");
        // A repository root with no deployment tree: the deployment rule is
        // skipped, which is what an engagement that names no deployed
        // addresses wants.
        let gate = Arc::new(ScopeGate::new(scope, PathBuf::from("/nonexistent-repo")));

        // 1. An unprivileged exploit on the insured asset commits.
        build_scope_fixture_draft(
            &draft,
            &idx,
            "Vault",
            "withdraw",
            unprivileged(),
            ImpactClass::DirectTheftOfFunds,
        )
        .await;
        let r = commit_through_gate(
            &draft,
            &idx,
            Some(gate.clone()),
            "withdraw lets a caller take more than it credits",
        )
        .await;
        assert!(r.starts_with("ok:"), "{r}");
        assert_eq!(draft.snapshot().await.completed.len(), 1);

        // 2. The same defect behind a role the protocol already grants is
        // refused, and the refusal names the role.
        build_scope_fixture_draft(
            &draft,
            &idx,
            "Vault",
            "withdraw",
            privileged("GUARDIAN_ROLE"),
            ImpactClass::DirectTheftOfFunds,
        )
        .await;
        let r = commit_through_gate(
            &draft,
            &idx,
            Some(gate.clone()),
            "withdraw drains the pool when the guardian calls it",
        )
        .await;
        assert!(r.starts_with("error:"), "{r}");
        assert!(r.contains("GUARDIAN_ROLE"), "{r}");
        assert_eq!(
            draft.snapshot().await.completed.len(),
            1,
            "a refused specification must not be committed"
        );

        // 3. A sequence that touches nothing the program insures is refused,
        // even with a payable impact and an unprivileged actor.
        build_scope_fixture_draft(
            &draft,
            &idx,
            "Base",
            "pause",
            unprivileged(),
            ImpactClass::DirectTheftOfFunds,
        )
        .await;
        let r = commit_through_gate(
            &draft,
            &idx,
            Some(gate.clone()),
            "pause strands the pool without an unpause path",
        )
        .await;
        assert!(r.starts_with("error:"), "{r}");
        assert!(r.contains("Base"), "the refusal names what was called: {r}");
        assert_eq!(draft.snapshot().await.completed.len(), 1);

        // 4. With no scope declared the same draft commits: the gate is an
        // engagement's own rules, not a new default.
        let unscoped = commit_through_gate(
            &draft,
            &idx,
            None,
            "pause strands the pool without an unpause path",
        )
        .await;
        assert!(unscoped.starts_with("ok:"), "{unscoped}");
        assert_eq!(draft.snapshot().await.completed.len(), 2);
    }

    /// The attacker tool's one cross-field rule: an exploit cannot claim
    /// privileged reach without naming the authority it relies on.
    #[tokio::test]
    async fn set_attacker_model_requires_a_role_for_privileged() {
        let draft = DraftHandle::new();
        let tool = SetAttackerModelTool {
            draft: draft.clone(),
        };

        let r = tool
            .invoke(SetAttackerModelArgs {
                privilege: AttackerPrivilege::Privileged,
                role: None,
                note: "runs as the operator".to_string(),
            })
            .await
            .unwrap();
        assert!(r.starts_with("error:"), "{r}");
        assert!(draft.snapshot().await.attacker.is_none());

        let r = tool
            .invoke(SetAttackerModelArgs {
                privilege: AttackerPrivilege::PrivilegeEscalation,
                role: None,
                note: "a caller takes the operator role".to_string(),
            })
            .await
            .unwrap();
        assert!(r.starts_with("ok:"), "{r}");
        assert_eq!(
            draft.snapshot().await.attacker.map(|a| a.privilege),
            Some(AttackerPrivilege::PrivilegeEscalation)
        );

        let r = tool
            .invoke(SetAttackerModelArgs {
                privilege: AttackerPrivilege::Unprivileged,
                role: None,
                note: "   ".to_string(),
            })
            .await
            .unwrap();
        assert!(r.starts_with("error:"), "a blank note is not a declaration: {r}");
    }

    #[tokio::test]
    async fn a_property_claim_grounds_against_the_project_and_can_carry_a_commit() {
        let idx = project_index();
        let draft = DraftHandle::new();
        let tool = AddPropertyTool {
            draft: draft.clone(),
            project: idx.clone(),
        };

        // An unknown contract is refused; an unknown function is refused.
        let r = tool
            .invoke(AddPropertyArgs {
                kind: PropertyKind::BranchExpected,
                target: "NotAContract.withdraw".to_string(),
                claim: "the branch must be the sell side".to_string(),
            })
            .await
            .unwrap();
        assert!(r.contains("not a project contract"), "{r}");
        let r = tool
            .invoke(AddPropertyArgs {
                kind: PropertyKind::BranchExpected,
                target: "Vault.notAFunction".to_string(),
                claim: "the branch must be the sell side".to_string(),
            })
            .await
            .unwrap();
        assert!(r.contains("has no function"), "{r}");
        assert!(draft.snapshot().await.properties.is_empty());

        // A real function on the fixture's contract is accepted.
        let r = tool
            .invoke(AddPropertyArgs {
                kind: PropertyKind::BranchExpected,
                target: "Vault.withdraw".to_string(),
                claim: "with vethBalance above target, the sell branch must be selected"
                    .to_string(),
            })
            .await
            .unwrap();
        assert!(r.starts_with("ok:"), "{r}");
        assert_eq!(draft.snapshot().await.properties.len(), 1);

        // A property-only specification commits: no post_attack invariant is
        // required when the defect is a computation.
        SetImpactClassTool {
            draft: draft.clone(),
        }
        .invoke(SetImpactClassArgs {
            class: ImpactClass::DirectTheftOfFunds,
        })
        .await
        .unwrap();
        SetAttackerModelTool {
            draft: draft.clone(),
        }
        .invoke(SetAttackerModelArgs {
            privilege: AttackerPrivilege::Unprivileged,
            role: None,
            note: "any holder can call it".to_string(),
        })
        .await
        .unwrap();
        SetCallSequenceTool {
            draft: draft.clone(),
            project: idx.clone(),
        }
        .invoke(SetCallSequenceArgs {
            steps: vec![CallStepArg {
                contract: "Vault".to_string(),
                function: "withdraw".to_string(),
                intention: "drive the computation".to_string(),
            }],
        })
        .await
        .unwrap();
        let r = commit_through_gate(&draft, &idx, None, "the direction branch is chosen wrong").await;
        assert!(r.starts_with("ok:"), "{r}");
        let committed = draft.snapshot().await.completed;
        assert_eq!(committed.len(), 1);
        assert_eq!(committed[0].properties.len(), 1);
        assert_eq!(committed[0].properties[0].kind, PropertyKind::BranchExpected);
    }

    #[tokio::test]
    async fn a_specification_that_proves_nothing_still_cannot_commit() {
        let idx = project_index();
        let draft = DraftHandle::new();
        SetImpactClassTool {
            draft: draft.clone(),
        }
        .invoke(SetImpactClassArgs {
            class: ImpactClass::DirectTheftOfFunds,
        })
        .await
        .unwrap();
        SetCallSequenceTool {
            draft: draft.clone(),
            project: idx.clone(),
        }
        .invoke(SetCallSequenceArgs {
            steps: vec![CallStepArg {
                contract: "Vault".to_string(),
                function: "withdraw".to_string(),
                intention: "nothing asserted".to_string(),
            }],
        })
        .await
        .unwrap();
        let r = commit_through_gate(&draft, &idx, None, "no claim at all").await;
        assert!(r.contains("proves nothing yet"), "{r}");
        assert!(r.contains("add_property"), "the refusal must name both routes: {r}");
    }
}
