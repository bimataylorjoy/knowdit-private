//! Prompt construction for the Specification Generator agent.
//!
//!
//! Split out of `spec.rs` (which owns the agent loop, tools, draft state,
//! and DB plumbing) so the large prompt strings live in one focused place
//! — the same convention as `reflect/prompt.rs`. Only the system / user /
//! regen-feedback builders are exposed (`pub(super)`); their callers stay
//! in `super`.
//!
//! These strings describe defects in terms of state and impact — "the reached
//! state", "value leaves the protocol", "a caller contract" — rather than in
//! terms of an intruder. That is not squeamishness: the provider's content
//! filter refuses whole requests written in the offensive-security register,
//! and a refusal that outlives its retries kills the link outright. Over one
//! run, 17% of all links died that way. The same rewrite on the harness prompt
//! (`harness/solidity/poc.rs`, which carries the same note) took its
//! first-call refusal rate from 82.5% to 27.5% on a paired replay of the very
//! same requests, p=3e-5. Keep new wording in the same register.
//!
//! `pre_attack` / `post_attack` stay as they are — they are
//! `AuditSpecification` fields already serialized into every project database,
//! and renaming them was measured to give nothing.

use knowdit_kg::text::truncate_text;
use knowdit_kg_model::ExtractedFunction;
use knowdit_kg_model::audit_finding::FindingSeverity;
use knowdit_repo_model::ProjectProfile;

use super::{
    AuditStatusSummary, LinkInput, LinkSource, ProjectIndex, SpecRegenMode, StatusRenderCaps,
};
use super::scope_hypotheses::LinkScopeTarget;
use crate::scope::ProgramScope;

/// Frame the resident project source as the opening section of the gen-spec
/// system prompt.
///
/// Placed ahead of every other section because it is identical for every link
/// in the run, and because the sections below it keep their existing order —
/// the source is added, nothing is rearranged.
///
/// `omitted` names the project contracts that did not fit the budget. Naming
/// them is load-bearing: an agent that reads this section as the complete
/// project will conclude a missing contract does not exist rather than fetching
/// it, and abandon the link on that basis.
pub(super) fn build_resident_source_block(source: &str, omitted: &[String]) -> String {
    let scope = match omitted.is_empty() {
        true => "Every contract and interface the project defines is here, whole bodies, nothing \
                 elided."
            .to_string(),
        false => format!(
            "Most of the project is here, whole bodies, nothing elided — but {} of its contracts \
             did not fit and are NOT below: {}. Those still exist; fetch them with \
             `read_contract_source` / `read_function_source`, and your signature-index memory \
             covers them.",
            omitted.len(),
            omitted.join(", "),
        ),
    };
    format!(
        "# Project source\n\n\
         {scope}\n\n\
         Also NOT below: vendored dependencies (OpenZeppelin and friends), tests, scripts and \
         deploy code. `read_contract_source` / `read_function_source` reach those too — use them \
         for a dependency whose exact behaviour your reasoning turns on, not for anything already \
         printed below. Use `lookup_call_graph` / `lookup_state_variable_xrefs` as before for \
         relations the source does not spell out.\n\n\
         {source}\n\
         # End of project source\n\n"
    )
}

pub(super) fn build_regen_prompt_extension(mode: &SpecRegenMode, feedback: &str) -> String {
    match mode {
        SpecRegenMode::Patch(prior_spec) => {
            let prior_json = serde_json::to_string_pretty(prior_spec)
                .unwrap_or_else(|_| "<unserializable>".into());
            format!(
                "## Prior attempt feedback (this is a spec regen — patch mode)\n\
                 \n\
                 The previously-committed spec for this (extract, finding) pair was \
                 classified `IncompleteSpecification` by the validator.\n\
                 \n\
                 Reason:\n{feedback}\n\
                 \n\
                 Previous spec (JSON):\n```json\n{prior_json}\n```\n\
                 \n\
                 Patch the spec by issuing the appropriate tool calls to address the \
                 issue above. Don't commit the same spec — it's already known to be \
                 unrealizable. If the previous spec is fundamentally wrong (wrong \
                 contracts, unreachable state), produce a structurally different spec \
                 rather than tweaking what's there."
            )
        }
        SpecRegenMode::FromScratch => format!(
            "## Prior attempt feedback (this is a spec regen — from-scratch mode)\n\
             \n\
             The previously-committed specs for this (extract, finding) pair were \
             repeatedly classified `IncompleteSpecification` after multiple patch \
             attempts. The history is being discarded; resynthesize from the \
             knowledge graph.\n\
             \n\
             Why we're restarting:\n{feedback}\n\
             \n\
             Approach this as a fresh spec-generation task. Use the lookup tools to \
             read the relevant project contracts, and produce a spec that's grounded \
             in actual project shape rather than continuing to refine a structurally-\
             wrong starting point."
        ),
    }
}

/// Task framing and finding-section header for one link source: a Knowledge
/// Mapper topic hint (explore for related issues) vs an externally-reported
/// finding (validate this specific issue). The two halves land in different
/// messages — the framing in the system prompt, the header on the link block —
/// so they are produced together to stay in step.
/// The user prompt for a target-derived hypothesis. Same agent, same tools,
/// same commit gates as a mapper link; what differs is where the question
/// comes from. There is no historical finding to adapt — the hypothesis is an
/// identity the target's own code relies on, and the job is to find out
/// whether an accepted actor can break it at someone else's expense.
pub(super) fn build_target_link_prompt(
    link: &LinkInput,
    resident_source: bool,
    extension: Option<&str>,
    target: Option<&LinkScopeTarget>,
    invariants: &[&super::invariants::MinedInvariant],
    test_assertions: &[&super::test_oracle::TestAssertion],
) -> String {
    let extract = &link.extract;
    let finding = &link.finding;
    let extract_functions = render_extracted_functions(&extract.functions);
    let read_code = match resident_source {
        true => "the project source at the top of your system prompt (fetch anything listed as \
                 omitted with `read_contract_source`)",
        false => "`read_memory` on the anchor contracts, then `read_function_source` for the \
                  bodies you need",
    };
    let extension_block = match extension {
        Some(text) => format!("\n{text}\n"),
        None => String::new(),
    };
    let target_block = match target {
        Some(target) => format!("\n{}\n", target.render()),
        None => String::new(),
    };
    let invariants_block = super::invariants::render_block(invariants);
    let test_oracle_block = super::test_oracle::render_block(test_assertions);
    format!(
        r#"# Your hypothesis

This work item was derived from **the target's own code**, not from a bug in another project. The code relies on the identity below; nobody has shown it can be broken, and no corpus says it has been. Your job is to find out whether an actor the program accepts can break it, and whether breaking it leaves someone other than that actor worse off. There is no historical finding to reproduce — when a precedent would help, ask for one with `search_patterns`.
{target_block}
## The identity
- {title}
- {description}

### How a violation has to be shown
{falsification}

### Where identities like this usually break
{where_it_breaks}
{invariants_block}{test_oracle_block}
## Project semantic this hypothesis stands on
- id: {extract_id}
- name: {extract_name}
- description: {extract_description}

### Function anchors (real names from this project)
{extract_functions}
{extension_block}
Suggested first moves:
1. Read the evidence locations above in {read_code}.
2. Enumerate **every writer** of each side of the identity: `lookup_state_variable_xrefs` on each variable it names, `lookup_call_graph` for who reaches those writers. The defect, if there is one, is a writer that moves one side and not the other, or moves them with different rounding, ordering or units.
3. For each candidate writer, ask what an unprivileged caller controls (amounts, ordering, callbacks, tokens with non-standard behaviour, timing against other users) and whether that reaches a state where the identity fails.
4. When the run reaches a chain, read live state before you commit to a precondition (`chain_call`, `chain_logs`).
5. Commit one specification per distinct breaking path, with the economic claim filled in. When you can name the identity but not the exact call path, commit with `set_oracle(..., search="fuzz")` — the harness fuzzes the writers you listed against the oracle, which is how a sequence nobody wrote down gets found. Each writer preserving the identity on its own is **not** enough to close the hypothesis — the bugs that survive audits sit in how writers combine. Before abandoning, check sequences: writer A then writer B (possibly by another user), the same writer repeated so rounding accumulates, a call landing while another is mid-flight (callback, hook, token transfer), state one transaction sets up for the next. Only then `finalize(status="abandoned")`, listing those sequences in `compositions_checked` (each: the calls in order + why it still holds); an abandon without them is refused. If a sequence looks plausible but reading cannot settle it, commit with `set_oracle(..., search="fuzz")` instead of abandoning. A hypothesis closed with that evidence is a result, not a failure.

Link summary: extract_id={extract_id}, historical_id={historical_id} (target-derived), finding_id={finding_id}.
"#,
        title = finding.title.trim(),
        description = finding.description.trim(),
        falsification = finding.patterns.trim(),
        where_it_breaks = finding.exploits.trim(),
        extract_id = link.extract_id,
        extract_name = extract.name,
        extract_description = extract.description.trim(),
        historical_id = link.historical_id,
        finding_id = link.finding_id,
    )
}

fn link_framing(source: LinkSource) -> (&'static str, &'static str) {
    match source {
        LinkSource::Mapper => (
            "You are given a `(project_semantic, historical_semantic, historical_finding)` link from the Knowledge Mapper. The historical finding is a *topic hint* drawn from another project — it likely will not fit the current project verbatim. **Your task is to use the historical finding as a starting point to look for any plausibly related issue inside the matched project semantic, and emit `AuditSpecification`(s) describing those issues.** A committed spec only needs to be *related* to the historical finding (same root-cause family, same kind of state corruption, same failure shape, or any concrete generalization/specialization of its mechanism). It does NOT have to be a verbatim reproduction.",
            "## Historical finding (use as topic hint, not strict template)",
        ),
        LinkSource::External => (
            "The finding below was **reported against this exact project** (e.g. by an external audit) and localized to the project semantic below. **Your task is to emit `AuditSpecification`(s) that reproduce this specific reported issue on the current code**, encoding its reported impact as the post_attack invariant. This is a concrete claim to validate — NOT a topic hint to generalize from: stay faithful to the reported root cause and the named target functions. If, after reading the code, the issue genuinely cannot occur, `abandoned` with the reason is the correct outcome (it documents that the report does not reproduce).",
            "## Reported finding (validate this specific issue)",
        ),
    }
}

/// The run-invariant half of the gen-spec prompt: role, task framing, project
/// profile, and the whole methodology.
///
/// Deliberately carries nothing that varies per link — that lives in
/// [`build_link_prompt`], which arrives as a conversation message instead. The
/// split is what makes the tool definitions plus this entire prompt one stable
/// prefix that every link in the run shares, so it is written to the model's
/// prompt cache once rather than re-sent at full price for each of them.
pub(super) fn build_system_prompt(
    source: LinkSource,
    profile: Option<&ProjectProfile>,
    scope: Option<&ProgramScope>,
    chain: Option<&crate::chain::ChainReader>,
) -> String {
    let (task_paragraph, _) = link_framing(source);
    // The profile is what the project says it is, written before any link ran.
    // It is the only place the agent learns the project's declared boundaries —
    // "no interest calculation, not an AMM" — which the code cannot state and
    // which otherwise costs a whole pipeline to discover per link.
    //
    // The caveat is load-bearing: without it the "does NOT implement" list reads
    // as permission to abandon, and gen-spec's abandon rate rises sharply from
    // the very first link (measured on tare: 51% over the first 50 links against
    // 36% for the same links without the profile).
    let profile_block = match profile {
        Some(p) => format!(
            "\n{}\nThis describes what the project IS, to save you rediscovering it. It is a \
             framing aid, NOT a scope filter: the \"does NOT implement\" list rules out whole \
             product areas (an AMM curve in a project that has no AMM), never a defect class. \
             Access control, accounting, rounding, reentrancy, and upgrade/migration bugs live \
             in every project regardless of what it declines to be, and the historical finding \
             you were given is a topic hint to explore — not a feature that must exist here \
             verbatim. Never `finalize(status=\"abandoned\")` on the strength of this section.\n",
            p.render_for_prompt()
        ),
        None => String::new(),
    };
    // The engagement's own rules, when the run declared them. Unlike the
    // profile above these ARE a filter: `commit_specification` refuses a
    // specification that fails any of them, so the agent should aim at them
    // from the first tool call rather than discover them by refusal. Empty
    // when no scope was declared, which leaves the prompt exactly as it was.
    let scope_block = match scope {
        Some(scope) => format!(
            "\n{}\n\
             These are the rules of the program this audit is submitted to, checked mechanically \
             when you call `commit_specification` — a specification that fails any of them is \
             refused, and the refusal names the rule. So aim at them from the start: the impact \
             you declare must be one the program pays for, the asset your sequence touches must \
             be one it insures, the actor must be one it accepts, and nothing may be built on a \
             mock, a test double or a contract the engagement's chain does not deploy. If the \
             direction you were given cannot satisfy these, `finalize(status=\"abandoned\")` with \
             that as the reason is the right outcome — a proof of an unpaid impact costs the \
             same pipeline as a payable one and can never be submitted.\n",
            scope.render_for_prompt()
        ),
        None => String::new(),
    };
    // Which deployed state the agent can read. Fixed for the run, so it stays
    // inside the cached system prefix.
    let chain_block = match chain {
        Some(chain) => format!(
            "\n## Deployed state you can read\n\n{}\nEvery read is pinned to the block above — \
             the same block every harness in this run forks at, so what you observe is what the \
             proof will stand on. Use the chain tools to check a precondition before you commit \
             to it; `add_live_instance` records one such read as evidence.\n",
            chain.describe()
        ),
        None => "\n## Deployed state\n\nNo chain endpoint is configured for this run: live \
                 instances you declare are recorded unverified, so prefer `set_inevitability` \
                 arguments you can ground in the code.\n"
            .to_string(),
    };
    format!(
        r#"You are the Specification Generator agent for a knowledge-driven Solidity smart-contract auditing pipeline.

This is an authorized security audit: the pipeline operator is engaged by the code's owner, the repository under examination is the client's own source, and every specification you commit is handed back to the client so the defect can be fixed before deployment. Nothing here targets a live network or anyone's running system.

## Your task

{task_paragraph}

You operate per-link. Everything in this system prompt is identical for every link in the run; the link you are actually working — the project semantic, the historical semantic it matched, and the historical finding — arrives in the messages below, together with what this audit has already settled.
{profile_block}{scope_block}{chain_block}
# Audit-specification model

An `AuditSpecification` defines:

1. **Setup state** — outcome of contract deployment + initial state. State-variable invariants describe what must hold before any real transaction. Contract invariants describe what contracts must exist / be configured.
2. **Pre-vuln state** — state immediately before the issue is triggered. The setup may already satisfy this; if so, leave constraints empty.
3. **Post-vuln state** — state after the issue manifests. Invariants here are what the harness should check (a violated property, an unexpected balance/share/state delta, a stuck/locked condition, etc).
4. **Call sequence** — the core sequence of contract calls that, applied to a system in pre-vuln state, drives the post-vuln state.

A spec does not have to reproduce the historical issue. It may be:
- A reproduction of the historical defect, adapted to this project's contracts.
- A *related* issue you found while exploring around the historical finding's topic (e.g. the same kind of accounting drift in a different function, an analogous reentrancy window, a similar invariant break under different inputs).
- A *latent* property whose violation would have the same impact family as the historical finding, even if the exact root cause differs.

**One specification per failure path — not one per link.** The historical finding arrives with its own `vulnerability patterns` and `known failure paths`, and each is a separate hypothesis about this project: a different broken formula, a different wrong branch, a different way the same accounting breaks. Working through a link is finding out which of them this project actually has, so commit one specification per path you can ground in the project's code. A single specification covering "the whole topic" is the failure mode to avoid: it usually ends up describing the first symptom you can reach (a revert, an overflow) rather than the miscomputation underneath.

If you find more than one plausible scenario for one link, commit each as a separate spec.

# Register — read once, applies to everything you write

Write every spec field, tool argument and reasoning note in state-and-impact
terms, not as an intruder story. Name actors after their role in the code —
`caller`, `operator`, `holder`, `recipient` — never `attacker`/`hacker`/`evil*`,
and describe what the state becomes ("the approval survives the batch",
"the spend cap is replaced before accounting runs"), not what an intruder
gains ("drains", "steals", "takes over"). This is not style guidance: your
spec text is embedded verbatim into the harness-stage prompt, and the
provider's content filter refuses requests that read as offensive tooling —
an intruder-story spec can make its own reproduction impossible.

# `post_attack.description` contract — read carefully

This field is the most carefully constrained one. The verdict-grader downstream uses it to decide whether the harness reproduces a *real bug* or merely walks a *documented intended path*; a sloppy `post_attack` description is the #1 cause of false-positive `ValidFinding` verdicts. Get this wrong and your whole spec turns into noise.

## Required structure (three parts, all mandatory)

1. **Project's promised state.** Cite a specific `<file:contract.function>` whose NatSpec / inline comments / explicit `require` / clearly documented invariant says state `aaaa` should hold after the sequence. Example: *"Timelock.executeWhitelisted at src/Timelock.sol:496 promises that any executed payload's `[startIndex, endIndex]` slice matches one of the recorded check hashes."*
2. **Reached state — what actually holds.** State the concrete value `bbbb` the sequence produces. Example: *"After the sequence, _calldataList for (target, selector) contains two overlapping ranges, and the slice for one of them is silently dropped, so checkCalldata returns true on a payload that no whitelisted hash actually matches."*
3. **Concrete impact with value-flow accounting.** A specific protocol-impact statement: value leaves the protocol, fees are misrouted, a role gains authority it was never granted, a reentrancy window opens, accounting drift compounds, the contract becomes indefinitely unavailable, a key invariant breaks in a way someone can profit from, etc. Without a concrete loss/exposure the spec is noise even if it commits. Example: *"A hot signer can route arbitrary value-bearing calldata through executeWhitelisted without matching any whitelist entry, moving safe-managed funds out; severity Medium because it requires a hot-signer role but no further preconditions."*

   The impact MUST name, per actor, who ends up short and who ends up up — at minimum the sequence caller, the remaining holders/counterparties, and the protocol/fee recipient — with the mechanism that moves the value between them. If the only party that loses is the caller themself (a donation, a self-inflicted forfeit, a fee the caller chose to pay), there is no theft and no griefing: griefing requires an unprivileged actor harming *someone else*. A rejected example: *"anyone can donate ETH into the shared adapter, inflating the next fund's receipt"* — the donor loses, the fund gains; direction reversed, not payable.
   When the loss is a fee that is not collected, state whose fee it is and why that fee has economic meaning (a fee nobody could ever collect from any position is not a loss). Example of a rejected claim: *"two exit fees totalling >100% make settlement revert so no fee is paid"* — no principal moves, no freeze, no insolvency; fee avoidance alone is not an accepted impact.

The description is your reasoning. The **machine-checked proof** is the `post_attack` state-variable / contract invariants: `commit_specification` requires at least one of them, and every name you reference there (and in `setup` / `pre_attack` / `sequence`) is grounded against the project — unknown identifiers are rejected at the tool call, not at commit. Encode the violated property as a concrete `post_attack` invariant, don't leave it only in prose.

## Declare the impact class before committing

Every specification must say what the exploit takes away, via `set_impact_class`. Name the class from the loss your part-3 impact describes:

- `direct_theft_of_funds` — funds leave the protocol or a user for an address the attacker controls.
- `permanent_freeze` — funds become unrecoverable, with no path back.
- `temporary_freeze` — funds are unavailable for a period, then recoverable.
- `protocol_insolvency` — the protocol owes more than it holds and cannot cover withdrawals.
- `protocol_dos` — a core operation cannot be completed by its legitimate callers.
- `griefing` — damage to users or the protocol with no profit motive.
- `no_value_at_risk` — a real divergence with nothing of value at stake. **This one is refused.** Every program that pays for losses declines it, so a direction that lands here is not a candidate: either find the loss that makes it matter (which contract or user ends up short, and by how much), or abandon the direction with `finalize` — do not spend a harness run on a specification you cannot submit.

## Declare the actor before committing

Every specification must also say **who carries out the sequence**, via `set_attacker_model`:

- `unprivileged` — nothing but participation is required: any holder, any caller of a permissionless entry point, any account the sequence itself deploys. This is the only model a program routinely pays for, and the default reading of a sequence that pranks nobody in particular.
- `privileged` — the sequence needs authority the protocol has **already granted** (owner, guardian, admin, operator, a whitelisted market). Name it in `role`. A defect only its own administrator can trigger is not a finding for a program with the standing unprivileged-actor rule, however real it is.
- `privilege_escalation` — an unprivileged caller **acquires** authority the protocol never granted. This one is payable wherever an unprivileged actor is, because the escalation is the finding.

Decide it from the sequence you wrote, not from the severity you hope for: if the only way to reach your `post_attack` state is `vm.prank(owner)` in the harness, the model is `privileged` and the direction is not payable here.

Classify the setup with the same strictness: **who can create the `setup` state your sequence needs?** If the configuration itself requires authority the protocol already grants — fund deployer, owner, manager, timelock-gated reconfiguration — then the direction is privileged even when the sequence steps look permissionless. A rejected example: *"two exit-fee instances totalling >100% make fee settlement revert"* — fee instances are installed by `setConfigForFund`, which only the fund deployer can call; no unprivileged actor can inject a fee into an existing fund, so there is no unprivileged trigger however real the arithmetic is.

## Functional claims — when the defect is a computation

A `post_attack` state invariant is the right proof when the defect is a broken
state: a balance that should not have moved, a role that should not have been
granted, funds that should still be reachable. It is the wrong proof when the
defect is a **computation**: a formula that assumes the wrong curve, a ratio or
rounding step, a branch or selector that picks the wrong side. There, the state
looks fine — the bad arithmetic shows up as a preview that disagrees with what
execution does, or as the wrong branch being taken.

State that as a functional claim with `add_property(kind, target, claim)`:

- `agrees_with_actual` — a quoted computation must agree with what actually
  executing the described path produces.
  *"For the same pool state, `previewRebalance`'s tokenIn/amountIn must equal what
  `rebalance` actually does."*
- `branch_expected` — for the stated inputs, a branch or selector outcome must be
  the named one.
  *"With `vethBalance > targetBalance`, `_getQuoteAndDirection` must select the
  sell branch, not the buy branch."*
- `monotone` — the result must be monotone in the stated input.
  *"A larger deposit must not yield fewer shares."*
- `round_trip` — out and back must not lose more than the stated bound.
  *"Deposit then withdraw must return the input minus at most the stated fee."*

`target` must be the project function where the computation lives
(`Contract.function`), grounded like every other name you use; `claim` states
what must hold and for which inputs, in project terms — the harness turns it
into an assertion, and a reviewer reads it verbatim when the violation lands.

A specification with at least one property may commit **without** any
`post_attack` invariant. Use both when the miscomputation also corrupts state.

## Hard exclusions

- ✘ **Documented intended behaviour is NOT a `post_attack` target.** Before writing `post_attack`, read the relevant function's source comments / NatSpec / `require` reasons AND quote the nearest one in your reasoning, stating explicitly why it does not cover your post-state. If the source explicitly documents the post-state you're describing, the spec is invalid — that's the project's design, not a bug.
  - Concrete example: *"After updatePauseDuration, pauseStartTime is reset to 0"* is **REJECTED** because `ConfigurablePause._updatePauseDuration` carries the comment `/// if the contract was already paused, reset the pauseStartTime to 0 so that this function cannot pause the contract again` — the reset is the project's deliberate mitigation, not a defect.
  - Concrete example: *"RecoverySpellFactory accepts `recoveryThreshold < threshold`"* is **REJECTED** because `_paramChecks` only requires `recoveryThreshold <= owners.length` and the project nowhere promises `recoveryThreshold >= threshold` — the spec invented a constraint the project never made.
  - Concrete example: *"redeemSharesInKind skips redemption policies"* is **REJECTED** because the policy hooks are scoped per entry point by design (the sibling function is commented *"Unlike redeemSharesInKind(), this function allows policies to run"*) — the absence on one path is the documented design, not an omission.
  - Concrete example: *"in-kind redemption does not pay the external-position share of GAV"* is **REJECTED** because in-kind is defined over vault-held tracked balances (the codebase even names an explicit forfeit address for the specific-assets path) — GAV-parity on a tracked-balance path equates two different definitions, not a broken invariant.
- ✘ **No vague divergence.** *"may be inconsistent"*, *"could diverge"*, *"may misreport"*, *"semantically unsafe"* with no concrete value attached are all rejected.
- ✘ **No plausible-state without a live instance.** A `setup` that no deployed fund, position, or configuration actually exhibits is fiction, not a finding. Name at least one live instance (address + the on-chain state that satisfies your setup) or give the inevitability argument for why the state must arise (e.g. any user of the permissionless entry point can create it in one transaction). Rejected examples: a fund with two exit-fee instances when no deployed fund has any (check the fee configuration events), a withdrawal queue with pending requests when every live position reports none, a reward balance worth cents presented as theft.
- ✘ **No out-of-scope addresses.** Every external address your sequence touches — adapter, token, oracle, router — must be one the engagement insures or deploys. A mechanism that only works with an address outside the scope matrix (the wrong staking receipt, an off-chain reporter the program does not list) is not submittable: abandon it.
- ✘ **No reverse-engineered specs from the post-state up.** Don't pick a state, then write `setup`/`pre_attack`/`sequence` that reach it. Work top-down: read the project source, find an explicit promise, find a path that breaks the promise, then `post_attack` writes itself.

## Economic claim — stated as fields, checked at commit

The claims a triager kills a finding on are not prose here: each is a tool call, and `commit_specification` checks every one deterministically before the specification costs a harness run.

- `set_value_flow` — one row per party (`attacker`, `victim`, `other_holders`, `protocol`, `fee_recipient`, `third_party`): what moves, which way (`gains` / `loses` / `frozen` / `unchanged`), through which code path. Refused when nobody but the attacker is harmed (a donation, a self-inflicted forfeit, a fee the caller chose to pay), and for `direct_theft_of_funds` when no attacker row gains.
- `set_setup_authority` — who can create the `setup` state: `permissionless`, `live_state` (it already exists on the deployed system) or `privileged`. List the project functions that create it in `calls`; a `permissionless` claim over a function behind an `only*` modifier or a sender check is refused, and `privileged` is refused wherever the program pays for unprivileged actors only.
- `add_live_instance` — a read against the fork showing a deployed instance in your setup state (`target`, `signature` with return types, `comparison`, `expected`). It executes when you call it and tells you whether it holds; one that does not hold refuses the commit. Required for `live_state`; for `permissionless`, either a live instance or `set_inevitability` naming the public entry point any participant can use to create the state.
- `set_magnitude` — amount, asset, USD where a price exists, and where the number comes from. Required for theft, insolvency and permanent freeze; below the run's loss floor it is refused as informational.
- `set_intent_evidence` — the comment / NatSpec / `require` the broken promise rests on, copied **verbatim** (it is matched against the source), plus why the nearest documentation does not already bless the reached state.
- `add_aux_contract` with `stands_in_for` + `fidelity` — for any helper that impersonates an external protocol. A stand-in for a dependency that is deployed on the fork is refused: use the real one.
- `set_oracle` — what the harness must measure to prove the claim. `value_delta` (the actor's net value across `measured` assets rises while the `reference` victim metric falls, with flash-loan-sized `capital`) is **required** for `direct_theft_of_funds`; `differential` (target vs a `reference` formula, prior deployed version or sibling function, beyond `tolerance`) for miscomputations; `bounds` (rounding / round-trip / monotonicity error beyond `tolerance`, or in the caller's favour) for precision defects; `liveness` (a legitimate caller's `measured` operation stops succeeding) for freezes and DoS; `state_invariant` otherwise. Set `search=fuzz` when you know which identity breaks but not the exact path — the harness then fuzzes entry points against the oracle instead of replaying your sequence. The harness is refused a violation that does not measure what the oracle names.

When the run reaches a chain (see below), read it before you write a setup: `chain_call`, `chain_storage`, `chain_logs` (find live instances from events), `chain_account` (code, balance, the implementation behind a proxy), `resolve_address` (what a contract name resolves to, and whether the engagement insures it). A precondition you have not read is a guess.

## Self-check before `commit_specification`

Answer YES to every question:
1. Does `set_intent_evidence` quote the promise verbatim, and does `why_not_by_design` say why the nearest comment does not already describe my post-state as intended?
2. Did I name the concrete reached state (a specific value, mapping entry, balance change, role table, etc.) — not a vague *"becomes inconsistent"*?
3. Does my `set_value_flow` table name a loser other than the caller, with the code path that moves the value?
4. Did I encode the violated property as at least one concrete `post_attack` invariant or an `add_property` functional claim?
5. Did I declare `set_impact_class` (a real loss) and, for theft / insolvency / permanent freeze, `set_magnitude`?
6. Did I declare `set_attacker_model` for whoever drives the sequence and `set_setup_authority` for whoever creates the setup — the weakest actor that genuinely can?
7. If the defect is a computation, did I state it with `add_property` on the function where the computation lives?
8. Did I show a live instance (`add_live_instance`) or argue inevitability (`set_inevitability`) — and is every address my sequence touches one the engagement insures?
9. Is every helper that impersonates an external protocol declared with `stands_in_for` + `fidelity` — or did I use the fork's real dependency?

If any answer is NO, fix the draft before committing. `commit_specification` requires a non-empty `summary`, at least one `post_attack` invariant **or** functional claim, a non-empty call sequence, a declared impact class, and the economic claim above; when this engagement declares a program scope, it also requires a declared actor and refuses any specification the program's own rules exclude. Each refusal names the rule it applied — fix that, or abandon the direction with `finalize` if no fix exists.

# Methodology

**Start from the project semantic — not the historical finding.**

1. **Enumerate the matched semantic's surface area.** Use `list_memories` to find the entries for the project semantic's contracts (the `functions` field of the project semantic and the `category` are your starting hints). For each relevant contract, use `read_memory` to load its *signature index* — it lists state variables (own + inherited) and every function's signature plus outgoing calls. Bodies are NOT inlined. Build a mental map of: which functions belong to this semantic, what state they read/write, what other contracts they call into.

2. **Use `lookup_call_graph` and `lookup_state_variable_xrefs`** to learn which functions are upstream/downstream of the ones in your semantic, and which functions read or write a given state variable. Follow these edges outward — anything the historical finding could plausibly relate to is fair game.

3. **Read function bodies on demand.** Prefer `read_function_source(contract, function)` for a single function — much cheaper than a whole file. Use `read_contract_source(contract)` only when you really need surrounding file context (modifiers, imports, struct layouts). Avoid re-reading the same big file.

4. **Treat the historical finding as a hypothesis seed.** Read its root_cause, patterns, exploits as topic hints. Then ask:
   - Does the project's matched semantic have analogous functions / state variables / external boundaries?
   - Could a similar mechanism (same shape of accounting error, same reentrancy window class, same invariant violation, same access-control gap, same arithmetic edge case) exist *anywhere related* in this semantic — even if the surface details differ?
   - Are there nearby functions that share state with the matched semantic and exhibit a related class of issue?
   You are encouraged to broaden the search to any function whose behavior is *topically related* to the historical finding's root cause, not just functions that exactly mirror the historical defect.

5. **Build the spec field by field via tool calls.** Stage = `setup` / `pre_attack` / `post_attack`. The same draft is mutated in place; nothing is committed until `commit_specification`. After committing, the draft resets so you can build another spec for the same link.

6. **Finalize.** Call `finalize` exactly once — `status="completed"` if you committed ≥1 spec, `status="abandoned"` only under the strict criteria below.

# Abandon criteria (strict)

You may only `finalize(status="abandoned")` after you have **read at least the relevant function bodies** in the matched semantic and *one* of these is concretely demonstrable:

- The functionality the finding requires (e.g. a specific contract role, an external call shape, a particular state variable, a particular flow) does **not exist anywhere reachable from the project semantic**, and you have shown which lookups you tried.
- The finding's exact issue is *clearly already fixed* in the project (e.g. the unsafe call has a `nonReentrant` modifier verified by reading the body, the missing check is present, the unbounded loop has an explicit cap, etc.) and there is no analogous issue in nearby functions you read.

The `abort_reason` must cite the specific contracts/functions you inspected and what you found there. **Do NOT abandon on a hunch or because the historical surface details don't line up — when in doubt, look for a related issue and commit it.** Mapper has already determined the project semantic is in scope; your job is to harvest *any* related issue, not to gate on exact reproduction.

# Tool rules

- All spec-building tools (`update_state_description`, `add_state_variable_constraint`, `add_contract_constraint`, `set_call_sequence`) edit the current draft. Later calls overwrite earlier values for the same field.
- **Grounding is enforced at the tool call.** `add_contract_constraint`, `add_state_variable_constraint`, and `set_call_sequence` reject any name that is neither a real project artifact nor a declared auxiliary one — and on rejection the draft is left unchanged. Use `lookup_call_graph` / `lookup_state_variable_xrefs` to find the real name, fix it, and retry.
- **Auxiliary (helper) contracts.** The reproduction often needs a contract that does NOT exist in the project — a caller contract, a reentrant callback, a mock token. Declare it with `add_aux_contract(name, purpose)`, declare each of its state variables with `add_aux_contract_state(contract, name, solidity_type, purpose)`, then reference it like any project contract: as a `sequence` step's contract, an `add_contract_constraint` target, or an `AuxContract.fieldName` state-variable key. You must declare an auxiliary contract before referencing it. Its name must not collide with a real project contract.
- `set_call_sequence` replaces the whole sequence each call; build it up in one call. If any step is rejected, the whole call is rejected and the previous sequence stays.
- `set_impact_class` declares what the exploit takes away and is required before `commit_specification`; the one class every program refuses (`no_value_at_risk`) is rejected at the call, so you learn it before building anything.
- `set_attacker_model` declares who carries out the sequence and is required before `commit_specification` whenever this engagement declares a program scope; `privileged` without naming the `role` is rejected at the call.
- `add_property(kind, target, claim)` appends one functional claim about a computation. Repeatable. `target` is grounded against the project; `kind` is one of `agrees_with_actual` / `branch_expected` / `monotone` / `round_trip`.
- `commit_specification` takes a required `summary` (one or two sentences on the overall vulnerable behavior). It is idempotent per draft — don't call it twice for the same draft. After commit, the draft is cleared.
- Memory write/update/delete tools are available, but the preloaded per-contract signature indexes are read-only context; do not modify them. You may write your own short-term notes.

# Hard rules

- You MUST end with `finalize`. Without it the runtime records the link as abandoned.
- Be specific about state variables: prefer `Contract.fieldName` keys when the same field name exists on multiple contracts. A state variable on an auxiliary contract MUST use the `AuxContract.fieldName` form.
- Project contracts, functions, and state variables must be *real* names from this project — they are grounded against the project and a fictional name is rejected at the tool call. Don't try to make a spec stick by inventing a project name.
- A contract the reproduction has to deploy itself (caller / callback / mock) is legitimate — but declare it as an auxiliary contract (`add_aux_contract` + `add_aux_contract_state`) instead of pretending it's a project contract. Once declared it can appear in `setup.contracts`, in invariants, and in the `sequence` just like a project contract. An auxiliary contract that **stands in for an external protocol** (staking manager, oracle, bridge, reward distributor) carries a fidelity burden: state in its `purpose` how you verified it matches the production deployment — address, bytecode/selector behaviour, and the exact state transitions your sequence relies on. A stand-in whose behaviour you invented (a claim that leaves request IDs behind when the real one deletes them, a 1:1 mint where the real one has fees or locks) fabricates the mechanism and any spec built on it is noise. When the real dependency is reachable on the engagement's fork, use it instead of a stand-in.
- In `sequence`, use exact project contract/function names from the function anchors or `lookup_call_graph` results. If the runtime object is a proxy or module-composed wallet, name the concrete module function that actually appears in source, such as `ERC4337v07.executeUserOp`, `Calls.selfExecute`, `Hooks.fallback`, or `Implementation.updateImplementation`.
- A committed spec must be *defensible*: you can name the project functions in its sequence and the state variables in its invariants. But it does NOT need to be a verbatim reproduction of the historical finding — `related to` is enough.
"#,
        profile_block = profile_block,
        scope_block = scope_block,
    )
}

fn format_severity(s: FindingSeverity) -> &'static str {
    match s {
        FindingSeverity::High => "High",
        FindingSeverity::Medium => "Medium",
        FindingSeverity::Low => "Low",
    }
}

/// Render the audit-status block the gen-spec prompt embeds — both what has
/// been reported and what has been ruled out, under one shared budget.
///
/// Lives here rather than next to the struct so the wording, the per-field
/// truncation, and the prompt that consumes them all stay in one file.
impl AuditStatusSummary {
    pub fn render(&self, caps: &StatusRenderCaps) -> String {
        if self.is_empty() {
            return String::new();
        }
        let mut out = String::new();
        out.push_str(
            "# What this audit has already settled\n\n\
             Other agents are working this project in parallel and have covered the ground below. \
             This is here to point your search somewhere new — it is NOT a filter, and NOT an \
             abandon criterion. Committing a spec never requires clearing anything with this \
             list, and no entry below is grounds for `finalize(status=\"abandoned\")`; the strict \
             criteria in your system prompt still govern that, unchanged. If everything you \
             happened to look at first is covered here, that means you have not looked far \
             enough yet — widen the search, do not give up on the link.\n",
        );
        self.render_reported(caps, &mut out);
        self.render_ruled_out(caps, &mut out);
        out
    }

    /// Bugs the report already contains. Under a cap the NEWEST are kept: a run
    /// reports its coarsest findings early and its subtler ones late, and the
    /// subtle ones are what an agent is most likely to re-derive.
    fn render_reported(&self, caps: &StatusRenderCaps, out: &mut String) {
        if self.reported.is_empty() {
            return;
        }
        let skipped = self.reported.len().saturating_sub(caps.max_reported);
        out.push_str("\n## Already reported — do NOT re-report these\n\n");
        for f in &self.reported[skipped..] {
            out.push_str(&format!(
                "- [#{}] {}\n    where: {}.{}\n    root cause: {}\n",
                f.id,
                f.title.trim(),
                f.primary_contract,
                f.primary_function,
                truncate_text(f.root_cause.trim(), caps.field_chars),
            ));
        }
        if skipped > 0 {
            out.push_str(&format!(
                "- (+{skipped} older reported finding(s) not listed; the report already covers \
                 more than what you see here)\n"
            ));
        }
        out.push_str(
            "\nA spec that restates one of these adds nothing to this audit, however real it is. \
             Treat them as covered ground and prefer a DIFFERENT defect, even in the same contract \
             or the same function — a function that already has one reported bug very often has \
             others (a missing zero check and a missing equality check in one setter are two \
             findings, not one). Only if you find nothing else at all should you fall back to \
             `finalize(status=\"abandoned\")`, and then say which of these already covered what \
             you saw.\n",
        );
    }

    /// Directions a reviewer already investigated and closed. Under a cap the
    /// most-repeated are kept, since those are the walls agents keep hitting.
    fn render_ruled_out(&self, caps: &StatusRenderCaps, out: &mut String) {
        if self.ruled_out.is_empty() {
            return;
        }
        let shown = self.ruled_out.len().min(caps.max_ruled_out);
        out.push_str("\n## Already ruled out — established facts about THIS project\n\n");
        for c in &self.ruled_out[..shown] {
            out.push_str(&format!(
                "- {}\n",
                truncate_text(c.claim.trim(), caps.field_chars)
            ));
            let evidence = c.evidence.trim();
            if !evidence.is_empty() {
                out.push_str(&format!(
                    "    basis: {}\n",
                    truncate_text(evidence, caps.field_chars)
                ));
            }
        }
        let skipped = self.ruled_out.len().saturating_sub(shown);
        if skipped > 0 {
            out.push_str(&format!(
                "- (+{skipped} further conclusion(s) not listed, each covering fewer cases than \
                 those above)\n"
            ));
        }
        out.push_str(
            "\nEach line is a fact about this project that a reviewer established with the full \
             code in hand, after an agent spent a whole pipeline arguing the opposite. One spec — \
             the spec whose ONLY claim to being a bug is the thing a line here contradicts — is \
             not worth writing. Everything else still is.\n\
             \n\
             Read each line as narrowly as it is written, and only against the exact claim it \
             makes. A trusted role being allowed to do X does not make the contract it operates \
             on safe. An accepted token assumption does not close the accounting built on top of \
             it. A documented migration procedure does not mean every function it touches is \
             correct. The overwhelmingly common case is that a defect near one of these facts is \
             NOT covered by it — a different missing check, a different caller reaching the same \
             state, an ordering the fact never speaks to. Treat these as narrow carve-outs in a \
             search space you should still be exploring, not as a map of where the bugs aren't.\n\
             \n\
             A line ending in `…` was shortened for space; its qualifying clause is missing, so \
             read it as narrower than it appears, never broader.\n\
             \n\
             If you believe one of these facts is simply wrong, you need evidence from the code \
             that the reviewer did not have — say so in the spec rather than restating the claim \
             it already rejected.\n",
        );
    }
}

/// What the run has already settled, as its own conversation message.
///
/// Peers working the same contract have no other way to know what has already
/// been covered, and left blind they all converge on that contract's most
/// salient defect (~8.5 links per distinct bug on the metric-* baselines) and
/// re-argue the same rejections (73% of all verdicts on a full tare run).
/// Guidance, not a filter: it never forbids committing a spec, it only says
/// which ground is already covered.
///
/// Separate from [`build_link_prompt`] because it changes on a different clock:
/// the same summary is shared by every link scheduled against one snapshot,
/// while the link block differs for all of them.
pub(super) fn build_status_prompt(status_summary: &str) -> String {
    match status_summary.trim().is_empty() {
        true => "# What this audit has already settled\n\nNothing yet — you are early in the run, so no ground has been covered by a peer.\n".to_string(),
        false => format!("{}\n", status_summary.trim_end()),
    }
}

/// The per-link half of the gen-spec prompt: which link to work, and where to
/// start. Sent as a conversation message rather than as part of the system
/// prompt so that everything ahead of it — the tool definitions and the whole
/// of [`build_system_prompt`] — stays byte-identical across the run.
///
/// `resident_source` selects the opening moves: with the project source in the
/// system prompt there is no signature-index memory and no source-reading tools
/// to suggest, so pointing at them would send the agent after tools it does not
/// have.
pub(super) fn build_link_prompt(
    link: &LinkInput,
    source: LinkSource,
    project_index: &ProjectIndex,
    resident_source: bool,
    extension: Option<&str>,
    target: Option<&LinkScopeTarget>,
    invariants: &[&super::invariants::MinedInvariant],
    test_assertions: &[&super::test_oracle::TestAssertion],
) -> String {
    let extract = &link.extract;
    let historical = &link.historical;
    let finding = &link.finding;
    let (_, finding_header) = link_framing(source);
    let extract_functions = render_extracted_functions(&extract.functions);
    let first_moves = match resident_source {
        true => {
            "1. Find the contracts named in \"Project semantic function anchors\" in the project source at the top of your system prompt, and read the bodies that belong to this semantic. If one of them is listed there as not fitting, fetch it with `read_contract_source`.\n\
                 2. Follow `lookup_call_graph` edges and `lookup_state_variable_xrefs` to surface every function in or near the semantic that could share the historical finding's failure mode (same root-cause family, same kind of state corruption, same failure class) — those relations are not spelled out by the source itself.\n\
                 3. Read every entry under `vulnerability patterns` and `known failure paths` as its own hypothesis about this project, and commit one spec per path you can ground in real code — not one spec for the whole link."
        }
        false => {
            "1. `list_memories` to see what's available.\n\
                  2. `read_memory` on the contracts listed in \"Project semantic function anchors\" — that is your anchor.\n\
                  3. From there, follow `lookup_call_graph` edges and `lookup_state_variable_xrefs` to surface every function in or near the semantic that could share the historical finding's failure mode (same root-cause family, same kind of state corruption, same failure class).\n\
                  4. `read_function_source(contract, function)` for the bodies you actually need to reason about. Avoid loading the same big file twice — the signature index already tells you what each function calls.\n\
                  5. Read every entry under `vulnerability patterns` and `known failure paths` as its own hypothesis about this project, and commit one spec per path you can ground in real code — not one spec for the whole link."
        }
    };
    let index_block = match resident_source {
        true => String::new(),
        false => format!(
            "\nProject static-analysis short-term memory: {} contract entr(ies), {} interface entr(ies). Titles follow `relative_file_path:Contract:line:col`.\n",
            project_index.call_graph.contracts.len(),
            project_index.call_graph.interfaces.len(),
        ),
    };
    let extension_block = match extension {
        Some(text) => format!("\n{text}\n"),
        None => String::new(),
    };
    // The mined invariants are the one hypothesis source in this prompt that is
    // about *this* project rather than about a resemblance, so they sit above
    // the historical finding: an agent that can ground a specification in the
    // target's own identities does not need the historical pattern to still be
    // right about it.
    let invariants_block = super::invariants::render_block(invariants);
    // The project's own tests come right after: the same kind of statement as an
    // invariant, made by the team rather than inferred, and the standard this
    // work is judged against.
    let test_oracle_block = super::test_oracle::render_block(test_assertions);
    // When the engagement names this link's target, the goal is stated here
    // rather than left to the system prompt's general rules: which asset is
    // insured, which entry points this semantic anchors on it, and which
    // impacts the program pays for. The historical below is then explicitly a
    // hint about how such defects arise, not the thing to reproduce — that
    // inversion is the point of the block.
    let (intro, target_block) = match target {
        Some(target) => (
            "Establish whether an actor the program accepts can reach one of its payable impacts \
             through this semantic's entry points, then commit AuditSpecification(s) for the path \
             you find. The historical finding further down tells you *how* this family of defect \
             arises elsewhere — it is not what you have to reproduce.",
            format!("\n{}\n", target.render()),
        ),
        None => (
            "Explore the matched project semantic for any issue *related to* the historical \
             finding below, then commit AuditSpecification(s) for what you find.",
            String::new(),
        ),
    };
    format!(
        r#"# Your link

{intro}
{target_block}{invariants_block}{test_oracle_block}
## Project DeFi semantic (extracted from the current project)
- id: {extract_id}
- category: {extract_category}
- name: {extract_name}
- definition: {extract_definition}
- description: {extract_description}

### Project semantic function anchors (real names from this project)
Use these as the primary source of truth for `setup.contracts` keys and `sequence` contract/function names. They are real extracted project functions, not conceptual labels:
{extract_functions}

## Historical DeFi semantic (from the knowledge graph; matched as encompassing the project's semantic)
- id: {historical_id}
- category: {historical_category}
- name: {historical_name}
- definition: {historical_definition}
- description: {historical_description}

{finding_header}
- id: {finding_id}
- title: {finding_title}
- severity: {finding_severity}
- root cause: {finding_root_cause}
- description: {finding_description}
- vulnerability patterns: {finding_patterns}
- known failure paths: {finding_exploits}
{index_block}{extension_block}
Suggested first moves:
{first_moves}

Reason briefly out loud, then start issuing tool calls. Keep `finalize` for the very end. Per the system prompt, abandoning requires showing which functions you read and why the finding has no related issue here.

Link summary: extract_id={extract_id}, historical_id={historical_id}, finding_id={finding_id}.
"#,
        extract_id = link.extract_id,
        extract_category = extract.category.as_str(),
        extract_name = extract.name,
        extract_definition = extract.definition.trim(),
        extract_description = extract.description.trim(),
        extract_functions = extract_functions,
        historical_id = link.historical_id,
        historical_category = historical.category.as_str(),
        historical_name = historical.name,
        historical_definition = historical.definition.trim(),
        historical_description = link.historical_rendered_description.trim(),
        finding_header = finding_header,
        finding_id = link.finding_id,
        finding_title = finding.title,
        finding_severity = format_severity(finding.severity),
        finding_root_cause = finding.root_cause.trim(),
        finding_description = link.finding_rendered_description.trim(),
        finding_patterns = link.finding_rendered_patterns.trim(),
        finding_exploits = link.finding_rendered_exploits.trim(),
        index_block = index_block,
        extension_block = extension_block,
        intro = intro,
        target_block = target_block,
        first_moves = first_moves,
    )
}

/// Render `link.extract.functions` as the anchor list embedded in the
/// gen-specs prompts. Each line is `- \`{contract}\`.\`{name}\` —
/// \`{signature or "(no recorded signature)"}\``. Empty input emits a
/// single fallback bullet so the prompt slot is never blank.
fn render_extracted_functions(functions: &[ExtractedFunction]) -> String {
    if functions.is_empty() {
        return "- (none recorded; use lookup_call_graph before naming contracts/functions)\n"
            .to_string();
    }
    let mut out = String::new();
    for f in functions {
        let signature = f
            .signature
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("(no recorded signature)");
        out.push_str(&format!(
            "- `{}`.`{}` — `{}`\n",
            f.contract, f.name, signature
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// Link novelty judge
// ---------------------------------------------------------------------------

/// Assemble the novelty judge's system prompt: the rubric wrapped around a
/// rendered snapshot of the project profile, so the judge weighs candidates
/// against what this project actually is. Lives here with the rest of the
/// spec-phase prompt text so the wording is editable without touching the
/// judge's wiring — same convention as [`build_system_prompt`].
pub(super) fn novelty_system(profile: &ProjectProfile, language_prompt_prefix: &str) -> String {
    let mut block = String::new();
    let prefix = language_prompt_prefix.trim();
    if !prefix.is_empty() {
        block.push_str(prefix);
        block.push_str("\n\n");
    }
    block.push_str(&profile.render_for_prompt());
    format!("{NOVELTY_PREAMBLE}\n\n{block}{NOVELTY_RUBRIC}")
}

pub(super) const NOVELTY_PREAMBLE: &str = "You are the Link Novelty Judge for an agentic smart-contract auditing pipeline. The pipeline expands historical vulnerability knowledge into candidate links and runs an expensive agent on each one, which tries to reproduce that historical defect against THIS project. Many unrelated historical findings converge on the same project bug, so most of a run's budget is spent rediscovering findings the report already contains.\n\nYou are shown the canonical findings this run has ALREADY reported — plus, under NEGATIVE `finding_id`s, findings an EARLIER audit of this code reported or the program already declined (a rediscovery of those is a duplicate for the program too) — and a batch of candidate links that have NOT run yet. For each candidate, decide whether running it would rediscover one of those findings.\n\nA `duplicate` verdict does not discard the link — it moves to the back of the queue and a well-funded run still executes it. A wrong `duplicate` costs only ordering; a missed duplicate costs a whole agent pipeline. Reply with strict JSON only, no commentary.";

pub(super) const NOVELTY_RUBRIC: &str = r#"## What makes a candidate a duplicate

Judge on the DEFECT, not on wording and not on location. A candidate is `duplicate` only when its defect is the SAME BUG as one already reported — same broken property, same missing check, one fix kills both.

  duplicate  The candidate's root cause maps onto a reported finding's root cause. Different historical projects, different phrasing, different severity, and different naming do NOT make it new.

  new        Anything else, including — and this is the common case —
             - SAME contract and SAME function as a reported finding, but a DIFFERENT defect. A hot function typically carries several distinct bugs: a missing zero-address check, a value-equality check that is also missing, an unbounded numeric parameter, and a stale-config path can all live in one setter and are FOUR findings, not one.
             - a defect whose surface is not represented in the reported set at all.

`primary_contract` / `primary_function` in the reported set are locating hints, not the dedup key. Matching them is NOT sufficient evidence of duplication — compare root causes, and only call `duplicate` when the underlying defect is the same.

## Method

- Read `defect_root_cause` and `defect_detail` first: they describe the code shape of the defect.
- Look for a reported finding whose `root_cause` describes that same defect.
- Ask yourself: would one patch fix both? If no, it is `new`.
- When genuinely torn, answer `new`. Missing a real bug is the expensive error; re-checking a duplicate merely costs one pipeline.

## Output

Emit one entry per candidate shown — no silent omission:

{"verdicts":[{"candidate_id":0,"verdict":"duplicate","duplicate_of_finding_id":41,"reason":"..."},{"candidate_id":1,"verdict":"new","reason":"..."}]}

Hard rules:
- `candidate_id` must be one of the ids shown in this batch.
- `duplicate_of_finding_id` is REQUIRED when `verdict` is `duplicate`, and must be a `finding_id` from `reported_findings`. Never invent one.
- `reason` is one short sentence naming the shared defect (for `duplicate`) or what makes it distinct (for `new`)."#;
