//! The scope judge: the last check a specification meets at commit, for the
//! program exclusions no keyword can decide.
//!
//! The deterministic gates (impact class, assets, enforced known issues and
//! exclusions, deployment, economic claim) answer questions a field can answer.
//! "Economically impractical", "DoS bypassable by other means", "relies on the
//! victim's own mistake" and "subaccounts are trusted by design" are judgments
//! over the whole claim; left to the authoring agent they were ignored (live
//! GMX run: 13 of 13 committed specifications fell under such an exclusion).
//! This judge reads the finished specification against the program's own
//! rules and refuses what the program would not pay for. A refusal reaches the
//! agent like any gate refusal: fix the claim or abandon.

use color_eyre::eyre::Result;
use llmy::client::client::LLM;
use serde::Deserialize;

use crate::scope::{KnownIssue, ProgramScope, TriageProfile};
use crate::types::AuditSpecification;

/// Characters of the rendered specification shown to the judge.
const MAX_SPEC_CHARS: usize = 40_000;

const SYSTEM_PROMPT: &str = r#"You are the triage judge of a bug bounty program. A security researcher's tooling has produced a candidate vulnerability specification. Decide whether the candidate, if proven exactly as written, would be accepted and paid under the program's rules below. Be strict: a candidate the program would close as out of scope, invalid or informational wastes a costly proof.

Refuse (admit = false) when ANY of these holds:
- It needs a trusted or privileged party (admin, keeper, governance, a role holder, an oracle signer, a delegate the victim chose to trust) to act maliciously, be compromised, or behave outside its job. Honest keepers doing their job may be part of the sequence.
- Its setup or pre-attack state needs a protocol parameter, role assignment, feature flag, or market/vault configuration that only a privileged role sets, to hold a value that enables the attack (a minimum or cap set to zero, a safeguard disabled, an unusual listing, a fresh vault or market a keeper must create). Refuse unless the specification shows the live or default deployment already has that value; the program assumes honest configuration.
- It relies on the victim's own mistake: sending funds outside the documented flow (e.g. outside the multicall that consumes them), signing or re-granting an approval, re-trusting an address, misconfiguring their own account.
- The program states the behaviour is by design or trusted (an exclusion below says so).
- It is economically impractical: the attacker's cost is comparable to or larger than the damage, or the damage is negligible (dust, a small bounded fee, rounding of a few units).
- It is a denial of service that is temporary, self-inflicted, limited to the attacker's own funds, or bypassable (another market or path works, a keeper can retry with more gas, an admin can unblock without loss).
- It depends on incorrect oracle data, delayed price feeds, or manipulating prices on external venues, or on state no unprivileged actor can create.
- Its impact is not one the program lists, or the specification's own text shows the claimed loss does not happen (for example the "stuck" state is reverted in the same transaction, or both sides of an identity move together).
- It restates a known issue: compare it against "Known issues relevant to this candidate" below; the same function and the same root cause is a duplicate even when the wording, impact or entry point differs. Cite the id.
- It restates an exclusion listed below.
- When you cannot tell whether the attack needs anything beyond what an unprivileged user controls, refuse and say what is missing.

Admit (admit = true) only when an unprivileged attacker, through permissionless steps, causes a listed impact to a party other than themselves, without the victim's error, at a cost below the damage.

You have NOT read the project's code. Ground every refusal in one of two things only:
1. the program's rules below (an exclusion, a trusted role, the victim's own mistake, economic impracticality, an unlisted impact, a known issue), applied to what the specification says; or
2. a contradiction inside the specification's own text (its arithmetic, its quoted code or NatSpec, its stated setup versus its stated sequence).
Never refuse on your own belief about how the code behaves ("this function reverts", "that value is capped") unless the specification itself states or quotes it. If a refusal would need such a belief, admit and let the proof settle it. When the specification's own text shows an exclusion applies, refuse.

Respond with strict JSON only, in exactly this shape:
{"admit": true|false, "rule": "<the exclusion id or a short name of the reason; empty when admitting>", "reason": "<one or two sentences citing the specification>"}"#;

/// The contest judge: grades instead of refusing on a bounty's exclusions.
/// The program's rules (with [`crate::scope::CONTEST_SEVERITY_RUBRIC`]) are
/// appended below it, as for the bounty prompt.
const CONTEST_SYSTEM_PROMPT: &str = r#"You are the judge of a security audit competition. A researcher's tooling has produced a candidate vulnerability specification. Decide the severity the competition's judge would give it if it were proven exactly as written, using the program's rules and severity criteria below. Only High and Medium are rewarded: a candidate that would be graded Low, Informational or invalid wastes a costly proof.

Refuse (admit = false) when ANY of these holds:
- It needs a trusted or privileged party the program names (an owner, admin, approved operator or transformer, a role holder) to act maliciously, be compromised, or misconfigure — or its setup needs a parameter or configuration only such a party sets to hold a value the honest deployment would not use. (At most Low.) A trusted party doing its documented job honestly, which then triggers a defect in the code, does not count: grade the defect.
- It is ultimately the victim's own mistake that a front-end can easily manage. (At most Informational.)
- It restates an exclusion listed below, or behaviour the program, the code's comments or the specification's own quotes show to be intended.
- It restates a known issue: compare it against "Known issues relevant to this candidate" below; the same function and the same root cause is a duplicate even when the wording, impact or entry point differs. Cite the id. An incomplete fix, a bypass of the fix or a regression the fix introduced is new — admit it on its merits.
- Its loss is dust or rounding, or it depends on a non-standard ERC-20's behaviour. (At most Low.)
- The specification's own text contradicts its claimed impact, under the severity criteria's rules on that: a freeze the owner can still exit, a DoS of a helper with another working path, a sequence that misuses a documented parameter, a revert reverted in the same transaction, both sides of an identity moving together, a value flow in which nobody but the caller ends up short. Grade the impact the text does support instead of the label, and refuse only if that grade is Low or below.
- Graded through the matrix, it comes out Low or Informational.

Do NOT refuse merely because the denial of service is temporary, the loss needs specific but realistic market or timing conditions, the attack costs the attacker a lot, or the defect sits in how the code handles an oracle, sequencer-uptime or TWAP edge state it claims to handle: grade those through the matrix (they usually lower the likelihood, not void the finding).
Do NOT refuse because the specification has not yet quantified the loss, leaves a state section empty, or has a step the proof still has to demonstrate: it is a candidate for a proof, not the proof. Grade the claim as written unless its own text contradicts it.

You have NOT read the project's code. Ground every refusal in one of two things only:
1. the program's rules and severity criteria below, applied to what the specification says; or
2. a contradiction inside the specification's own text (its arithmetic, its quoted code or NatSpec, its stated setup versus its stated sequence).
Never refuse on your own belief about how the code behaves unless the specification itself states or quotes it. If a refusal would need such a belief, grade the claim as written and let the proof settle it.

Respond with strict JSON only, in exactly this shape:
{"impact": "high|medium|low", "likelihood": "high|medium|low", "severity": "high|medium|low|informational|invalid", "admit": true|false, "rule": "<the exclusion id, known-issue id or a short name of the reason; empty when admitting>", "reason": "<one or two sentences citing the specification>"}
admit is true exactly when severity is high or medium."#;

/// The judge's answer.
#[derive(Debug, Clone, Deserialize)]
pub struct JudgeVerdict {
    pub admit: bool,
    #[serde(default)]
    pub rule: String,
    #[serde(default)]
    pub reason: String,
    /// Contest grading; empty under the bounty profile.
    #[serde(default)]
    pub impact: String,
    #[serde(default)]
    pub likelihood: String,
    #[serde(default)]
    pub severity: String,
}

impl JudgeVerdict {
    /// Hold a contest verdict to the matrix: only High and Medium are paid,
    /// whatever `admit` says. A missing grade is a refusal, because an
    /// ungraded admit is the judge skipping the question it is asked.
    fn enforce_contest_grade(mut self) -> Self {
        let severity = self.severity.trim().to_ascii_lowercase();
        if self.admit && !matches!(severity.as_str(), "high" | "medium") {
            self.admit = false;
            if self.rule.trim().is_empty() {
                self.rule = if severity.is_empty() {
                    "severity-missing".to_string()
                } else {
                    format!("severity-{severity}")
                };
            }
        }
        self
    }
}

pub struct ScopeJudge {
    llm: LLM,
    system_prompt: String,
    /// Every known issue of the engagement; the prompt shows the ones that
    /// touch this candidate's contracts and functions, with their summaries.
    known_issues: Vec<KnownIssue>,
    contest: bool,
}

impl std::fmt::Debug for ScopeJudge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScopeJudge").finish_non_exhaustive()
    }
}

impl ScopeJudge {
    pub fn new(llm: LLM, scope: &ProgramScope) -> Self {
        let contest = scope.triage_profile == TriageProfile::Contest;
        let preamble = if contest { CONTEST_SYSTEM_PROMPT } else { SYSTEM_PROMPT };
        let system_prompt = format!(
            "{preamble}\n\n# The program's rules\n{}",
            scope.render_for_prompt()
        );
        Self {
            llm,
            system_prompt,
            known_issues: scope.known_issues.clone(),
            contest,
        }
    }

    /// Judge one finished specification.
    pub async fn judge(&self, spec: &AuditSpecification) -> Result<JudgeVerdict> {
        let user_prompt = format!(
            "{}\n\n{}",
            render_spec(spec, self.contest),
            render_relevant_known_issues(spec, &self.known_issues)
        );
        let verdict: JudgeVerdict = self
            .llm
            .prompt_json_with_retry(&self.system_prompt, &user_prompt, Some("scope-judge"), None, None)
            .await?;
        Ok(if self.contest {
            verdict.enforce_contest_grade()
        } else {
            verdict
        })
    }
}

/// Known issues shown to the judge per candidate. The scope's own prompt lists
/// only the first few dozen of hundreds; a duplicate is only visible when the
/// matching issue is in front of the judge, so pick by overlap instead.
const MAX_RELEVANT_KNOWN_ISSUES: usize = 20;
const KNOWN_ISSUE_SUMMARY_CHARS: usize = 500;

/// Identifier-like words (contract and function names) of a text.
fn identifiers(text: &str) -> std::collections::BTreeSet<String> {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|word| word.len() >= 6 && word.chars().any(|c| c.is_ascii_uppercase() || c == '_') || word.len() >= 10)
        .map(|word| word.to_ascii_lowercase())
        .collect()
}

fn render_relevant_known_issues(spec: &AuditSpecification, issues: &[KnownIssue]) -> String {
    let contracts: std::collections::BTreeSet<String> = spec
        .sequence_contracts()
        .iter()
        .map(|c| c.rsplit('/').next().unwrap_or(c).trim_end_matches(".sol").to_ascii_lowercase())
        .collect();
    let spec_words = identifiers(&spec.searchable_text());
    let mut scored: Vec<(usize, &KnownIssue)> = issues
        .iter()
        .filter_map(|issue| {
            let text = format!("{} {}", issue.title, issue.summary);
            let contract_hits = issue
                .contracts
                .iter()
                .filter(|c| contracts.contains(&c.to_ascii_lowercase()))
                .count();
            let word_hits = identifiers(&text).intersection(&spec_words).count();
            let score = contract_hits * 3 + word_hits;
            (score >= 2).then_some((score, issue))
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0));
    if scored.is_empty() {
        return "# Known issues relevant to this candidate
(none overlap its contracts or functions)".to_string();
    }
    let mut out = String::from("# Known issues relevant to this candidate
");
    for (_, issue) in scored.into_iter().take(MAX_RELEVANT_KNOWN_ISSUES) {
        let summary: String = issue.summary.chars().take(KNOWN_ISSUE_SUMMARY_CHARS).collect();
        out.push_str(&format!("- [{}] {}: {}\n", issue.id, issue.title.trim(), summary.trim()));
    }
    out
}

fn render_spec(spec: &AuditSpecification, contest: bool) -> String {
    let mut body = serde_json::to_string_pretty(spec).unwrap_or_else(|_| spec.summary.clone());
    if body.len() > MAX_SPEC_CHARS {
        let mut cut = MAX_SPEC_CHARS;
        while !body.is_char_boundary(cut) {
            cut -= 1;
        }
        body.truncate(cut);
        body.push_str("\n… (truncated)");
    }
    let question = if contest {
        "What severity would the competition's judge give this candidate, and is it High or Medium?"
    } else {
        "Is this candidate payable under the program's rules?"
    };
    format!("# Candidate specification\n```json\n{body}\n```\n\n{question}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_reads_the_documented_shape_and_tolerates_missing_fields() {
        let refuse: JudgeVerdict = serde_json::from_str(
            r#"{"admit": false, "rule": "gmx:subaccounts-trusted", "reason": "the attacker is the victim's delegate"}"#,
        )
        .unwrap();
        assert!(!refuse.admit);
        assert_eq!(refuse.rule, "gmx:subaccounts-trusted");
        let admit: JudgeVerdict = serde_json::from_str(r#"{"admit": true}"#).unwrap();
        assert!(admit.admit && admit.rule.is_empty());
    }

    #[test]
    fn relevant_known_issues_are_picked_by_contract_and_function_overlap() {
        let mut spec = AuditSpecification::default();
        spec.summary = "GlpMigrator._redeemGlp lets the caller redeem at the reduced fee".to_string();
        let issue = |id: &str, contract: &str, title: &str| KnownIssue {
            id: id.to_string(),
            title: title.to_string(),
            contracts: vec![contract.to_string()],
            summary: format!("{title} in {contract}"),
            ..Default::default()
        };
        let issues = vec![
            issue("GLPM-1", "GlpMigrator", "Reduced Redemption Fees Gamed via _redeemGlp"),
            issue("OTHER-1", "OrderHandler", "Unrelated order bug"),
        ];
        let block = render_relevant_known_issues(&spec, &issues);
        assert!(block.contains("GLPM-1"), "{block}");
        assert!(!block.contains("OTHER-1"), "{block}");
    }

    #[test]
    fn long_specifications_are_cut_on_a_char_boundary() {
        let mut spec = AuditSpecification::default();
        spec.summary = "é".repeat(MAX_SPEC_CHARS);
        let rendered = render_spec(&spec, false);
        assert!(rendered.contains("(truncated)"));
    }

    fn verdict(json: &str) -> JudgeVerdict {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn contest_verdicts_pay_only_high_and_medium() {
        let medium = verdict(r#"{"admit": true, "impact": "high", "likelihood": "low", "severity": "medium"}"#)
            .enforce_contest_grade();
        assert!(medium.admit);
        let low = verdict(r#"{"admit": true, "impact": "low", "likelihood": "medium", "severity": "Low"}"#)
            .enforce_contest_grade();
        assert!(!low.admit);
        assert_eq!(low.rule, "severity-low");
        let ungraded = verdict(r#"{"admit": true}"#).enforce_contest_grade();
        assert!(!ungraded.admit);
        assert_eq!(ungraded.rule, "severity-missing");
        let refused = verdict(r#"{"admit": false, "severity": "high", "rule": "R-18"}"#).enforce_contest_grade();
        assert!(!refused.admit);
        assert_eq!(refused.rule, "R-18");
    }

    #[test]
    fn the_scope_file_picks_the_judge() {
        let bounty = ProgramScope::parse(r#"{"program": "p"}"#, std::path::Path::new("knowdit-scope.json")).unwrap();
        assert_eq!(bounty.triage_profile, TriageProfile::Bounty);
        let contest = ProgramScope::parse(
            r#"{"program": "p", "triage_profile": "contest"}"#,
            std::path::Path::new("knowdit-scope.json"),
        )
        .unwrap();
        assert_eq!(contest.triage_profile, TriageProfile::Contest);
        assert!(contest.render_for_prompt().contains("Likelihood Low | Medium | Low | Informational"));
        assert!(!bounty.render_for_prompt().contains("Informational"));
        assert!(render_spec(&AuditSpecification::default(), true).contains("High or Medium"));
    }
}
