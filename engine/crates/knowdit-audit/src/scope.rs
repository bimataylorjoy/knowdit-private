//! The engagement's audit-program scope, as data, plus the deterministic
//! predicate that decides whether one specification is a candidate the
//! program could pay for.
//!
//! Everything else in this crate judges a candidate from the inside: is the
//! defect real (verdict grader), what does it cost (severity grader), is the
//! proof against deployed code (admissibility). None of those can answer the
//! question a bounty program actually asks, because it is not a property of
//! the code — it is a property of the *program*: does this loss land on an
//! asset the program insures, by an actor it considers an attacker, in an
//! impact it says it pays for, on a chain it deploys, in code no earlier
//! report has already covered. Programs publish exactly that, per engagement,
//! and the same finding flips between payable and unpaid depending on the
//! answer.
//!
//! So the answer is carried as data ([`ProgramScope`]) and applied as a
//! predicate ([`ProgramScope::admit`]) the moment a specification is
//! committed — before the harness agent, the fuzzer and three graders are
//! paid for. The alternative, which is what this module replaces, is asking
//! a reviewer agent to weigh free-text scope prose after all of that money
//! is already spent, with no way to tell a wrong call from a right one.
//!
//! The predicate is deliberately conservative in one direction only: it must
//! not pass a specification the program's own rules exclude, and it must not
//! invent an exclusion out of a signal the program never stated. Every rule
//! is therefore keyed to a field the scope file actually declares, and every
//! refusal names the requirement it fails ([`ScopeVerdict::clause`]) so the
//! record reads as an investigation result rather than a silent drop.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use color_eyre::eyre::{Result, WrapErr};
use knowdit_repo_model::ReviewSeverity;
use serde::{Deserialize, Serialize};

use crate::types::{AttackerPrivilege, AuditSpecification, ImpactClass};

/// Conventional filenames the pipeline discovers when `--scope-config` is
/// not given. JSON first: the file is consumed by machines, and every other
/// per-run input in this pipeline (deployment artifacts, spec JSON) is JSON.
pub const SCOPE_FILENAMES: [&'static str; 2] = ["knowdit-scope.json", "knowdit-scope.toml"];

/// Where the platform's default exclusion list was read from, carried on every
/// entry that came from it so a refusal can point at the source.
pub const SELF_DEFAULT_SOURCE: &'static str =
    "https://immunefi.com/common-vulnerabilities-to-exclude/";

/// Whose rules decide a candidate the program's impact table does not list.
///
/// Programs declare this and it changes the answer: under primacy of impact
/// an unlisted impact can still be paid (the program weighs the impact
/// itself), while under primacy of rules the published table is exhaustive —
/// an impact that is not on it is not payable however real it is. Encoding
/// it here keeps that call out of a prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Primacy {
    /// An unlisted impact is still admissible; the table is a floor.
    Impact,
    /// The published impact table is exhaustive.
    #[default]
    Rules,
}

impl Primacy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Impact => "impact (an unlisted impact may still be paid)",
            Self::Rules => "rules (the published impact list is exhaustive)",
        }
    }
}

/// Which bar a candidate has to clear to be worth a proof.
///
/// A bug bounty pays for a loss on deployed code and closes everything else,
/// so its triage refuses temporary DoS, oracle edge states and costly attacks
/// outright. An audit competition grades every valid defect through an
/// impact × likelihood matrix and pays High and Medium: the same candidates
/// are graded there instead of refused, while the proof that the claimed
/// impact really happens matters as much as ever.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriageProfile {
    /// Bug bounty on deployed code (Immunefi-style). The default.
    #[default]
    Bounty,
    /// Audit competition on a commit (Cantina's default severity criteria).
    Contest,
}

/// The severity criteria a [`TriageProfile::Contest`] engagement is graded
/// by: Cantina's default competition criteria, plus the most common reason a
/// valid mechanism is still rejected in triage (an impact that is labelled
/// rather than shown).
pub const CONTEST_SEVERITY_RUBRIC: &str = r#"## How this engagement grades a finding (audit competition, Cantina's default criteria)
Only High and Medium are rewarded; Low and Informational are not. Severity is impact × likelihood:
- Impact High: funds can be lost (theft, permanent freeze, insolvency). Impact Medium (not defined by the criteria; read it as): a core function stops working for its legitimate users, funds are unavailable for a meaningful time, or value is lost only partially or under conditions. Impact Low: annoying but bearable.
- Likelihood High: any participant can trigger it. Medium: conditional but realistic — a market state, timing or deployment setting the protocol will actually meet. Low: needs the stars to align, or nobody has an incentive to do it.

| | Impact High | Impact Medium | Impact Low |
|---|---|---|---|
| Likelihood High | High | High | Medium |
| Likelihood Medium | High | Medium | Low |
| Likelihood Low | Medium | Low | Informational |

When in doubt, ask what happens to the protocol if the bug is not fixed: a catastrophic scenario anyone can trigger or that occurs naturally is High; if the protocol can function without the fix, it is likely Low. A fix that goes against the protocol's design philosophy makes it at most Informational.
Caps from the criteria: a user error the front-end can easily manage is at most Informational; anything needing admin access (or an equivalent trusted role) is at most Low; behaviour of non-standard ("weird") ERC-20 tokens is at most Low; dust or rounding losses are at most Low; an issue acknowledged in a previous report is invalid — but an incomplete fix, a fix bypass or a regression introduced by a fix is new.
Unlike a bug bounty, these are graded, not refused on sight: a temporary denial of service, a loss that needs specific but realistic market or timing conditions, an attack whose cost is high relative to its gain (that lowers likelihood), and a defect in how the code handles oracle, sequencer-uptime or TWAP edge states it claims to handle. Wrong data from a trusted oracle itself is still not a finding.
An honest trusted party doing its documented job (an owner updating a parameter through the intended setter, a keeper running a scheduled call) that triggers a defect in the code is not an admin-access issue: the defect is graded. Only a trusted party acting maliciously, being compromised or misconfiguring is capped.
The claimed impact must be the one that actually happens — this is where a real mechanism most often fails. The proof is written later, so missing numbers or an unproven step are not a reason to refuse; what is, is the claim contradicting itself or the protocol's documented behaviour:
- a "freeze" means the rightful owner has no path left to recover the funds; if repaying, withdrawing or another exit still works, nothing is frozen;
- a denial of service of an optional or peripheral helper whose users keep another working path is at most Low;
- every parameter of the sequence must be used as the NatSpec, the docs or the team's own tests define it; a sequence that passes a value the documented usage never would (an amount the docs size from a static call set to type(uint256).max, say) and reports the resulting revert proves nothing;
- behaviour the code, its comments, the docs or the team's tests show to be intended is not a finding."#;

/// One thing the program insures: a deployed contract, its chain, and what
/// is at risk there.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeAsset {
    /// Contract name as the project's source and the specification name it
    /// (`VaultCore`); a proxy's implementation is matched by the same stem
    /// rule the deployment book uses.
    pub name: String,
    /// Chain the program insures the deployment on.
    #[serde(default)]
    pub chain_id: Option<u64>,
    /// Deployed address, when the program lists one. Advisory: a scope file
    /// is often written before the address is known, and the repository's own
    /// deployment tree is the better source once it exists.
    #[serde(default)]
    pub address: Option<String>,
    /// What is at risk there, in USD. Drives the value floor and, downstream,
    /// the severity conversation on a program that tiers on value at risk.
    #[serde(default)]
    pub value_usd: Option<u64>,
}

/// A defect a previous report already established. Programs exclude these
/// explicitly, and they are the cheapest possible finding to avoid: they cost
/// a full pipeline to rediscover and cannot be paid.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnownIssue {
    /// Program-side identifier, for the report line.
    pub id: String,
    /// One-line description, rendered into refusal reasons.
    pub title: String,
    /// Contracts the prior report covers. Empty means "any contract".
    #[serde(default)]
    pub contracts: Vec<String>,
    /// Substrings that must appear in the specification's own text for this
    /// issue to be considered the same defect. Empty means the contract match
    /// alone is enough — use that only when the program's list is that
    /// specific.
    #[serde(default)]
    pub match_any: Vec<String>,
    /// Whether the commit gate refuses specifications matching this issue.
    /// `false` keeps it as context only — rendered to every agent and to the
    /// novelty judge — for an issue that could not be pinned to a specific
    /// function, where enforcing on the contract alone would refuse every new
    /// finding on it.
    #[serde(default = "default_true")]
    pub enforce: bool,
    /// A short account of the defect (the report's own words), for readers
    /// that dedup by meaning rather than by name — the novelty judge.
    #[serde(default)]
    pub summary: String,
}

/// One exclusion that applies to this engagement.
///
/// Two sources feed the same list: what the program publishes on its own page
/// ("impacts requiring a privileged role are out of scope", "issues already
/// documented in a published audit"), and the platform's *default* exclusion
/// list, which every program inherits unless it says otherwise. Both are read
/// where the scope is read, so a candidate that falls under either is never
/// built — the point is to not spend a harness run on something the program
/// does not buy.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutOfScopeRule {
    /// Stable id, e.g. `immunefi:privileged-addresses` or a program's own
    /// `OOS-3`. Referenced by the gate's refusals so a decision can be traced
    /// back to the line it applied.
    pub id: String,
    /// The exclusion as published. Kept verbatim, because a paraphrase is a
    /// different rule.
    pub title: String,
    /// Where it came from, when the program published a link.
    #[serde(default)]
    pub source: Option<String>,
    /// Contracts the exclusion is about, for a rule the commit gate enforces.
    /// Empty means "any contract".
    #[serde(default)]
    pub contracts: Vec<String>,
    /// Substrings of the specification's own text that put it under this
    /// exclusion. An enforced rule needs at least one: an exclusion matched on
    /// contracts alone would refuse every finding on them.
    #[serde(default)]
    pub match_any: Vec<String>,
    /// Refuse matching specifications at commit, the way an enforced known
    /// issue is. Off by default: most exclusions ("economically impractical",
    /// "DoS bypassable by other means") are judgments no keyword can make, and
    /// stay context for the agents and the scope judge.
    #[serde(default)]
    pub enforce: bool,
}

/// One row of the program's own severity table.
///
/// Programs publish severity as a table of *their* tiers over *their* named
/// impacts; the same defect is `critical` on one program and `medium` on
/// another. Transcribing it is what lets the report speak the program's
/// language instead of an invented universal scale — and lets the pipeline
/// notice when it is about to claim more than the program's table allows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeverityRule {
    /// The tier as the program names it (`critical`, `high`, …).
    pub tier: String,
    /// Which of this pipeline's impact classes land on that tier, in the
    /// program's wording mapped onto ours by whoever transcribed the table.
    pub impacts: Vec<ImpactClass>,
    /// Value-at-risk floor the program publishes for the tier, when it does.
    #[serde(default)]
    pub min_value_usd: Option<u64>,
    /// Which of this pipeline's report tiers the program's tier corresponds to.
    ///
    /// Transcribed, never inferred: a program's `critical` is not
    /// automatically our `High`, and guessing the correspondence is exactly the
    /// invention this file exists to avoid. Absent means the tier is reported
    /// under the program's own name and the pipeline's severity is left alone.
    #[serde(default)]
    pub report_severity: Option<ReviewSeverity>,
}

/// What a program said it pays for, transcribed per engagement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramScope {
    /// Program name as the platform shows it (e.g. `Immunefi: Example Protocol`).
    pub program: String,
    #[serde(default)]
    pub primacy: Primacy,
    /// Chain the engagement is scoped to. When set, a specification is
    /// required to be reproducible there (see the deployment rule).
    #[serde(default)]
    pub chain_id: Option<u64>,
    #[serde(default)]
    pub assets: Vec<ScopeAsset>,
    /// Impacts the program pays for. Empty means "not restricted here", which
    /// is the honest reading of a scope file that does not say.
    #[serde(default)]
    pub impacts_in_scope: Vec<ImpactClass>,
    /// Exclusions the program publishes itself, beyond the platform defaults.
    ///
    /// Transcribed from the program page's own out-of-scope section, verbatim:
    /// its wording is the rule, and a summary of it is a different rule.
    #[serde(default)]
    pub out_of_scope: Vec<OutOfScopeRule>,
    /// Whether the platform's default exclusion list applies.
    ///
    /// Defaults to true, because that is what Immunefi states: these are the
    /// impacts every program is recommended to exclude unless it says
    /// otherwise. A program that publishes a narrower list cannot be inferred
    /// from the default one — turn this off in the scope file when the program
    /// has its own, and transcribe theirs into `out_of_scope`.
    #[serde(default = "default_true")]
    pub use_platform_defaults: bool,
    /// The program's own severity table, in the order it publishes it (most
    /// severe first). Empty when the program's severity story is not
    /// transcribed, which leaves the pipeline's own tiers in charge — the
    /// behaviour before the table existed.
    #[serde(default)]
    pub severity_table: Vec<SeverityRule>,
    /// Impacts the program names as excluded. Applies under either primacy:
    /// an exclusion is stated, not implied.
    #[serde(default)]
    pub excluded_impacts: Vec<ImpactClass>,
    /// Value floor, in USD. A specification whose matched asset is worth less
    /// than this cannot clear the program's minimum.
    #[serde(default)]
    pub min_asset_value_usd: Option<u64>,
    /// The program requires an unprivileged attacker. Defaults to true: it is
    /// the standing rule of every program of this kind, and the commonest
    /// reason a real defect is unpaid.
    #[serde(default = "default_true")]
    pub requires_unprivileged_actor: bool,
    /// The program pays for privileged-actor impacts too (some do, explicitly).
    /// Only then may an exploit that needs an existing role pass.
    #[serde(default)]
    pub allow_privileged_impacts: bool,
    /// Contract-name patterns no specification may be built on: mocks, test
    /// doubles, deployment scaffolding. `*` matches any run of characters.
    #[serde(default)]
    pub excluded_contracts: Vec<String>,
    #[serde(default)]
    pub known_issues: Vec<KnownIssue>,
    /// Assumptions the program states it will not question (honest oracle,
    /// standard ERC-20, a trusted multisig). Rendered into the prompts; not a
    /// predicate, because "assumes X" is only decidable by reading the
    /// specification against it.
    #[serde(default)]
    pub assumptions: Vec<String>,
    /// Bounty or contest triage; see [`TriageProfile`].
    #[serde(default)]
    pub triage_profile: TriageProfile,
}

fn default_true() -> bool {
    true
}

impl ProgramScope {
    /// Load a scope file. Format follows the extension: `.toml` is parsed as
    /// TOML, anything else as JSON.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .wrap_err_with(|| format!("failed to read scope config {}", path.display()))?;
        Self::parse(&text, path)
    }

    /// Parse scope text. Split from [`Self::load`] so the fixture tests do not
    /// touch the filesystem.
    pub fn parse(text: &str, path: &Path) -> Result<Self> {
        let is_toml = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("toml"));
        // Both parsers fail with their own error type; the message is what
        // matters here, and it is wrapped with the path either way.
        let parsed: std::result::Result<Self, String> = if is_toml {
            toml::from_str(text).map_err(|error| error.to_string())
        } else {
            serde_json::from_str(text).map_err(|error| error.to_string())
        };
        let scope: Self = parsed.map_err(|error| {
            color_eyre::eyre::eyre!("failed to parse scope config {}: {error}", path.display())
        })?;
        if scope.assets.is_empty() && scope.impacts_in_scope.is_empty() {
            tracing::warn!(
                scope = %path.display(),
                "scope config declares neither assets nor in-scope impacts; the gate can only \
                 enforce the standing rules (unprivileged actor, exclusions, known issues, \
                 deployment)"
            );
        }
        Ok(scope)
    }

    /// Find a scope file in `repo_root` when the caller did not name one.
    pub fn discover(repo_root: &Path) -> Option<PathBuf> {
        SCOPE_FILENAMES
            .iter()
            .map(|name| repo_root.join(name))
            .find(|path| path.is_file())
    }

    /// Resolve the engagement scope for a run: the explicit path when given,
    /// otherwise whatever sits in the repository. `None` means the engagement
    /// declared no scope, and the gate is not installed at all.
    pub fn resolve(repo_root: &Path, explicit: Option<&Path>) -> Result<Option<Self>> {
        let path = match explicit {
            Some(path) => path.to_path_buf(),
            None => match Self::discover(repo_root) {
                Some(path) => path,
                None => return Ok(None),
            },
        };
        tracing::info!("[scope] engagement scope loaded from {}", path.display());
        Ok(Some(Self::load(&path)?))
    }

    /// Fold one more known issue into the scope, replacing an entry with the
    /// same `id` — a re-recorded verdict corrects the earlier one rather than
    /// piling a second copy beside it. Returns true when an entry was replaced.
    ///
    /// This is how a program's answer becomes ground the gate enforces: the
    /// rejection arrives long after the scope file was written, and once it is
    /// in here every later run refuses that defect before spending a harness
    /// agent on it, instead of merely warning the agent about it.
    pub fn add_known_issue(&mut self, issue: KnownIssue) -> bool {
        match self
            .known_issues
            .iter_mut()
            .find(|known| known.id == issue.id)
        {
            Some(existing) => {
                *existing = issue;
                true
            }
            None => {
                self.known_issues.push(issue);
                false
            }
        }
    }

    /// Write the scope back, atomically, in the format its path is named in.
    ///
    /// Atomic because a half-written scope file silently changes what the gate
    /// enforces: the next run would read a truncated impact list and start
    /// admitting candidates the program excludes.
    pub fn write(&self, path: &Path) -> Result<()> {
        let is_toml = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("toml"));
        let text = match is_toml {
            true => toml::to_string(self)
                .map_err(|error| color_eyre::eyre::eyre!("failed to serialize scope: {error}"))?,
            false => serde_json::to_string_pretty(self)
                .wrap_err("failed to serialize scope as JSON")?,
        };
        let temp = path.with_extension("tmp");
        std::fs::write(&temp, format!("{text}\n"))
            .wrap_err_with(|| format!("failed to write {}", temp.display()))?;
        std::fs::rename(&temp, path)
            .wrap_err_with(|| format!("failed to publish {}", path.display()))?;
        Ok(())
    }

    /// Immunefi's "Common Vulnerability Exclusion List" — the impacts the
    /// platform recommends every program exclude, transcribed from
    /// <https://immunefi.com/common-vulnerabilities-to-exclude/> (read
    /// 2026-09-20). Only the entries that can apply to a smart-contract
    /// engagement are here: the website/application section of that page (XSS,
    /// HTTP headers, SSL, captcha, …) cannot be reached by a specification
    /// about on-chain state, so rendering it would be noise in every prompt.
    ///
    /// Ids are stable so a refusal can cite the line it applied. Verbatim
    /// wording is deliberate: "attacks requiring access to privileged
    /// addresses" and "privileged roles are assumed to act in good faith" are
    /// different exclusions, and a program may publish the second while the
    /// platform default supplies the first.
    const PLATFORM_DEFAULTS: [(&'static str, &'static str, &'static str); 13] = [
        (
            "immunefi:privileged-addresses",
            "Impacts caused by attacks requiring access to privileged addresses (governance, strategist) except in such cases where the contracts are intended to have no privileged access to functions that make the attack possible",
            SELF_DEFAULT_SOURCE,
        ),
        (
            "immunefi:leaked-keys",
            "Impacts caused by attacks requiring access to leaked keys/credentials",
            SELF_DEFAULT_SOURCE,
        ),
        (
            "immunefi:self-exploited",
            "Impacts requiring attacks that the reporter has already exploited themselves, leading to damage",
            SELF_DEFAULT_SOURCE,
        ),
        (
            "immunefi:stablecoin-depeg",
            "Impacts relying on attacks involving the depegging of an external stablecoin where the attacker does not directly cause the depegging due to a bug in code",
            SELF_DEFAULT_SOURCE,
        ),
        (
            "immunefi:secrets-in-github",
            "Mentions of secrets, access tokens, API keys, private keys, etc. in Github will be considered out of scope without proof that they are in-use in production",
            SELF_DEFAULT_SOURCE,
        ),
        (
            "immunefi:best-practice",
            "Best practice recommendations",
            SELF_DEFAULT_SOURCE,
        ),
        (
            "immunefi:feature-requests",
            "Feature requests",
            SELF_DEFAULT_SOURCE,
        ),
        (
            "immunefi:test-and-config-files",
            "Impacts on test files and configuration files unless stated otherwise in the bug bounty program",
            SELF_DEFAULT_SOURCE,
        ),
        (
            "immunefi:third-party-oracle-data",
            "Incorrect data supplied by third party oracles (this does not exclude oracle manipulation/flash loan attacks)",
            SELF_DEFAULT_SOURCE,
        ),
        (
            "immunefi:economic-governance-attacks",
            "Impacts requiring basic economic and governance attacks (e.g. 51% attack)",
            SELF_DEFAULT_SOURCE,
        ),
        (
            "immunefi:liquidity",
            "Lack of liquidity impacts",
            SELF_DEFAULT_SOURCE,
        ),
        (
            "immunefi:sybil",
            "Impacts from Sybil attacks",
            SELF_DEFAULT_SOURCE,
        ),
        (
            "immunefi:centralization",
            "Impacts involving centralization risks",
            SELF_DEFAULT_SOURCE,
        ),
    ];

    /// Every exclusion that applies to this engagement: the program's own
    /// entries, then the platform defaults it inherits. Deduped by id so a
    /// program that transcribes a default into its own list does not see it
    /// twice.
    pub fn out_of_scope_rules(&self) -> Vec<OutOfScopeRule> {
        let mut rules: Vec<OutOfScopeRule> = Vec::new();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for rule in &self.out_of_scope {
            if seen.insert(rule.id.clone()) {
                rules.push(rule.clone());
            }
        }
        if self.use_platform_defaults {
            for (id, title, source) in Self::PLATFORM_DEFAULTS {
                if seen.insert(id.to_string()) {
                    rules.push(OutOfScopeRule {
                        id: id.to_string(),
                        title: title.to_string(),
                        source: Some(source.to_string()),
                        ..Default::default()
                    });
                }
            }
        }
        rules
    }

    /// Which tier of the program's table an impact lands on, or `None` when the
    /// table does not say (no table, or an impact it does not list).
    ///
    /// The one mapping the severity side of the report needs, and it is a pure
    /// read of the table — no inference from the tier's name.
    pub fn tier_for_impact(&self, impact: ImpactClass) -> Option<&SeverityRule> {
        self.severity_table
            .iter()
            .find(|rule| rule.impacts.contains(&impact))
    }

    /// The impacts this program buys, in the order it published them (or, when
    /// it published none, every class it does not exclude). Never contains
    /// `no_value_at_risk`: no program pays for that, so it can only ever be the
    /// honest label for a direction that is not a candidate.
    ///
    /// This is the list the hypothesis planner aims at and the list the prompts
    /// state as the goal, so both read it from one place rather than each
    /// re-deriving "what does this program pay for".
    pub fn payable_impacts(&self) -> Vec<ImpactClass> {
        let mut out: Vec<ImpactClass> = match self.impacts_in_scope.is_empty() {
            true => ImpactClass::ALL
                .iter()
                .copied()
                .filter(|impact| *impact != ImpactClass::NoValueAtRisk)
                .collect(),
            false => self.impacts_in_scope.clone(),
        };
        out.retain(|impact| {
            *impact != ImpactClass::NoValueAtRisk && !self.excluded_impacts.contains(impact)
        });
        let mut seen: BTreeSet<ImpactClass> = BTreeSet::new();
        out.retain(|impact| seen.insert(*impact));
        out
    }

    /// Which of the program's assets these contracts touch, by name.
    ///
    /// The link-level counterpart of [`ScopeGate::admit`]: a link is a
    /// project semantic plus a historical finding, so before a specification
    /// exists the only thing that can be checked is whether the semantic is
    /// anchored anywhere the program insures. An empty result means the
    /// project side of the link never touches an insured asset, which is not
    /// the same as no assets being declared — callers must check
    /// [`Self::assets`] first, since an empty asset list restricts nothing.
    pub fn assets_matching<'a>(
        &self,
        contracts: impl IntoIterator<Item = &'a str>,
    ) -> Vec<String> {
        let touched: Vec<&str> = contracts.into_iter().map(contract_name).collect();
        let mut matched: Vec<String> = self
            .assets
            .iter()
            .filter(|asset| {
                touched
                    .iter()
                    .any(|candidate| same_contract(&asset.name, candidate))
            })
            .map(|asset| asset.name.clone())
            .collect();
        matched.sort();
        matched.dedup();
        matched
    }

    /// The scope as the agents read it. Both the specification generator and
    /// the review agent get the same block, so what they aim at and what they
    /// are judged against cannot drift apart.
    pub fn render_for_prompt(&self) -> String {
        let mut out = format!("## Engagement scope — {}\n\n", self.program.trim());
        out.push_str(&format!("- primacy: {}\n", self.primacy.as_str()));
        if let Some(chain) = self.chain_id {
            out.push_str(&format!("- chain: {chain}\n"));
        }
        if self.assets.is_empty() {
            out.push_str("- assets in scope: (none listed — treat the project's own deployed contracts as the surface)\n");
        } else {
            out.push_str("- assets in scope:\n");
            for asset in &self.assets {
                let mut line = format!("  - `{}`", asset.name);
                if let Some(chain) = asset.chain_id {
                    line.push_str(&format!(" (chain {chain})"));
                }
                if let Some(value) = asset.value_usd {
                    line.push_str(&format!(" — ${value} at risk"));
                }
                if let Some(address) = &asset.address {
                    line.push_str(&format!(" — `{address}`"));
                }
                out.push_str(&line);
                out.push('\n');
            }
        }
        out.push_str(&format!(
            "- impacts in scope: {}\n",
            render_impacts(&self.impacts_in_scope, "(not restricted — declare the loss honestly)")
        ));
        if !self.excluded_impacts.is_empty() {
            out.push_str(&format!(
                "- impacts explicitly excluded: {}\n",
                render_impacts(&self.excluded_impacts, "")
            ));
        }
        if !self.severity_table.is_empty() {
            out.push_str("- the program's severity table, as it publishes it:\n");
            for rule in &self.severity_table {
                let mut line = format!(
                    "  - `{}` — impacts: {}",
                    rule.tier,
                    render_impacts(&rule.impacts, "(none transcribed)")
                );
                if let Some(floor) = rule.min_value_usd {
                    line.push_str(&format!(" — from ${floor} at risk"));
                }
                match rule.report_severity {
                    Some(mapped) => line.push_str(&format!(
                        " — this pipeline reports that tier as {}\n",
                        mapped.as_str()
                    )),
                    None => line.push_str(" — report it under the program's own tier name\n"),
                }
                out.push_str(&line);
            }
            out.push_str(
                "\nA finding's tier comes from this table, not from a general rule about how much \
                 liquidity a violation drains. When two tiers could apply, name the one the \
                 program's own wording fits and say which.
",
            );
        }
        if let Some(floor) = self.min_asset_value_usd {
            out.push_str(&format!("- minimum asset value at risk: ${floor}\n"));
        }
        out.push_str(&format!(
            "- attacker model: {}\n",
            if self.requires_unprivileged_actor && !self.allow_privileged_impacts {
                "an unprivileged actor only — an exploit that needs a role the protocol already \
                 grants is not a finding"
            } else if self.allow_privileged_impacts {
                "privileged-actor impacts are payable for this program"
            } else {
                "not specified"
            }
        ));
        if !self.excluded_contracts.is_empty() {
            out.push_str(&format!(
                "- contracts excluded from any specification: {}\n",
                self.excluded_contracts.join(", ")
            ));
        }
        if !self.assumptions.is_empty() {
            out.push_str("- stated assumptions (the program does not question these):\n");
            for assumption in &self.assumptions {
                out.push_str(&format!("  - {}\n", assumption.trim()));
            }
        }
        let exclusions = self.out_of_scope_rules();
        if exclusions.is_empty() {
            out.push_str(
                "- exclusions: none declared and the platform defaults are turned off for this \
                 engagement\n",
            );
        } else {
            out.push_str(
                "- exclusions that apply — a candidate falling under any of these is not reportable \n  however real the defect is:\n",
            );
            for rule in &exclusions {
                let origin = match rule.source.as_deref() {
                    Some(source) if source == SELF_DEFAULT_SOURCE => "platform default",
                    Some(_) => "program",
                    None => "program",
                };
                out.push_str(&format!("  - [{origin}] `{}` — {}\n", rule.id, rule.title.trim()));
            }
            if self.use_platform_defaults {
                out.push_str(&format!(
                    "  (the platform defaults are inherited from {SELF_DEFAULT_SOURCE}; a program \
                     that publishes a narrower list has it transcribed above, and `\
                     use_platform_defaults: false` turns the inherited ones off)\n"
                ));
            }
        }
        if !self.known_issues.is_empty() {
            // A prior audit can carry dozens of findings; the prompt lists a
            // bounded number, the gate and the novelty judge still see all.
            const LISTED: usize = 40;
            out.push_str(
                "- already known — reported by an earlier audit or declined by the program; a \
                 rediscovery is a duplicate, not a finding:\n",
            );
            for issue in self.known_issues.iter().take(LISTED) {
                let anchor = match (issue.contracts.first(), issue.match_any.first()) {
                    (Some(contract), Some(function)) => format!(" ({contract}.{function})"),
                    (Some(contract), None) => format!(" ({contract})"),
                    _ => String::new(),
                };
                out.push_str(&format!("  - [{}] {}{anchor}\n", issue.id, issue.title.trim()));
            }
            if self.known_issues.len() > LISTED {
                out.push_str(&format!(
                    "  - … and {} more, still enforced at commit\n",
                    self.known_issues.len() - LISTED
                ));
            }
        }
        if self.triage_profile == TriageProfile::Contest {
            out.push('\n');
            out.push_str(CONTEST_SEVERITY_RUBRIC);
            out.push_str(
                "\nWherever these instructions say \"payable\", read: High or Medium under the matrix \
                 above.\n",
            );
        }
        out
    }

    /// Decide whether one specification is a candidate this program could pay
    /// for. Deterministic: it reads the specification, the scope, and (when
    /// available) the project's deployment tree — never a model.
    ///
    /// `is_project_contract` is the caller's project-scope predicate, so the
    /// rule that a specification must stand on the project's own code means
    /// the same thing here as everywhere else.
    pub(crate) fn admit(&self, spec: &AuditSpecification, ctx: &ScopeContext<'_>) -> ScopeVerdict {
        let mut notes: Vec<String> = Vec::new();
        let touched = spec.sequence_contracts();

        if let Some(rule) = self.check_attacker(spec) {
            return ScopeVerdict::excluded(rule);
        }
        if let Some(rule) = self.check_impact(spec, &mut notes) {
            return ScopeVerdict::excluded(rule);
        }
        if let Some(rule) = self.check_declared_contracts(&touched) {
            return ScopeVerdict::excluded(rule);
        }
        if let Some(rule) = self.check_project_scaffolding(&touched, ctx) {
            return ScopeVerdict::excluded(rule);
        }
        let matched = match self.check_assets(&touched, &mut notes) {
            Ok(matched) => matched,
            Err(rule) => return ScopeVerdict::excluded(rule),
        };
        if let Some(rule) = self.check_known_issues(spec) {
            return ScopeVerdict::excluded(rule);
        }
        if let Some(rule) = self.check_program_exclusions(spec) {
            return ScopeVerdict::excluded(rule);
        }
        if let Some(rule) = self.check_deployed(spec, ctx) {
            return ScopeVerdict::excluded(rule);
        }
        ScopeVerdict::Admissible { matched_assets: matched, notes }
    }

    /// The program pays for an unprivileged attacker, and this specification
    /// needs one that already holds a role.
    fn check_attacker(&self, spec: &AuditSpecification) -> Option<ScopeRule> {
        if !self.requires_unprivileged_actor || self.allow_privileged_impacts {
            return None;
        }
        match spec.attacker.as_ref().map(|a| a.privilege) {
            Some(AttackerPrivilege::Privileged) => Some(ScopeRule::PrivilegedActor {
                role: spec
                    .attacker
                    .as_ref()
                    .and_then(|a| a.role.clone())
                    .unwrap_or_else(|| "a role the protocol already grants".to_string()),
            }),
            // Undeclared is refused too: the predicate cannot check a claim the
            // specification never made, and the standing rule is unprivileged.
            None => Some(ScopeRule::AttackerNotDeclared),
            Some(_) => None,
        }
    }

    /// The declared impact is one the program named as excluded, or one its
    /// table does not list while the program decides by rules rather than by
    /// impact.
    fn check_impact(&self, spec: &AuditSpecification, notes: &mut Vec<String>) -> Option<ScopeRule> {
        let Some(impact) = spec.impact else {
            return Some(ScopeRule::ImpactNotDeclared);
        };
        if self.excluded_impacts.contains(&impact) {
            return Some(ScopeRule::ImpactExcluded { impact });
        }
        if self.impacts_in_scope.contains(&impact) {
            return None;
        }
        match self.primacy {
            Primacy::Rules if !self.impacts_in_scope.is_empty() => {
                Some(ScopeRule::ImpactNotInScope { impact })
            }
            Primacy::Impact => {
                notes.push(format!(
                    "impact `{}` is not on the program's listed impacts; primacy is impact, so \
                     the program weighs the impact itself rather than excluding it",
                    impact.as_str()
                ));
                None
            }
            _ => None,
        }
    }

    /// Mocks, test doubles and deployment scaffolding are not the system under
    /// audit, so nothing may be built on them.
    fn check_declared_contracts(&self, touched: &[String]) -> Option<ScopeRule> {
        for contract in touched {
            let name = contract_name(contract);
            if let Some(pattern) = self
                .excluded_contracts
                .iter()
                .find(|pattern| matches_pattern(pattern, name))
            {
                return Some(ScopeRule::ContractExcluded {
                    contract: name.to_string(),
                    pattern: pattern.clone(),
                });
            }
            if is_scaffold_contract_name(name) {
                let pattern = SCAFFOLD_CONTRACT_PREFIXES
                    .iter()
                    .find(|pattern| matches_pattern(pattern, name))
                    .copied()
                    .unwrap_or("Mock*");
                return Some(ScopeRule::ContractExcluded {
                    contract: name.to_string(),
                    pattern: pattern.to_string(),
                });
            }
        }
        None
    }

    /// At least one contract the sequence calls is the project's own — a
    /// specification built entirely out of its own helpers proves nothing
    /// about the system under audit.
    fn check_project_scaffolding(
        &self,
        touched: &[String],
        ctx: &ScopeContext<'_>,
    ) -> Option<ScopeRule> {
        if touched.is_empty() {
            return None;
        }
        let empty = BTreeSet::new();
        let aux = ctx
            .aux_contracts
            .unwrap_or(&empty)
            .iter()
            .map(|name| name.as_str())
            .collect::<BTreeSet<_>>();
        let any_project = touched.iter().any(|contract| {
            let name = contract_name(contract);
            !aux.contains(name) && (ctx.is_project_contract)(name)
        });
        if any_project {
            return None;
        }
        Some(ScopeRule::NoProjectContract)
    }

    /// The program insures specific assets. A specification that never
    /// touches one is aimed somewhere the program does not pay, and a matched
    /// asset below the program's value floor cannot clear its minimum.
    fn check_assets(
        &self,
        touched: &[String],
        notes: &mut Vec<String>,
    ) -> std::result::Result<Vec<String>, ScopeRule> {
        if self.assets.is_empty() {
            notes.push(
                "the scope file lists no assets; the gate can only check the standing rules"
                    .to_string(),
            );
            return Ok(Vec::new());
        }
        let mut matched: Vec<String> = Vec::new();
        for asset in &self.assets {
            let hit = touched
                .iter()
                .map(|contract| contract_name(contract))
                .any(|name| same_contract(&asset.name, name));
            if !hit {
                continue;
            }
            if let (Some(floor), Some(value)) = (self.min_asset_value_usd, asset.value_usd)
                && value < floor
            {
                return Err(ScopeRule::AssetValueBelowFloor {
                    asset: asset.name.clone(),
                    value_usd: value,
                    floor_usd: floor,
                });
            }
            matched.push(asset.name.clone());
        }
        if matched.is_empty() {
            return Err(ScopeRule::NoInScopeAssetTouched {
                contracts: touched
                    .iter()
                    .map(|contract| contract_name(contract).to_string())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect(),
            });
        }
        matched.sort();
        matched.dedup();
        Ok(matched)
    }

    /// A defect a previous report already established cannot be paid for
    /// again, however clean the reproduction is.
    fn check_known_issues(&self, spec: &AuditSpecification) -> Option<ScopeRule> {
        if self.known_issues.is_empty() {
            return None;
        }
        let text = spec.searchable_text();
        let called = spec.sequence_contracts();
        let named: BTreeSet<&str> = called
            .iter()
            .map(|contract| contract_name(contract))
            .collect();
        let distinguished: BTreeSet<&str> = spec
            .distinct_from
            .iter()
            .filter(|d| d.difference.trim().len() >= 40)
            .map(|d| d.id.as_str())
            .collect();
        let hit = self
            .known_issues
            .iter()
            .filter(|issue| issue.enforce && !distinguished.contains(issue.id.as_str()))
            .find(|issue| {
            let contract_match = issue.contracts.is_empty()
                || issue.contracts.iter().any(|known| {
                    named
                        .iter()
                        .any(|candidate| same_contract(known, candidate))
                });
            if !contract_match {
                return false;
            }
            issue.match_any.is_empty()
                || issue
                    .match_any
                    .iter()
                    .any(|needle| contains_ci(&text, needle))
        })?;
        Some(ScopeRule::KnownIssue {
            id: hit.id.clone(),
            title: hit.title.clone(),
        })
    }

    /// An exclusion the program publishes and the scope file marks `enforce`:
    /// a specification on its contracts whose text names the excluded behaviour
    /// is refused, unless the agent recorded why it is not that behaviour (the
    /// same `distinguish_from_known_issue` escape a known issue has).
    fn check_program_exclusions(&self, spec: &AuditSpecification) -> Option<ScopeRule> {
        let enforced: Vec<&OutOfScopeRule> = self
            .out_of_scope
            .iter()
            .filter(|rule| rule.enforce && !rule.match_any.is_empty())
            .collect();
        if enforced.is_empty() {
            return None;
        }
        let text = spec.searchable_text();
        let called = spec.sequence_contracts();
        let named: BTreeSet<&str> = called.iter().map(|contract| contract_name(contract)).collect();
        let distinguished: BTreeSet<&str> = spec
            .distinct_from
            .iter()
            .filter(|d| d.difference.trim().len() >= 40)
            .map(|d| d.id.as_str())
            .collect();
        let hit = enforced
            .into_iter()
            .filter(|rule| !distinguished.contains(rule.id.as_str()))
            .find(|rule| {
                let contract_match = rule.contracts.is_empty()
                    || rule
                        .contracts
                        .iter()
                        .any(|known| named.iter().any(|candidate| same_contract(known, candidate)));
                contract_match && rule.match_any.iter().any(|needle| contains_ci(&text, needle))
            })?;
        Some(ScopeRule::ProgramExclusion {
            id: hit.id.clone(),
            title: hit.title.clone(),
        })
    }

    /// The proof must be reproducible on the chain the program insures. Only
    /// checked when both the engagement's chain and the repository's
    /// deployment tree are known: absence of evidence is not evidence.
    fn check_deployed(&self, spec: &AuditSpecification, ctx: &ScopeContext<'_>) -> Option<ScopeRule> {
        let chain = self.chain_id?;
        let book = ctx.deployed?;
        if !book.knows_chain(chain) {
            return None;
        }
        let missing = book.undeployed_for_spec(chain, spec);
        if missing.is_empty() {
            return None;
        }
        Some(ScopeRule::NotDeployedOnChain {
            chain_id: chain,
            contracts: missing,
        })
    }
}

/// Contract-name prefixes that are never the system under audit. Kept here
/// rather than left to each project's scope file because they are universal:
/// every project ships test doubles, and a program pays for none of them.
const SCAFFOLD_CONTRACT_PREFIXES: [&'static str; 6] =
    ["Mock*", "Stub*", "Fake*", "Harness*", "Test*", "Dummy*"];

/// Whether a contract name is scaffolding rather than the system under audit.
///
/// Shared by the link-stage filter and the commit gate, so the earliest layer
/// and the last one exclude the same names — a link built on a mock is a
/// candidate no program buys (`immunefi:test-and-config-files`), and refusing
/// it before an agent runs costs nothing.
pub fn is_scaffold_contract_name(name: &str) -> bool {
    let name = contract_name(name);
    SCAFFOLD_CONTRACT_PREFIXES
        .iter()
        .any(|pattern| matches_pattern(pattern, name))
}

/// Everything the predicate needs that is not the scope or the specification.
///
/// `deployed` and `chain` are optional by design: a run with no deployment
/// tree, or an engagement that names no chain, skips the corresponding rule
/// instead of guessing.
pub(crate) struct ScopeContext<'a> {
    /// The caller's project-scope predicate — the same notion the harness and
    /// review layers use. `Sync` because the predicate is held across an
    /// `.await` inside the agent's tool call, which must stay `Send`.
    pub is_project_contract: &'a (dyn Fn(&str) -> bool + Sync),
    /// Contracts the specification declared as its own helpers. They are not
    /// project code and must not be counted as such.
    pub aux_contracts: Option<&'a BTreeSet<String>>,
    /// The repository's deployed addresses, when they were readable.
    pub deployed: Option<&'a crate::harness::solidity::DeploymentBook>,
}

/// The requirement a specification failed. One variant per rule, so a refusal
/// names the rule as well as the reason and the two cannot drift.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScopeRule {
    /// `attacker` was never declared, and this program requires an
    /// unprivileged one.
    AttackerNotDeclared,
    /// The exploit needs an account that already holds a role.
    PrivilegedActor { role: String },
    /// No impact class was declared.
    ImpactNotDeclared,
    /// The declared impact is on the program's exclusion list.
    ImpactExcluded { impact: ImpactClass },
    /// The program publishes its impacts and decides by rules; this one is not
    /// on the list.
    ImpactNotInScope { impact: ImpactClass },
    /// Nothing the sequence calls is one of the program's assets.
    NoInScopeAssetTouched { contracts: Vec<String> },
    /// The matched asset is worth less than the program's floor.
    AssetValueBelowFloor {
        asset: String,
        value_usd: u64,
        floor_usd: u64,
    },
    /// The specification is built on a mock / test double / scaffolding.
    ContractExcluded { contract: String, pattern: String },
    /// The sequence calls no project contract at all.
    NoProjectContract,
    /// A previous report already covers this defect.
    KnownIssue { id: String, title: String },
    /// An exclusion the program publishes, enforced by keyword.
    ProgramExclusion { id: String, title: String },
    /// The sequence calls contracts this chain does not deploy.
    NotDeployedOnChain { chain_id: u64, contracts: Vec<String> },
}

impl ScopeRule {
    /// The platform-default exclusion this rule enforces, when it enforces one.
    ///
    /// Kept here rather than in prose so a refusal can name the published line
    /// it applied: "excluded because the program does not buy privileged-actor
    /// impacts" is checkable, "our pipeline decided against it" is not. Rules
    /// that enforce the engagement's own tables (its impact list, its assets,
    /// its known issues) return `None` — those exclusions are the program's,
    /// quoted in the rule's own clause.
    pub fn out_of_scope_id(&self) -> Option<&'static str> {
        match self {
            Self::AttackerNotDeclared | Self::PrivilegedActor { .. } => {
                Some("immunefi:privileged-addresses")
            }
            // A sequence that never calls the project's own code is the same
            // exclusion by another route: only scaffolding is being exercised.
            Self::ContractExcluded { .. } | Self::NoProjectContract => {
                Some("immunefi:test-and-config-files")
            }
            _ => None,
        }
    }

    /// Short machine-readable id, for logs and the regression suite.
    pub fn id(&self) -> &'static str {
        match self {
            Self::AttackerNotDeclared => "scope-attacker-not-declared",
            Self::PrivilegedActor { .. } => "scope-privileged-actor",
            Self::ImpactNotDeclared => "scope-impact-not-declared",
            Self::ImpactExcluded { .. } => "scope-impact-excluded",
            Self::ImpactNotInScope { .. } => "scope-impact-not-in-scope",
            Self::NoInScopeAssetTouched { .. } => "scope-no-asset-touched",
            Self::AssetValueBelowFloor { .. } => "scope-asset-below-floor",
            Self::ContractExcluded { .. } => "scope-contract-excluded",
            Self::NoProjectContract => "scope-no-project-contract",
            Self::KnownIssue { .. } => "scope-known-issue",
            Self::ProgramExclusion { .. } => "scope-program-exclusion",
            Self::NotDeployedOnChain { .. } => "scope-not-deployed",
        }
    }

    /// One line naming the requirement, for the refusal the agent reads.
    pub fn clause(&self) -> String {
        match self {
            Self::AttackerNotDeclared => "no attacker model was declared, and this program pays \
                 only for impacts reachable by an unprivileged actor — declare the actor with \
                 `set_attacker_model` (and if the exploit needs a role the protocol already \
                 grants, this direction is not payable here)"
                .to_string(),
            Self::PrivilegedActor { role } => format!(
                "the exploit needs `{role}`, an authority the protocol already grants; this \
                 program does not pay for impacts that require a privileged actor"
            ),
            Self::ImpactNotDeclared => {
                "no impact class was declared, so the loss this proves cannot be matched against \
                 what the program pays for"
                    .to_string()
            }
            Self::ImpactExcluded { impact } => format!(
                "the program names `{}` as an excluded impact",
                impact.as_str()
            ),
            Self::ImpactNotInScope { impact } => format!(
                "the program publishes its impacts and decides by rules; `{}` is not on that list",
                impact.as_str()
            ),
            Self::NoInScopeAssetTouched { contracts } => format!(
                "the sequence touches none of the program's assets (it calls {})",
                contracts.join(", ")
            ),
            Self::AssetValueBelowFloor {
                asset,
                value_usd,
                floor_usd,
            } => format!(
                "`{asset}` holds ${value_usd}, below the program's ${floor_usd} floor for a \
                 payable finding"
            ),
            Self::ContractExcluded { contract, pattern } => format!(
                "`{contract}` matches the excluded pattern `{pattern}` — a mock, test double or \
                 deployment scaffold is not the system under audit"
            ),
            Self::NoProjectContract => "the sequence calls no contract of the project under \
                 audit, so nothing it proves is about this system"
                .to_string(),
            Self::KnownIssue { id, title } => format!(
                "a previous report already covers this defect ([{id}] {title}). If yours has a \
                 different root cause, call `distinguish_from_known_issue` with id `{id}` and the \
                 difference; otherwise abandon — a rediscovery is a duplicate"
            ),
            Self::ProgramExclusion { id, title } => format!(
                "the program excludes this ([{id}] {title}). If your candidate does not fall \
                 under that exclusion — for example the attacker is an outsider, not the party \
                 the program trusts — call `distinguish_from_known_issue` with id `{id}` and the \
                 difference; otherwise abandon"
            ),
            Self::NotDeployedOnChain {
                chain_id,
                contracts,
            } => format!(
                "the sequence calls {} on chain {chain_id}, which does not deploy it — the \
                 deployment on the chain this program insures cannot reach the state",
                contracts.join(", ")
            ),
        }
    }
}

/// Whether a specification is a candidate this program could pay for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScopeVerdict {
    /// Nothing the program stated excludes it. `notes` carry what the scope
    /// file left open (an unlisted impact under primacy of impact, an asset
    /// list with no value floor) so the record shows what was *not* checked.
    Admissible {
        matched_assets: Vec<String>,
        notes: Vec<String>,
    },
    Excluded { rule: ScopeRule },
}

impl ScopeVerdict {
    fn excluded(rule: ScopeRule) -> Self {
        Self::Excluded { rule }
    }

    pub fn is_admissible(&self) -> bool {
        matches!(self, Self::Admissible { .. })
    }

    /// The refusal, phrased as the requirement the specification fails. This
    /// is what the agent reads on a refused commit and what the log records.
    pub fn clause(&self) -> Option<String> {
        match self {
            Self::Admissible { .. } => None,
            Self::Excluded { rule } => {
                let cited = match rule.out_of_scope_id() {
                    Some(id) => format!(
                        " Excluded by the platform's default list (`{id}`, {SELF_DEFAULT_SOURCE})."
                    ),
                    None => String::new(),
                };
                Some(format!(
                    "Not a candidate for this program: {}.{cited} Draft unchanged.",
                    rule.clause()
                ))
            }
        }
    }

    pub fn matched_assets(&self) -> &[String] {
        match self {
            Self::Admissible { matched_assets, .. } => matched_assets,
            Self::Excluded { .. } => &[],
        }
    }

    pub fn notes(&self) -> &[String] {
        match self {
            Self::Admissible { notes, .. } => notes,
            Self::Excluded { .. } => &[],
        }
    }
}

/// The gate as one run holds it: the scope, the repository it applies to, and
/// the deployment tree loaded once on first use (the same resolve-on-first-use
/// contract the resident-source decision uses, so the predicate costs nothing
/// until a specification is actually committed).
pub struct ScopeGate {
    scope: ProgramScope,
    repo_root: PathBuf,
    deployed: tokio::sync::OnceCell<Option<Arc<crate::harness::solidity::DeploymentBook>>>,
    /// The LLM judge for exclusions no field can decide, when the run enabled
    /// it. Runs last at commit, after every deterministic gate admitted.
    judge: Option<Arc<crate::spec::scope_judge::ScopeJudge>>,
}

impl std::fmt::Debug for ScopeGate {
    /// The gate rides inside `SpecGenOptions` and every tool holding it, all of
    /// which derive `Debug`; print the program and the repository, which is
    /// what identifies it in a log line.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScopeGate")
            .field("program", &self.scope.program)
            .field("repo_root", &self.repo_root)
            .finish_non_exhaustive()
    }
}

impl ScopeGate {
    pub fn new(scope: ProgramScope, repo_root: PathBuf) -> Self {
        Self {
            scope,
            repo_root,
            deployed: tokio::sync::OnceCell::new(),
            judge: None,
        }
    }

    /// Add the LLM scope judge (see [`crate::spec::scope_judge`]).
    pub fn with_judge(mut self, judge: Arc<crate::spec::scope_judge::ScopeJudge>) -> Self {
        self.judge = Some(judge);
        self
    }

    pub fn judge(&self) -> Option<&Arc<crate::spec::scope_judge::ScopeJudge>> {
        self.judge.as_ref()
    }

    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }

    pub fn scope(&self) -> &ProgramScope {
        &self.scope
    }

    /// The repository's deployment tree, read once per run. `None` when the
    /// repository has none — the deployment rule is then skipped rather than
    /// guessed at.
    async fn deployed(&self) -> Option<&Arc<crate::harness::solidity::DeploymentBook>> {
        self.deployed
            .get_or_init(|| async {
                let book = crate::harness::solidity::DeploymentBook::load(&self.repo_root).await;
                if book.is_empty() {
                    tracing::info!(
                        "[scope] no deployment artifacts under {}; the chain/deployment rule is \
                         skipped for this run",
                        self.repo_root.display()
                    );
                    None
                } else {
                    Some(Arc::new(book))
                }
            })
            .await
            .as_ref()
    }

    /// Apply the predicate: the scope, plus this run's project contract set
    /// and deployment tree.
    pub async fn admit(
        &self,
        spec: &AuditSpecification,
        is_project_contract: &(dyn Fn(&str) -> bool + Sync),
        aux_contracts: Option<&BTreeSet<String>>,
    ) -> ScopeVerdict {
        let deployed = self.deployed().await;
        let ctx = ScopeContext {
            is_project_contract,
            aux_contracts,
            deployed: deployed.map(Arc::as_ref),
        };
        self.scope.admit(spec, &ctx)
    }
}

fn render_impacts(impacts: &[ImpactClass], fallback: &str) -> String {
    if impacts.is_empty() {
        return fallback.to_string();
    }
    impacts
        .iter()
        .map(|impact| format!("`{}`", impact.as_str()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// A sequence step's contract field may be qualified (`VaultCore.vault`); the
/// contract name is what every comparison here is about.
///
/// The leading path is dropped because a project's call graph anchors
/// contracts by file, not by identifier: a graph may store
/// `src/adapters/AaveStrategy.sol`, and comparing that against a scope that
/// lists `AaveStrategy` (the only form a program or a reviewer writes) would
/// match nothing and silently drop every link.
///
/// The path goes before the `.` split, not after it: a relative path
/// (`./src/X.sol`) or a versioned directory (`lib/v1.2/Foo.sol`) carries dots
/// of its own, and splitting on them first leaves an empty or wrong stem.
pub(crate) fn contract_name(contract: &str) -> &str {
    let head = contract
        .trim()
        .split([' ', ':'])
        .next()
        .unwrap_or(contract)
        .trim();
    let file = match head.rsplit_once('/') {
        Some((_, tail)) if !tail.is_empty() => tail,
        _ => head,
    };
    file.split('.').next().unwrap_or(file).trim()
}

/// Whether two contract names are the same contract. Deployment artifacts and
/// specifications disagree on the proxy suffix (`VaultCore` vs
/// `VaultCoreProxy`), which is the same pairing the deployment book makes.
pub(crate) fn same_contract(left: &str, right: &str) -> bool {
    left == right || stem(left) == stem(right)
}

fn stem(name: &str) -> &str {
    for suffix in ["Proxy", "Implementation", "Impl"] {
        if let Some(stripped) = name.strip_suffix(suffix)
            && !stripped.is_empty()
        {
            return stripped;
        }
    }
    name
}

/// Case-insensitive substring containment, for matching a specification's own
/// text against a program's known-issue keywords.
fn contains_ci(haystack: &str, needle: &str) -> bool {
    let needle = needle.trim();
    if needle.is_empty() {
        return false;
    }
    haystack
        .to_ascii_lowercase()
        .contains(&needle.to_ascii_lowercase())
}

/// Minimal `*`-wildcard matching, used for `excluded_contracts` patterns. Only
/// `*` is special (any run of characters) — the patterns a scope file writes
/// are name families like `Mock*` or `*Harness`, and anything richer here
/// would be a regex engine in a scope file.
pub(crate) fn matches_pattern(pattern: &str, name: &str) -> bool {
    let pattern = pattern.trim();
    if pattern.is_empty() {
        return false;
    }
    let (pattern, name) = (pattern.to_ascii_lowercase(), name.to_ascii_lowercase());
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == name;
    }
    let mut rest = name.as_str();
    for (index, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        match rest.find(part) {
            Some(at) => {
                // The first fragment must anchor at the start, the last at the
                // end; anything between only has to appear in order.
                if index == 0 && at != 0 {
                    return false;
                }
                rest = &rest[at + part.len()..];
            }
            None => return false,
        }
    }
    if let Some(last) = parts.last()
        && !last.is_empty()
        && !name.ends_with(last)
    {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCOPE_JSON: &str = r#"
{
  "program": "Immunefi: Example Finance",
  "primacy": "rules",
  "chain_id": 1,
  "assets": [
    { "name": "VaultCore", "chain_id": 1, "address": "0x1111111111111111111111111111111111111111", "value_usd": 5000000 },
    { "name": "RewardsPool", "chain_id": 1, "value_usd": 25000 }
  ],
  "impacts_in_scope": ["direct_theft_of_funds", "permanent_freeze", "protocol_insolvency"],
  "excluded_impacts": ["griefing"],
  "severity_table": [
    { "tier": "critical", "impacts": ["direct_theft_of_funds"], "min_value_usd": 1000000, "report_severity": "High" },
    { "tier": "high", "impacts": ["permanent_freeze", "protocol_insolvency"], "report_severity": "High" },
    { "tier": "medium", "impacts": ["protocol_dos", "temporary_freeze"], "report_severity": "Medium" }
  ],
  "min_asset_value_usd": 100000,
  "excluded_contracts": ["*Oracle*"],
  "assumptions": ["the chainlink feed is honest"],
  "known_issues": [
    { "id": "KI-3", "title": "rounding dust in RewardsPool.claim", "contracts": ["RewardsPool"], "match_any": ["rounding"] }
  ]
}
"#;

    fn scope() -> ProgramScope {
        ProgramScope::parse(SCOPE_JSON, Path::new("knowdit-scope.json")).expect("scope parses")
    }

    fn project(name: &str) -> bool {
        matches!(name, "VaultCore" | "RewardsPool" | "OracleAdapter" | "Strategy")
    }

    fn ctx<'a>(aux: Option<&'a BTreeSet<String>>) -> ScopeContext<'a> {
        ScopeContext {
            is_project_contract: &project,
            aux_contracts: aux,
            deployed: None,
        }
    }

    /// A specification on `VaultCore` with a declared unprivileged actor and an
    /// in-scope impact: the shape every fixture below starts from.
    fn spec_on(contract: &str, impact: ImpactClass) -> AuditSpecification {
        let mut spec = AuditSpecification::default();
        spec.impact = Some(impact);
        spec.attacker = Some(crate::types::AttackerModel {
            privilege: AttackerPrivilege::Unprivileged,
            role: None,
            note: String::new(),
        });
        spec.summary = "value leaves the vault without a matching deposit".to_string();
        spec.sequence.push(crate::types::SuqenceCallStep {
            contract: contract.to_string(),
            function: "deposit".to_string(),
            intention: String::new(),
        });
        spec
    }

    #[test]
    fn contract_name_strips_paths_before_qualifiers() {
        assert_eq!(contract_name("VaultCore.vault"), "VaultCore");
        assert_eq!(contract_name("src/adapters/AaveStrategy.sol"), "AaveStrategy");
        assert_eq!(contract_name("./src/X.sol"), "X");
        assert_eq!(contract_name("lib/v1.2/Foo.sol"), "Foo");
        assert_eq!(contract_name("src/Foo.sol:Foo"), "Foo");
        assert_eq!(contract_name("  Vault  "), "Vault");
    }

    #[test]
    fn accepts_a_specification_on_an_in_scope_asset() {
        let verdict = scope().admit(&spec_on("VaultCore", ImpactClass::DirectTheftOfFunds), &ctx(None));
        assert!(verdict.is_admissible(), "{:?}", verdict.clause());
        assert_eq!(verdict.matched_assets(), ["VaultCore".to_string()]);
        assert!(verdict.notes().is_empty());
    }

    #[test]
    fn refuses_a_privileged_actor() {
        let mut spec = spec_on("VaultCore", ImpactClass::DirectTheftOfFunds);
        spec.attacker = Some(crate::types::AttackerModel {
            privilege: AttackerPrivilege::Privileged,
            role: Some("GUARDIAN_ROLE".to_string()),
            note: String::new(),
        });
        let verdict = scope().admit(&spec, &ctx(None));
        assert!(!verdict.is_admissible());
        let clause = verdict.clause().expect("refusal clause");
        assert!(clause.contains("GUARDIAN_ROLE"), "{clause}");
    }

    #[test]
    fn refuses_an_undeclared_attacker() {
        let mut spec = spec_on("VaultCore", ImpactClass::DirectTheftOfFunds);
        spec.attacker = None;
        assert!(!scope().admit(&spec, &ctx(None)).is_admissible());
    }

    #[test]
    fn privilege_escalation_is_not_a_privileged_actor() {
        let mut spec = spec_on("VaultCore", ImpactClass::DirectTheftOfFunds);
        spec.attacker = Some(crate::types::AttackerModel {
            privilege: AttackerPrivilege::PrivilegeEscalation,
            role: None,
            note: "unprivileged caller takes a role".to_string(),
        });
        assert!(scope().admit(&spec, &ctx(None)).is_admissible());
    }

    #[test]
    fn refuses_an_impact_the_program_excludes() {
        let verdict = scope().admit(&spec_on("VaultCore", ImpactClass::Griefing), &ctx(None));
        assert!(!verdict.is_admissible());
        assert!(verdict.clause().expect("clause").contains("griefing"));
    }

    #[test]
    fn refuses_an_unlisted_impact_under_primacy_of_rules() {
        let verdict = scope().admit(
            &spec_on("VaultCore", ImpactClass::TemporaryFreeze),
            &ctx(None),
        );
        assert!(!verdict.is_admissible());
        assert!(verdict.clause().expect("clause").contains("temporary_freeze"));
    }

    #[test]
    fn an_unlisted_impact_passes_under_primacy_of_impact_with_a_note() {
        let mut scope = scope();
        scope.primacy = Primacy::Impact;
        let verdict = scope.admit(
            &spec_on("VaultCore", ImpactClass::TemporaryFreeze),
            &ctx(None),
        );
        assert!(verdict.is_admissible(), "{:?}", verdict.clause());
        assert!(
            verdict.notes().iter().any(|n| n.contains("primacy is impact")),
            "{:?}",
            verdict.notes()
        );
    }

    #[test]
    fn refuses_a_specification_that_touches_no_asset() {
        let verdict = scope().admit(&spec_on("Strategy", ImpactClass::DirectTheftOfFunds), &ctx(None));
        assert!(!verdict.is_admissible());
        let clause = verdict.clause().expect("clause");
        assert!(clause.contains("Strategy"), "{clause}");
    }

    #[test]
    fn refuses_an_asset_below_the_value_floor() {
        let verdict = scope().admit(&spec_on("RewardsPool", ImpactClass::DirectTheftOfFunds), &ctx(None));
        assert!(!verdict.is_admissible());
        let clause = verdict.clause().expect("clause");
        assert!(clause.contains("25000") && clause.contains("100000"), "{clause}");
    }

    #[test]
    fn refuses_a_mock_contract_even_when_the_scope_file_is_silent() {
        let verdict = scope().admit(
            &spec_on("MockVault", ImpactClass::DirectTheftOfFunds),
            &ctx(None),
        );
        assert!(!verdict.is_admissible());
        assert!(verdict.clause().expect("clause").contains("MockVault"));
    }

    #[test]
    fn refuses_a_contract_the_program_names_as_excluded() {
        let verdict = scope().admit(
            &spec_on("OracleAdapter", ImpactClass::DirectTheftOfFunds),
            &ctx(None),
        );
        assert!(!verdict.is_admissible());
        assert!(verdict.clause().expect("clause").contains("*Oracle*"));
    }

    #[test]
    fn refuses_a_specification_whose_sequence_is_all_auxiliary() {
        let mut spec = spec_on("HarnessCallback", ImpactClass::DirectTheftOfFunds);
        spec.sequence[0].contract = "HarnessCallback".to_string();
        let aux: BTreeSet<String> = ["HarnessCallback".to_string()].into_iter().collect();
        let verdict = scope().admit(&spec, &ctx(Some(&aux)));
        assert!(!verdict.is_admissible());
    }

    #[test]
    fn refuses_a_known_issue_by_contract_and_keyword() {
        let mut spec = spec_on("RewardsPool", ImpactClass::DirectTheftOfFunds);
        spec.summary = "rounding in claim() lets the last claimant take dust".to_string();
        let mut scope = scope();
        // Keep the value floor out of the way: this fixture is about the
        // known-issue rule.
        scope.min_asset_value_usd = None;
        let verdict = scope.admit(&spec, &ctx(None));
        assert!(!verdict.is_admissible());
        let clause = verdict.clause().expect("clause");
        assert!(clause.contains("KI-3"), "{clause}");
    }

    #[test]
    fn a_distinguished_or_context_only_known_issue_does_not_refuse() {
        let mut spec = spec_on("RewardsPool", ImpactClass::DirectTheftOfFunds);
        spec.summary = "rounding in claim() lets the last claimant take dust".to_string();
        let mut scope = scope();
        scope.min_asset_value_usd = None;

        // Context-only: shown to agents, never used to refuse.
        let mut context_only = scope.clone();
        context_only.known_issues.iter_mut().for_each(|issue| issue.enforce = false);
        assert!(context_only.admit(&spec, &ctx(None)).is_admissible());

        // A distinction too short to be an argument is ignored.
        spec.distinct_from = vec![crate::types::KnownIssueDistinction {
            id: "KI-3".to_string(),
            difference: "different".to_string(),
        }];
        assert!(!scope.admit(&spec, &ctx(None)).is_admissible());

        // A stated different root cause admits it for the reviewer to judge.
        spec.distinct_from[0].difference =
            "the known issue is rounding on the last claim; this is an index that is never reset"
                .to_string();
        assert!(scope.admit(&spec, &ctx(None)).is_admissible());
    }

    #[test]
    fn an_enforced_program_exclusion_refuses_until_distinguished() {
        let mut spec = spec_on("RewardsPool", ImpactClass::DirectTheftOfFunds);
        spec.summary = "a revoked delegate keeps spending the owner's rewards".to_string();
        let mut scope = scope();
        scope.min_asset_value_usd = None;
        scope.known_issues.clear();
        scope.out_of_scope = vec![OutOfScopeRule {
            id: "OOS-delegates".to_string(),
            title: "delegates are trusted by design".to_string(),
            contracts: vec!["RewardsPool".to_string()],
            match_any: vec!["delegate".to_string()],
            enforce: true,
            ..Default::default()
        }];
        let verdict = scope.admit(&spec, &ctx(None));
        assert!(!verdict.is_admissible());
        assert!(verdict.clause().expect("clause").contains("OOS-delegates"));

        // Context-only (the default): rendered, never used to refuse.
        let mut context_only = scope.clone();
        context_only.out_of_scope[0].enforce = false;
        assert!(context_only.admit(&spec, &ctx(None)).is_admissible());

        // An outsider attack on the same code is argued out of the exclusion.
        spec.distinct_from = vec![crate::types::KnownIssueDistinction {
            id: "OOS-delegates".to_string(),
            difference: "the attacker is never a delegate: any outsider front-runs the owner's call"
                .to_string(),
        }];
        assert!(scope.admit(&spec, &ctx(None)).is_admissible());
    }

    #[test]
    fn a_different_defect_on_a_known_contract_is_not_the_known_issue() {
        let mut spec = spec_on("RewardsPool", ImpactClass::DirectTheftOfFunds);
        spec.summary = "claim() pays out twice for one share".to_string();
        let mut scope = scope();
        scope.min_asset_value_usd = None;
        assert!(scope.admit(&spec, &ctx(None)).is_admissible());
    }

    #[test]
    fn refuses_a_specification_the_engagement_chain_does_not_deploy() {
        let scope = scope();
        // A book whose chain 1 deploys RewardsPool but not VaultCore.
        let book = book_with(&[("RewardsPool", "0x2222222222222222222222222222222222222222")]);
        let ctx = ScopeContext {
            is_project_contract: &project,
            aux_contracts: None,
            deployed: Some(&book),
        };
        let verdict = scope.admit(&spec_on("VaultCore", ImpactClass::DirectTheftOfFunds), &ctx);
        assert!(!verdict.is_admissible());
        let clause = verdict.clause().expect("clause");
        assert!(clause.contains("VaultCore") && clause.contains("chain 1"), "{clause}");
    }

    /// A one-chain deployment book, built through the real artifact layout so
    /// the fixture cannot drift from the loader.
    fn book_with(entries: &[(&str, &str)]) -> crate::harness::solidity::DeploymentBook {
        let root = std::env::temp_dir().join(format!(
            "knowdit-scope-book-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let network = root.join("deployments/mainnet");
        std::fs::create_dir_all(&network).expect("mkdir");
        std::fs::write(network.join(".chainId"), "1\n").expect("write chain id");
        for (name, address) in entries {
            std::fs::write(network.join(format!("{name}.json")), format!(r#"{{"address":"{address}"}}"#))
                .expect("write artifact");
        }
        let book = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime")
            .block_on(crate::harness::solidity::DeploymentBook::load(&root));
        std::fs::remove_dir_all(root).ok();
        book
    }

    #[test]
    fn renders_every_rule_the_gate_enforces() {
        let block = scope().render_for_prompt();
        assert!(block.contains("Immunefi: Example Finance"), "{block}");
        assert!(block.contains("$5000000"), "{block}");
        assert!(block.contains("unprivileged actor only"), "{block}");
        assert!(block.contains("KI-3"), "{block}");
        assert!(block.contains("rounding dust"), "{block}");
        assert!(block.contains("the chainlink feed is honest"), "{block}");
    }

    #[test]
    fn pattern_matching_is_family_based() {
        assert!(matches_pattern("Mock*", "MockVault"));
        assert!(matches_pattern("*Harness", "VaultHarness"));
        assert!(matches_pattern("*Oracle*", "PriceOracleAdapter"));
        assert!(!matches_pattern("Mock*", "VaultMock"));
        assert!(!matches_pattern("Vault", "VaultCore"));
    }

    #[test]
    fn exclusions_merge_the_programs_list_with_the_platform_defaults() {
        let mut scope = scope();
        scope.out_of_scope = vec![
            OutOfScopeRule {
                id: "OOS-1".to_string(),
                title: "Issues already documented in a published audit or the known-issues section"
                    .to_string(),
                source: Some("https://immunefi.com/bug-bounty/example/scope/".to_string()),
                ..Default::default()
            },
            // A program that transcribes a default into its own list must not
            // see it twice.
            OutOfScopeRule {
                id: "immunefi:privileged-addresses".to_string(),
                title: "program's own wording of the same exclusion".to_string(),
                source: None,
                ..Default::default()
            },
        ];
        let rules = scope.out_of_scope_rules();
        assert_eq!(
            rules
                .iter()
                .filter(|rule| rule.id == "immunefi:privileged-addresses")
                .count(),
            1,
            "deduped by id: {:?}",
            rules.iter().map(|r| &r.id).collect::<Vec<_>>()
        );
        assert!(
            rules
                .iter()
                .find(|rule| rule.id == "immunefi:privileged-addresses")
                .is_some_and(|rule| rule.title.contains("program's own wording")),
            "the program's own wording wins over the platform's"
        );
        assert!(rules.iter().any(|rule| rule.id == "OOS-1"));
        assert!(
            rules.iter().any(|rule| rule.id == "immunefi:test-and-config-files"),
            "the platform defaults are inherited: {:?}",
            rules.iter().map(|r| &r.id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn the_platform_defaults_can_be_turned_off() {
        let mut scope = scope();
        scope.use_platform_defaults = false;
        assert!(scope.out_of_scope_rules().is_empty());
        let block = scope.render_for_prompt();
        assert!(block.contains("platform defaults are turned off"), "{block}");
    }

    #[test]
    fn the_rendered_scope_carries_the_exclusions_where_they_are_read() {
        let mut scope = scope();
        scope.out_of_scope = vec![OutOfScopeRule {
            id: "OOS-1".to_string(),
            title: "Issues already documented in a published audit".to_string(),
            source: Some("https://immunefi.com/bug-bounty/example/scope/".to_string()),
            ..Default::default()
        }];
        let block = scope.render_for_prompt();
        assert!(block.contains("[program] `OOS-1`"), "{block}");
        assert!(block.contains("Issues already documented"), "{block}");
        assert!(block.contains("[platform default] `immunefi:privileged-addresses`"), "{block}");
        assert!(
            block.contains(SELF_DEFAULT_SOURCE),
            "the exclusions carry their source: {block}"
        );
        assert!(
            block.contains("not reportable"),
            "the block must say what an exclusion means for a candidate: {block}"
        );
    }

    #[test]
    fn a_refusal_cites_the_published_exclusion_it_applied() {
        // Privileged actor: the platform default says these are not bought.
        let mut spec = spec_on("VaultCore", ImpactClass::DirectTheftOfFunds);
        spec.attacker = Some(crate::types::AttackerModel {
            privilege: AttackerPrivilege::Privileged,
            role: Some("GUARDIAN_ROLE".to_string()),
            note: String::new(),
        });
        let clause = scope().admit(&spec, &ctx(None)).clause().expect("refused");
        assert!(clause.contains("GUARDIAN_ROLE"), "{clause}");
        assert!(
            clause.contains("immunefi:privileged-addresses"),
            "the refusal must name the exclusion: {clause}"
        );
        assert!(clause.contains(SELF_DEFAULT_SOURCE), "{clause}");

        // Test/config code is the other mapped default.
        let mock = spec_on("MockVault", ImpactClass::DirectTheftOfFunds);
        let clause = scope().admit(&mock, &ctx(None)).clause().expect("refused");
        assert!(
            clause.contains("immunefi:test-and-config-files"),
            "{clause}"
        );

        // The engagement's own tables are not platform lines: an asset outside
        // the program's list cites the rule, not a default.
        let outside = spec_on("Strategy", ImpactClass::DirectTheftOfFunds);
        let clause = scope().admit(&outside, &ctx(None)).clause().expect("refused");
        assert!(clause.contains("Strategy"), "{clause}");
        assert!(
            !clause.contains("platform's default list"),
            "no platform line covers an asset the program simply does not insure: {clause}"
        );
    }

    #[test]
    fn the_severity_table_maps_impacts_to_the_programs_own_tiers() {
        let scope = scope();
        let critical = scope
            .tier_for_impact(ImpactClass::DirectTheftOfFunds)
            .expect("declared impact");
        assert_eq!(critical.tier, "critical");
        assert_eq!(critical.min_value_usd, Some(1_000_000));
        assert_eq!(critical.report_severity, Some(ReviewSeverity::High));

        let medium = scope
            .tier_for_impact(ImpactClass::ProtocolDos)
            .expect("declared impact");
        assert_eq!(medium.tier, "medium");
        assert_eq!(medium.report_severity, Some(ReviewSeverity::Medium));

        // An impact the program's table does not list maps to nothing: the
        // pipeline then has no ceiling to apply and says so by returning None.
        assert!(scope.tier_for_impact(ImpactClass::Griefing).is_none());

        let block = scope.render_for_prompt();
        assert!(block.contains("`critical`"), "{block}");
        assert!(block.contains("from $1000000 at risk"), "{block}");
        assert!(
            block.contains("reports that tier as High"),
            "the transcribed report mapping belongs in the prompt: {block}"
        );
        assert!(
            block.contains("not from a general rule"),
            "the prompt must say the table replaces the default tiering: {block}"
        );
    }

    #[test]
    fn a_scope_without_a_table_maps_nothing() {
        let mut scope = scope();
        scope.severity_table.clear();
        assert!(scope.tier_for_impact(ImpactClass::DirectTheftOfFunds).is_none());
        assert!(!scope.render_for_prompt().contains("severity table"));
    }

    #[test]
    fn a_rejected_finding_round_trips_into_a_scope_the_gate_enforces() {
        // The whole point of folding a program's answer into the scope: it
        // survives on disk and refuses the same defect next run, without the
        // claim ever being re-derived.
        let root = std::env::temp_dir().join(format!(
            "knowdit-scope-rejection-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&root).expect("create scratch dir");
        let path = root.join("knowdit-scope.json");

        let mut scope = scope();
        scope.min_asset_value_usd = None;
        let replaced = scope.add_known_issue(KnownIssue {
            id: "REJ-7".to_string(),
            title: "Immunefi: Fixture declined this: rounding dust is not a payable impact"
                .to_string(),
            contracts: vec!["RewardsPool".to_string()],
            match_any: vec!["claim".to_string()],
            enforce: true,
            summary: String::new(),
        });
        assert!(!replaced, "a fresh id is added, not replaced");
        scope.write(&path).expect("scope writes");

        let reloaded = ProgramScope::load(&path).expect("scope reloads");
        // The fixture already carries one known issue (KI-3); the rejection is
        // added beside it, not in place of it.
        assert!(
            reloaded.known_issues.iter().any(|issue| issue.id == "REJ-7"),
            "{:?}",
            reloaded.known_issues
        );
        assert!(reloaded.render_for_prompt().contains("REJ-7"));

        // The defect the program declined is now refused, deterministically.
        let mut declined = spec_on("RewardsPool", ImpactClass::DirectTheftOfFunds);
        declined.summary = "claim() pays twice for one share".to_string();
        let verdict = reloaded.admit(&declined, &ctx(None));
        assert!(!verdict.is_admissible());
        assert!(verdict.clause().expect("clause").contains("REJ-7"));

        // A different defect on the same contract is still a candidate.
        let mut other = spec_on("RewardsPool", ImpactClass::DirectTheftOfFunds);
        other.summary = "deposit() credits the wrong share".to_string();
        assert!(reloaded.admit(&other, &ctx(None)).is_admissible());

        // Re-recording the same id corrects the entry instead of duplicating it.
        let mut scope = reloaded;
        let replaced = scope.add_known_issue(KnownIssue {
            id: "REJ-7".to_string(),
            title: "Immunefi: Fixture declined this: duplicate of a prior report".to_string(),
            contracts: vec!["RewardsPool".to_string()],
            match_any: vec!["claim".to_string()],
            enforce: true,
            summary: String::new(),
        });
        assert!(replaced);
        assert_eq!(
            scope
                .known_issues
                .iter()
                .filter(|issue| issue.id == "REJ-7")
                .count(),
            1,
            "re-recording corrects the entry rather than duplicating it"
        );
        assert!(
            scope
                .known_issues
                .iter()
                .find(|issue| issue.id == "REJ-7")
                .is_some_and(|issue| issue.title.contains("duplicate"))
        );

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn toml_scope_files_parse_too() {
        let scope = ProgramScope::parse(
            "program = \"Immunefi: TOML Program\"\nimpacts_in_scope = [\"direct_theft_of_funds\"]\n",
            Path::new("knowdit-scope.toml"),
        )
        .expect("toml scope parses");
        assert_eq!(scope.program, "Immunefi: TOML Program");
        assert!(scope.requires_unprivileged_actor, "the standing rule defaults on");
    }
}
