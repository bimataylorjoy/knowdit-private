//! Prior audit reports as ground the pipeline starts from.
//!
//! A bounty program does not pay for what an earlier audit already reported:
//! the finding is a duplicate the moment it is submitted, however cleanly it
//! reproduces. Those reports exist before the first link runs, so the cheapest
//! place to use them is before the first link runs — as known issues the commit
//! gate enforces, and as an inventory the novelty judge defers against.
//!
//! This module reads a report (Code4rena, Sherlock, Spearbit, Trail of Bits and
//! plain markdown shapes; PDF through `pdftotext`) and turns each finding into a
//! [`KnownIssue`]. It never calls a model. The precision question — *which*
//! later specifications count as the same defect — is answered against the
//! project's own vocabulary: an issue is enforced by the gate only when it can be
//! pinned to a project contract **and** a specific project function. One that
//! names only a contract, or only generic entry points (`withdraw`, `deposit`),
//! is kept as `enforce: false`: shown to every agent and to the novelty judge,
//! never used to refuse a commit, because a gate keyed on "any spec touching
//! `Vault`" would refuse every new finding on the contract that matters most.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use color_eyre::eyre::{Result, WrapErr, eyre};
use knowdit_repo_model::cg::CallGraph;
use regex::Regex;

use crate::scope::KnownIssue;

/// Function names too generic to identify one defect: a known issue keyed on
/// them would refuse every later specification that calls the same standard
/// entry point.
const GENERIC_FUNCTIONS: [&str; 24] = [
    "transfer",
    "transferfrom",
    "approve",
    "allowance",
    "balanceof",
    "totalsupply",
    "deposit",
    "withdraw",
    "redeem",
    "mint",
    "burn",
    "claim",
    "stake",
    "unstake",
    "swap",
    "initialize",
    "constructor",
    "receive",
    "fallback",
    "name",
    "symbol",
    "decimals",
    "owner",
    "update",
];

/// The project's names, which is what a finding's identifiers are matched
/// against.
#[derive(Debug, Clone, Default)]
pub struct ProjectVocabulary {
    /// contract name -> its function names.
    functions_by_contract: BTreeMap<String, BTreeSet<String>>,
}

impl ProjectVocabulary {
    pub fn from_call_graph(call_graph: &CallGraph) -> Self {
        let mut functions_by_contract: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for contract in call_graph.contracts.values() {
            // A library the project vendors (OpenZeppelin's ERC20, Clones) is
            // named in every report and owns none of its findings.
            let path = contract.relative_file_path.display().to_string();
            if knowdit_repo_model::project_facts::is_vendored_path(&path) {
                continue;
            }
            let entry = functions_by_contract.entry(contract.name.clone()).or_default();
            entry.extend(contract.functions.iter().map(|f| f.name.clone()));
        }
        Self {
            functions_by_contract,
        }
    }

    /// Build directly from `(contract, functions)` pairs — for tests and for
    /// callers without a project database.
    pub fn from_pairs(pairs: &[(&str, &[&str])]) -> Self {
        Self {
            functions_by_contract: pairs
                .iter()
                .map(|(contract, functions)| {
                    (
                        contract.to_string(),
                        functions.iter().map(|f| f.to_string()).collect(),
                    )
                })
                .collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.functions_by_contract.is_empty()
    }
}

/// One finding read out of a report, before it becomes a [`KnownIssue`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PriorFinding {
    /// The report's own id (`H-01`, `5.1.3`), or a positional one.
    pub id: String,
    pub severity: Option<String>,
    pub title: String,
    pub body: String,
}

/// What an import produced.
#[derive(Debug, Clone, Default)]
pub struct ImportedIssues {
    pub issues: Vec<KnownIssue>,
    /// Findings skipped by the severity filter (gas notes by default).
    pub skipped: usize,
}

/// Reads findings out of report text and anchors them in the project.
pub struct PriorReportParser {
    vocabulary: ProjectVocabulary,
    /// A line that opens a finding: optional heading marks / bold, then an id.
    id_line: Regex,
    /// A line naming a severity section (`## High Risk`, `Medium Severity`).
    severity_section: Regex,
    /// A markdown heading.
    heading: Regex,
    /// A finding opened by a file-code id and a pipe (Guardian).
    code_id_line: Regex,
    /// A finding opened by a dash-less severity id (Dedaub).
    bare_id_line: Regex,
    /// A summary-table row opening with a code id (Guardian).
    table_row_id: Regex,
    /// A `● Severity` cell in a summary table.
    bullet_severity: Regex,
    /// Identifier-looking tokens in prose.
    identifier: Regex,
}

impl PriorReportParser {
    pub fn new(vocabulary: ProjectVocabulary) -> Self {
        Self {
            vocabulary,
            id_line: Regex::new(
                r"(?i)^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*(?:(?:issue|finding)\s+)?\[?((?:[CHMLGIN]|QA|TOB-[A-Z0-9]+|GAS|INFO)-\d{1,3}(?:\.\d+)?|\d{1,2}\.\d{1,2}\.\d{1,3})\]?\s*[:.\-–—)]?\s*(?:\*\*)?\s*(.{6,})$",
            )
            .expect("static id-line pattern compiles"),
            // Guardian: `GSU-1 | Incorrect Decrease Gas Estimation` — a file code
            // plus number, then a pipe. The pipe is required so the summary
            // table rows (`GSU-1   Title   Category   ● Medium`) do not open a
            // finding of their own.
            code_id_line: Regex::new(r"^\s*([A-Z]{2,8}-\d{1,3})\s*\|\s*(.{6,})$")
                .expect("static code-id pattern compiles"),
            // Dedaub: `C1     Reentrancy vulnerability in cancelOrder     OPEN`.
            bare_id_line: Regex::new(r"^\s*([CHMLA]\d{1,2})\s{2,}(\S.{5,}?)(?:\s{2,}(?:OPEN|RESOLVED|ACKNOWLEDGED|DISMISSED|FIXED|PARTIALLY\s+RESOLVED))?\s*$")
                .expect("static bare-id pattern compiles"),
            // Guardian summary table row carrying the severity for a code id:
            // `GLOBAL-1   Positive Impact Misrepresented   Logical Error   ● Medium   Acknowledged`.
            table_row_id: Regex::new(r"^\s*([A-Z]{2,8}-\d{1,3})(?:\s|$)")
                .expect("static table-row pattern compiles"),
            bullet_severity: Regex::new(r"(?i)●\s*(critical|high|medium|low|info(?:rmational)?)\b")
                .expect("static bullet-severity pattern compiles"),
            severity_section: Regex::new(
                r"(?i)^\s*#{0,6}\s*(critical|high|medium|low|informational|gas)(?:\s+(?:risk|severity|issues|findings))*\b",
            )
            .expect("static severity pattern compiles"),
            heading: Regex::new(r"^\s*(#{2,6})\s+(.{6,})$").expect("static heading pattern compiles"),
            identifier: Regex::new(r"[A-Za-z_][A-Za-z0-9_]{2,}").expect("static identifier pattern compiles"),
        }
    }

    /// Read `text` (a whole report) into findings, one per id. A report
    /// usually lists each finding twice — in its table of contents and as a
    /// section — so the occurrence with the longest body wins.
    pub fn findings(&self, text: &str) -> Vec<PriorFinding> {
        let lines: Vec<&str> = text.lines().collect();
        // (line index, id, severity, title)
        let mut starts: Vec<(usize, String, Option<String>, String)> = Vec::new();
        let mut section: Option<String> = None;
        let mut positional = 0usize;
        // Severities a summary table gives for file-code ids, which carry no
        // severity letter of their own.
        // A row whose title wraps carries its `● Severity` on a continuation
        // line, so a severity seen before the next table id belongs to the
        // last id that opened a row.
        let mut table: BTreeMap<String, String> = BTreeMap::new();
        let mut pending: Option<String> = None;
        for line in &lines {
            if self.code_id_line.is_match(line) {
                pending = None;
                continue;
            }
            if let Some(c) = self.table_row_id.captures(line) {
                pending = Some(c[1].to_uppercase());
            }
            if let (Some(id), Some(c)) = (pending.as_ref(), self.bullet_severity.captures(line)) {
                let sev = c[1].to_lowercase();
                let sev = if sev.starts_with("info") { "informational".to_string() } else { sev };
                table.entry(id.clone()).or_insert(sev);
                pending = None;
            }
        }
        for (index, line) in lines.iter().enumerate() {
            if let Some(captures) = self.code_id_line.captures(line) {
                let id = captures[1].to_uppercase();
                let title = captures[2].trim().to_string();
                let severity = table.get(&id).cloned().or_else(|| section.clone());
                starts.push((index, id, severity, title));
                continue;
            }
            if let Some(captures) = self.bare_id_line.captures(line) {
                let id = captures[1].to_uppercase();
                let title = captures[2].trim().to_string();
                let severity = Self::severity_of_id(&id).or_else(|| section.clone());
                starts.push((index, id, severity, title));
                continue;
            }
            if let Some(captures) = self.id_line.captures(line) {
                let id = captures[1].to_uppercase();
                let title = captures[2].trim().trim_end_matches("**").trim().to_string();
                let severity = Self::severity_of_id(&id).or_else(|| section.clone());
                starts.push((index, id, severity, title));
                continue;
            }
            // A severity section is a heading near the margin. A deeply
            // indented `Gas` is the wrapped tail of a table's category column
            // ("Gas Attack ● High"), and reading it as a section dropped real
            // High/Medium findings as gas notes.
            if let Some(captures) = self.severity_section.captures(line)
                && line.trim().len() < 60
                && line.len() - line.trim_start().len() <= 8
            {
                section = Some(captures[1].to_lowercase());
                continue;
            }
            // Spearbit-style: a plain heading under a severity section.
            if let (Some(current), Some(captures)) = (&section, self.heading.captures(line)) {
                positional += 1;
                starts.push((
                    index,
                    format!("{}-{positional}", current.chars().next().unwrap_or('x').to_ascii_uppercase()),
                    Some(current.clone()),
                    captures[2].trim().to_string(),
                ));
            }
        }
        let mut by_id: BTreeMap<String, PriorFinding> = BTreeMap::new();
        let mut order: Vec<String> = Vec::new();
        for (position, (start, id, severity, title)) in starts.iter().enumerate() {
            let end = starts
                .get(position + 1)
                .map(|(next, ..)| *next)
                .unwrap_or(lines.len());
            let body: String = lines[start + 1..end].join("\n").chars().take(8_000).collect();
            let finding = PriorFinding {
                id: id.clone(),
                severity: severity.clone(),
                title: title.clone(),
                body,
            };
            match by_id.get(id) {
                Some(existing) if existing.body.len() >= finding.body.len() => {}
                Some(_) => {
                    by_id.insert(id.clone(), finding);
                }
                None => {
                    order.push(id.clone());
                    by_id.insert(id.clone(), finding);
                }
            }
        }
        order.into_iter().filter_map(|id| by_id.remove(&id)).collect()
    }

    fn severity_of_id(id: &str) -> Option<String> {
        let upper = id.to_uppercase();
        // Only a single severity letter (or a known prefix) before the number
        // names a severity. A file-code id such as Guardian's `GLOBAL-1` or
        // `MKTU-2` starts with a letter by accident; reading it as `G` = gas
        // would silently drop it under the default `--skip-severity gas`.
        let prefix: String = upper.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
        let known = matches!(prefix.as_str(), "C" | "H" | "M" | "L" | "G" | "I" | "N" | "A" | "QA" | "GAS" | "INFO")
            || upper.starts_with("TOB-");
        if !known {
            return None;
        }
        if prefix == "A" {
            return Some("informational".to_string());
        }
        let severity = if upper.starts_with("QA") || upper.starts_with('N') {
            "low"
        } else if upper.starts_with("GAS") || upper.starts_with('G') {
            "gas"
        } else if upper.starts_with("INFO") || upper.starts_with('I') {
            "informational"
        } else {
            match upper.chars().next()? {
                'C' => "critical",
                'H' => "high",
                'M' => "medium",
                'L' => "low",
                _ => return None,
            }
        };
        Some(severity.to_string())
    }

    /// Anchor one finding in the project: the contracts it names (the title
    /// first, then the body by frequency) and the specific functions of those
    /// contracts its title names.
    fn anchor(&self, finding: &PriorFinding) -> (Vec<String>, Vec<String>) {
        let count = |text: &str| -> BTreeMap<String, usize> {
            let mut counts = BTreeMap::new();
            for token in self.identifier.find_iter(text) {
                *counts.entry(token.as_str().to_string()).or_insert(0usize) += 1;
            }
            counts
        };
        let title_tokens = count(&finding.title);
        let body_tokens = count(&finding.body);
        let mut contracts: Vec<(usize, String)> = self
            .vocabulary
            .functions_by_contract
            .keys()
            .filter_map(|contract| {
                let in_title = title_tokens.get(contract).copied().unwrap_or(0);
                let in_body = body_tokens.get(contract).copied().unwrap_or(0);
                (in_title + in_body > 0).then(|| (in_title * 100 + in_body, contract.clone()))
            })
            .collect();
        contracts.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let contracts: Vec<String> = contracts.into_iter().take(3).map(|(_, c)| c).collect();

        // Functions come from the title only. A body walks through half the
        // protocol to explain one defect, and keying the issue on every
        // function it mentions made a single report refuse every later
        // specification on the protocol's central entry point.
        let mut functions: Vec<(usize, String)> = Vec::new();
        for contract in &contracts {
            for function in &self.vocabulary.functions_by_contract[contract] {
                if GENERIC_FUNCTIONS.contains(&function.to_lowercase().as_str()) || function.len() < 4 {
                    continue;
                }
                let in_title = title_tokens.get(function).copied().unwrap_or(0);
                if in_title > 0 && !functions.iter().any(|(_, f)| f == function) {
                    functions.push((in_title, function.clone()));
                }
            }
        }
        functions.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let functions: Vec<String> = functions.into_iter().take(4).map(|(_, f)| f).collect();
        // Once the issue is pinned to functions, the contracts that matter are
        // the ones declaring them; a helper library the body happened to
        // mention would only widen what the gate refuses.
        let contracts = if functions.is_empty() {
            contracts
        } else {
            contracts
                .into_iter()
                .filter(|contract| {
                    functions
                        .iter()
                        .any(|f| self.vocabulary.functions_by_contract[contract].contains(f))
                })
                .collect()
        };
        (contracts, functions)
    }

    /// The first few hundred characters of a finding that describe the
    /// defect: report boilerplate (who submitted it, status, links to code) is
    /// skipped so the novelty judge compares mechanisms, not credits.
    fn summary(body: &str) -> String {
        let boilerplate = Regex::new(
            r"(?i)(submitted by|also found by|^\s*(severity|status|difficulty|type|target|context|lines? of code|links? to affected code|resolution|finding description and impact)\b)",
        )
        .expect("static boilerplate pattern compiles");
        body.lines()
            .filter(|line| !line.trim().is_empty() && !boilerplate.is_match(line))
            .collect::<Vec<_>>()
            .join(" ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .take(400)
            .collect()
    }

    /// Turn a report into known issues labelled `<label>:<id>`. Findings whose
    /// severity is in `skip_severities` are counted and dropped.
    pub fn import(&self, label: &str, text: &str, skip_severities: &[String]) -> ImportedIssues {
        let mut out = ImportedIssues::default();
        for finding in self.findings(text) {
            if finding
                .severity
                .as_deref()
                .is_some_and(|severity| skip_severities.iter().any(|s| s.eq_ignore_ascii_case(severity)))
            {
                out.skipped += 1;
                continue;
            }
            out.issues.push(self.to_issue(label, &finding));
        }
        out
    }

    /// Import a whole file as one issue, for a free-form note that holds a
    /// single finding rather than a report's list — an operator's kill-ledger
    /// entry, a one-issue write-up. The id is the caller's (the file stem);
    /// the title is the first markdown heading that carries that id (skipping a
    /// stray banner heading), else the first heading, else the id; the body is
    /// the whole file, so `anchor` and `summary` see everything the note names.
    /// The multi-finding [`Self::findings`] parser never fires here, so a note
    /// with no report-style id lines is no longer silently read as zero
    /// findings.
    pub fn import_note(&self, label: &str, id: &str, text: &str) -> ImportedIssues {
        let finding = PriorFinding {
            id: id.to_string(),
            severity: None,
            title: Self::note_title(text, id),
            body: text.chars().take(8_000).collect(),
        };
        ImportedIssues {
            issues: vec![self.to_issue(label, &finding)],
            skipped: 0,
        }
    }

    /// Build the [`KnownIssue`] for one parsed finding: anchor it, and enforce
    /// at the commit gate only when it pins both a contract and a function.
    fn to_issue(&self, label: &str, finding: &PriorFinding) -> KnownIssue {
        let (contracts, functions) = self.anchor(finding);
        let enforce = !contracts.is_empty() && !functions.is_empty();
        KnownIssue {
            id: format!("{label}:{}", finding.id),
            title: match &finding.severity {
                Some(severity) => format!("[{label} {severity}] {}", finding.title),
                None => format!("[{label}] {}", finding.title),
            },
            contracts,
            match_any: functions,
            enforce,
            summary: Self::summary(&finding.body),
        }
    }

    /// The title for a one-file note: the first markdown heading whose text
    /// holds `<letters>-...-<digits>` (the ledger's own id, skipping a leading
    /// banner heading that carries none), else the first heading, else the id.
    fn note_title(text: &str, id: &str) -> String {
        let id_token = Regex::new(r"[A-Za-z]+-(?:[A-Za-z]+-)*\d+").expect("static id token compiles");
        let headings: Vec<String> = text
            .lines()
            .filter_map(|line| {
                let trimmed = line.trim_start();
                trimmed
                    .strip_prefix('#')
                    .map(|_| trimmed.trim_start_matches('#').trim().to_string())
            })
            .filter(|h| !h.is_empty())
            .collect();
        headings
            .iter()
            .find(|h| id_token.is_match(h))
            .or_else(|| headings.first())
            .cloned()
            .unwrap_or_else(|| id.to_string())
    }
}

/// Read a report file as text: markdown / plain text directly, PDF through
/// `pdftotext -layout` when it is installed.
pub fn read_report(path: &Path) -> Result<String> {
    let is_pdf = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("pdf"));
    if !is_pdf {
        return std::fs::read_to_string(path)
            .wrap_err_with(|| format!("failed to read report {}", path.display()));
    }
    let output = std::process::Command::new("pdftotext")
        .arg("-layout")
        .arg(path)
        .arg("-")
        .output()
        .map_err(|error| {
            eyre!(
                "reading {} needs `pdftotext` (poppler-utils): {error}. Install it, or convert the \
                 report to text/markdown and pass that file instead",
                path.display()
            )
        })?;
    if !output.status.success() {
        return Err(eyre!(
            "pdftotext failed on {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parser() -> PriorReportParser {
        PriorReportParser::new(ProjectVocabulary::from_pairs(&[
            ("VirtualToken", &["cashIn", "cashOut", "transfer", "_update"]),
            ("LamboFactory", &["createLaunchPad", "addVTokenWhiteList"]),
            ("LamboVEthRouter", &["sellQuote", "buyQuote", "withdraw"]),
        ]))
    }

    const C4: &str = "Table of contents\n\
        [H-01] Loss of User Funds in VirtualToken's cashIn Function Due to Incorrect Amount Minting\n\
        [M-03] sellQuote and buyQuote are missing deadline check in LamboVEthRouter\n\
        QA-01 Outdated Curve StableNG Factory Address\n\
        Details\n\
        [H-01] Loss of User Funds in VirtualToken's cashIn Function Due to Incorrect Amount Minting\n\
        Submitted by alice, also found by bob\n\
        The `cashIn` function mints msg.value instead of amount.\n\
        function cashIn(uint256 amount) external payable { _mint(msg.sender, msg.value); }\n\
        [M-03] sellQuote and buyQuote are missing deadline check in LamboVEthRouter\n\
        No deadline parameter; a withdraw is unrelated.\n\
        [M-04] Accumulated ETH will be irretrievable\n\
        ETH sent to the router stays there; users cannot withdraw it.\n\
        QA-01 Outdated Curve StableNG Factory Address\n\
        The factory address changed.\n";

    #[test]
    fn reads_c4_reports_once_per_id_with_the_body() {
        let findings = parser().findings(C4);
        let ids: Vec<&str> = findings.iter().map(|f| f.id.as_str()).collect();
        assert_eq!(ids, ["H-01", "M-03", "QA-01", "M-04"]);
        assert!(findings[0].body.contains("mints msg.value"));
        assert_eq!(findings[0].severity.as_deref(), Some("high"));
        assert_eq!(findings[2].severity.as_deref(), Some("low"));
    }

    #[test]
    fn anchors_on_specific_functions_and_keeps_vague_ones_prompt_only() {
        let imported = parser().import("c4-2024-12", C4, &["gas".to_string()]);
        let h01 = imported.issues.iter().find(|i| i.id == "c4-2024-12:H-01").expect("H-01");
        assert_eq!(h01.contracts, ["VirtualToken"]);
        assert_eq!(h01.match_any, ["cashIn"]);
        assert!(h01.enforce);
        assert!(h01.summary.starts_with("The `cashIn` function mints"), "{}", h01.summary);
        let m03 = imported.issues.iter().find(|i| i.id == "c4-2024-12:M-03").expect("M-03");
        assert_eq!(m03.contracts, ["LamboVEthRouter"]);
        assert!(m03.match_any.contains(&"sellQuote".to_string()));
        // Names no contract: shown, never enforced.
        let m04 = imported.issues.iter().find(|i| i.id == "c4-2024-12:M-04").expect("M-04");
        assert!(!m04.enforce);
    }

    #[test]
    fn reads_guardian_sherlock_and_dedaub_formats() {
        let guardian = "ID         Title                          Category       Severity   Status\n\
            GLOBAL-1   Positive Impact Misrepresented   Logical Error    ● Medium   Acknowledged\n\
            MKTU-2     Funding Fees Partially Paid      Logical Error    ● Critical Resolved\n\
            GLOBAL-1 | Positive Impact Misrepresented\n\
            Description: the positive price impact in cashIn is shown wrongly.\n\
            MKTU-2 | Funding Fees Partially Paid\n\
            Description: funding fees are only partly paid.\n";
        let f = parser().findings(guardian);
        let ids: Vec<&str> = f.iter().map(|x| x.id.as_str()).collect();
        assert_eq!(ids, ["GLOBAL-1", "MKTU-2"], "{f:?}");
        assert_eq!(f[0].severity.as_deref(), Some("medium"), "table severity, not gas");
        assert_eq!(f[1].severity.as_deref(), Some("critical"));
        assert!(f[0].body.contains("price impact"));
        assert_eq!(parser().import("guardian", guardian, &["gas".to_string()]).skipped, 0);

        // Wrapped row: severity on the continuation line, and a category tail
        // `Gas` indented deep in the table must not become a "gas" section.
        let wrapped = "ID         Title                     Category        Severity  Status\n\
            GLOBAL-6   Block Stuffing Attack\n\
                                                       Gas Attack      ● High    Resolved\n\
                                                       Gas\n\
            GLOBAL-6 | Block Stuffing Attack\n\
            Description: keepers can be blocked by stuffing blocks.\n";
        let f = parser().findings(wrapped);
        assert_eq!(f.len(), 1, "{f:?}");
        assert_eq!(f[0].severity.as_deref(), Some("high"));
        assert_eq!(parser().import("g", wrapped, &["gas".to_string()]).skipped, 0);

        let sherlock = "Issue H-1: MarketUtils.getPoolValueInfo() does not use the right price\nbody one\n\
                        Issue M-1: User can lose funds when swapping in cashOut\nbody two\n";
        let f = parser().findings(sherlock);
        assert_eq!(f.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(), ["H-1", "M-1"]);
        assert_eq!(f[0].severity.as_deref(), Some("high"));

        let dedaub = "    C1     Reentrancy vulnerability in cancelOrder        OPEN\nbody\n\
                         M2    Self-close bad debt attack                      OPEN\nbody\n\
                         A1    Gas optimisation in a loop                      RESOLVED\nbody\n";
        let f = parser().findings(dedaub);
        assert_eq!(f.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(), ["C1", "M2", "A1"]);
        assert_eq!(f[0].title, "Reentrancy vulnerability in cancelOrder");
        assert_eq!(f[0].severity.as_deref(), Some("critical"));
        assert_eq!(f[2].severity.as_deref(), Some("informational"));
    }

    #[test]
    fn import_note_reads_a_whole_free_form_file_as_one_issue() {
        // A kill-ledger-shaped note: a banner heading with no id, then the real
        // H1 carrying the ledger id, then free-form prose naming a function.
        // The multi-finding parser finds no `[H-01]`-style ids here.
        let note = "# \u{26a0}\u{fe0f} CANCELLED — earlier claim was wrong\n\
                    # K-008 \u{b7} cashOut unsynced transfer re-used as cover — KILLED\n\
                    - Hypothesis: cashOut transfers collateral without a sync in VirtualToken.\n\
                    Measured P&L 0; killed.\n";
        assert!(parser().findings(note).is_empty());
        let imported = parser().import_note("killledger", "K-008-redeem-unsynced", note);
        assert_eq!(imported.issues.len(), 1);
        let issue = &imported.issues[0];
        assert_eq!(issue.id, "killledger:K-008-redeem-unsynced");
        assert!(issue.title.contains("K-008"), "title picks the id heading: {}", issue.title);
        assert!(!issue.title.contains("CANCELLED"), "title skips the banner: {}", issue.title);
        // Anchored on the function the body names (VirtualToken.cashOut in the
        // test vocabulary), so the summary sees the whole note.
        assert!(issue.match_any.contains(&"cashOut".to_string()));
        assert!(issue.summary.contains("unsynced"), "{}", issue.summary);
    }

    #[test]
    fn spearbit_headings_under_severity_sections() {
        let text = "## High Risk\n### cashIn mints the wrong amount in VirtualToken\nbody about `cashIn`\n\
                    ## Gas Optimizations\n### Cache array length\nbody\n";
        let imported = parser().import("spearbit", text, &["gas".to_string()]);
        assert_eq!(imported.issues.len(), 1);
        assert_eq!(imported.skipped, 1);
        assert!(imported.issues[0].enforce);
        assert_eq!(imported.issues[0].match_any, ["cashIn"]);
    }
}
