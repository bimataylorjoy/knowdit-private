//! The knowledge graph as a pattern library the agent searches, rather than
//! the queue that decides what gets looked at.
//!
//! A target-derived hypothesis arrives with no historical finding attached.
//! When the agent wants a precedent — how a share-rate identity was broken
//! elsewhere, what a mirror desync looked like in practice — it asks for one
//! with `search_patterns`, and gets the closest findings the project's mirror
//! holds. The search is lexical and deterministic: the corpus is small (the
//! findings the mapper pulled for this project), and a ranking the agent can
//! predict from its own query is worth more here than a fuzzy one.

use knowdit_kg_model::db::audit_finding;

use super::rarity;

/// One searchable finding.
#[derive(Debug, Clone)]
struct PatternEntry {
    finding: audit_finding::Model,
    mechanism: Option<&'static str>,
    title: String,
    root_cause: String,
    body: String,
}

/// Lexical index over the mirrored historical findings.
#[derive(Debug, Clone, Default)]
pub struct PatternLibrary {
    entries: Vec<PatternEntry>,
}

impl PatternLibrary {
    pub fn new(findings: Vec<audit_finding::Model>) -> Self {
        let entries = findings
            .into_iter()
            .map(|finding| {
                let text = format!(
                    "{} {} {} {} {}",
                    finding.title,
                    finding.root_cause,
                    finding.description,
                    finding.patterns,
                    finding.exploits
                );
                PatternEntry {
                    mechanism: rarity::classify(&text),
                    title: finding.title.to_lowercase(),
                    root_cause: finding.root_cause.to_lowercase(),
                    body: format!(
                        "{} {} {}",
                        finding.description, finding.patterns, finding.exploits
                    )
                    .to_lowercase(),
                    finding,
                }
            })
            .collect();
        Self { entries }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The `limit` findings that best match `query`, rendered for a prompt.
    /// Terms shorter than three characters are ignored; a title hit weighs
    /// three, a root-cause hit two, a body hit one.
    pub fn search(&self, query: &str, limit: usize) -> String {
        let terms: Vec<String> = query
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|term| term.len() >= 3)
            .map(str::to_string)
            .collect();
        if terms.is_empty() {
            return "error: give at least one search term of three or more characters".to_string();
        }
        let mut scored: Vec<(usize, &PatternEntry)> = self
            .entries
            .iter()
            .map(|entry| {
                let score = terms
                    .iter()
                    .map(|term| {
                        3 * usize::from(entry.title.contains(term.as_str()))
                            + 2 * usize::from(entry.root_cause.contains(term.as_str()))
                            + usize::from(entry.body.contains(term.as_str()))
                    })
                    .sum::<usize>();
                (score, entry)
            })
            .filter(|(score, _)| *score > 0)
            .collect();
        scored.sort_by(|(left_score, left), (right_score, right)| {
            right_score
                .cmp(left_score)
                .then(left.finding.id.cmp(&right.finding.id))
        });
        if scored.is_empty() {
            return format!(
                "no precedent in the {} finding(s) this project's mirror holds matches `{query}` — \
                 which is itself a signal: a failure mode with no precedent is what this audit is \
                 looking for",
                self.entries.len()
            );
        }
        let mut out = format!(
            "{} precedent(s) for `{query}` (showing {}):\n",
            scored.len(),
            scored.len().min(limit)
        );
        for (score, entry) in scored.into_iter().take(limit) {
            let finding = &entry.finding;
            out.push_str(&format!(
                "\n### finding {} — {} (score {score}{})\n- root cause: {}\n- patterns: {}\n- failure paths: {}\n",
                finding.id,
                finding.title.trim(),
                entry
                    .mechanism
                    .map(|m| format!(", mechanism {m}"))
                    .unwrap_or_default(),
                Self::clip(&finding.root_cause, 400),
                Self::clip(&finding.patterns, 400),
                Self::clip(&finding.exploits, 400),
            ));
        }
        out
    }

    fn clip(text: &str, max: usize) -> String {
        let text = text.trim();
        if text.chars().count() <= max {
            return text.to_string();
        }
        let kept: String = text.chars().take(max).collect();
        format!("{kept}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use knowdit_kg_model::audit_finding::FindingSeverity;

    fn finding(id: i32, title: &str, root_cause: &str) -> audit_finding::Model {
        audit_finding::Model {
            id,
            title: title.to_string(),
            severity: FindingSeverity::High,
            root_cause: root_cause.to_string(),
            description: String::new(),
            patterns: String::new(),
            exploits: String::new(),
        }
    }

    #[test]
    fn ranks_title_hits_first_and_reports_absence() {
        let library = PatternLibrary::new(vec![
            finding(1, "Rounding in withdraw", "share price rounds up"),
            finding(2, "Share inflation via donation", "first depositor"),
        ]);
        let out = library.search("share inflation", 5);
        let first = out.find("finding 2").expect("finding 2 listed");
        let second = out.find("finding 1").expect("finding 1 listed");
        assert!(first < second, "{out}");
        assert!(library.search("bridge replay", 5).starts_with("no precedent"));
        assert!(library.search("ab", 5).starts_with("error"));
    }
}
