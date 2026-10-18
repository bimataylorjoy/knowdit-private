//! The project's own tests as an oracle.
//!
//! A test suite is the only place a team writes down, executably, what its code
//! is supposed to do: `assertEq(vault.totalAssets(), deposits + yield)` is a
//! statement about the system that no audit page or NatSpec block repeats. Two
//! consequences follow, and both are cheap to act on:
//!
//! * a specification whose violation **contradicts** an assertion the project
//!   itself makes is a strong candidate — the team has already agreed the
//!   quantity must hold;
//! * a specification whose reached state **matches** an assertion the project
//!   makes is `ExpectedViolation` by definition, and grading it takes a run.
//!
//! This module mines those assertions. It is deliberately blind to what they
//! mean: it collects the line, the file, and which in-scope contract it names,
//! and the agent reads the surrounding test before using it. The value is not
//! that the statements are true — a test can be wrong — it is that they are
//! the project's own, which is the standard a report gets judged against.

use std::path::{Path, PathBuf};

/// One assertion found in the project's tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestAssertion {
    /// The in-scope contract this assertion names.
    pub contract: String,
    pub path: String,
    pub line: usize,
    pub text: String,
}

/// Assertion builders worth surfacing. `expectRevert` is included: a documented
/// revert is the project saying "this state is unreachable", which is exactly
/// what an `ExpectedViolation` verdict needs to cite.
const ASSERTION_MARKERS: [&str; 10] = [
    "asserteq(",
    "assertapproxeq",
    "assertgt(",
    "assertge(",
    "assertlt(",
    "assertle(",
    "asserttrue(",
    "assertfalse(",
    "expectrevert",
    "assertnotapproxeq",
];

/// Directories that hold tests, in the order a Solidity repo usually has them.
const TEST_DIRS: [&str; 2] = ["test", "tests"];

/// Mining limits. A suite can hold thousands of assertions; a prompt cannot,
/// and the agent's step budget is finite.
pub const MAX_TEST_FILES: usize = 40;
pub const MAX_ASSERTIONS: usize = 25;

pub struct TestOracle;

impl TestOracle {
    /// Scan a project for assertions that name one of `contracts`.
    pub fn mine(root: &Path, contracts: &[String]) -> Vec<TestAssertion> {
        let mut sources: Vec<(String, String)> = Vec::new();
        for directory in TEST_DIRS {
            let base = root.join(directory);
            if base.is_dir() {
                Self::collect(&base, root, &mut sources, 0);
            }
            if sources.len() >= MAX_TEST_FILES {
                break;
            }
        }
        Self::mine_from_sources(&sources, contracts)
    }

    /// The pure core: classify already-read sources. Kept separate so the rules
    /// are testable without a filesystem.
    pub fn mine_from_sources(
        sources: &[(String, String)],
        contracts: &[String],
    ) -> Vec<TestAssertion> {
        let mut found: Vec<TestAssertion> = Vec::new();
        for (path, content) in sources.iter().take(MAX_TEST_FILES) {
            let stripped = knowdit_repo_model::project_facts::strip_comments(content);
            for (index, line) in stripped.lines().enumerate() {
                let lower = line.to_lowercase();
                if !ASSERTION_MARKERS.iter().any(|marker| lower.contains(marker)) {
                    continue;
                }
                let Some(contract) = contracts.iter().find(|contract| {
                    let name = contract
                        .rsplit('/')
                        .next()
                        .unwrap_or(contract)
                        .trim_end_matches(".sol")
                        .split('.')
                        .next()
                        .unwrap_or(contract)
                        .to_lowercase();
                    !name.is_empty() && lower.contains(&name)
                }) else {
                    continue;
                };
                found.push(TestAssertion {
                    contract: contract.clone(),
                    path: path.clone(),
                    line: index + 1,
                    text: line.trim().to_string(),
                });
                if found.len() >= MAX_ASSERTIONS {
                    return found;
                }
            }
        }
        found
    }

    fn collect(directory: &Path, root: &Path, out: &mut Vec<(String, String)>, depth: usize) {
        if depth > 4 || out.len() >= MAX_TEST_FILES {
            return;
        }
        let Ok(entries) = std::fs::read_dir(directory) else {
            return;
        };
        let mut children: Vec<PathBuf> = entries.flatten().map(|entry| entry.path()).collect();
        children.sort();
        for child in children {
            if out.len() >= MAX_TEST_FILES {
                return;
            }
            let name = child
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_default();
            if child.is_dir() {
                // Vendored and generated trees are not this project's tests.
                if matches!(name.as_str(), "lib" | "node_modules" | "out" | "cache" | "artifacts") {
                    continue;
                }
                Self::collect(&child, root, out, depth + 1);
                continue;
            }
            if !name.ends_with(".sol") {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(&child) else {
                continue;
            };
            let relative = child
                .strip_prefix(root)
                .unwrap_or(&child)
                .to_string_lossy()
                .to_string();
            out.push((relative, content));
        }
    }
}

/// The invariants that touch any of `contracts`.
pub fn relevant<'a>(
    assertions: &'a [TestAssertion],
    contracts: &[String],
) -> Vec<&'a TestAssertion> {
    assertions
        .iter()
        .filter(|assertion| {
            contracts.iter().any(|contract| {
                crate::scope::same_contract(&assertion.contract, &super::invariants::contract_ref(contract))
            })
        })
        .collect()
}

/// The prompt block: the project's own statements, and what each outcome means.
pub fn render_block(assertions: &[&TestAssertion]) -> String {
    if assertions.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "## What this project's own tests assert\n\n\
         These lines are the team's own executable statements about this code, mined from its test \
         suite. They are the standard this work gets judged against:\n\n\
         - a violation that **contradicts** one of them is strong evidence — the team already \
           agreed the quantity must hold;\n\
         - a reached state that **matches** one of them is expected behaviour, not a finding — say \
           so rather than reporting it;\n\
         - a test can be wrong or stale; read it before leaning on it.\n\n",
    );
    for assertion in assertions {
        out.push_str(&format!(
            "- `{}` asserts at {}:{} — `{}`\n",
            assertion.contract,
            assertion.path,
            assertion.line,
            if assertion.text.len() > 160 {
                format!("{}…", &assertion.text[..160])
            } else {
                assertion.text.clone()
            }
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sources() -> Vec<(String, String)> {
        vec![
            (
                "test/Vault.t.sol".to_string(),
                "function test_deposit() public {\n  \
                 vault.deposit(1 ether);\n  \
                 assertEq(vault.totalAssets(), 1 ether, \"deposit is accounted\");\n  \
                 assertGt(router.quote(), 0);\n}\n"
                    .to_string(),
            ),
            (
                "test/Other.t.sol".to_string(),
                "// assertEq(vault.totalAssets(), 999);\n\
                 function test_unrelated() public { assertEq(1, 1); }\n"
                    .to_string(),
            ),
        ]
    }

    #[test]
    fn assertions_naming_an_in_scope_contract_are_mined() {
        let contracts = vec!["Vault".to_string(), "Router".to_string()];
        let found = TestOracle::mine_from_sources(&sources(), &contracts);
        assert_eq!(found.len(), 2, "{found:?}");
        assert!(found[0].text.contains("totalAssets"));
        assert_eq!(found[0].path, "test/Vault.t.sol");
        assert_eq!(found[0].line, 3, "the line number is the assertion's");
        assert!(found[1].text.contains("quote"));
    }

    #[test]
    fn a_commented_out_assertion_is_not_an_assertion() {
        let contracts = vec!["Vault".to_string()];
        let found = TestOracle::mine_from_sources(&sources(), &contracts);
        assert!(
            !found.iter().any(|assertion| assertion.line == 1 && assertion.path.contains("Other")),
            "the commented-out line must not be mined: {found:?}"
        );
    }

    #[test]
    fn only_assertions_that_name_an_in_scope_contract_are_kept() {
        let found = TestOracle::mine_from_sources(&sources(), &["Vault".to_string()]);
        assert!(
            found.iter().all(|assertion| assertion.contract == "Vault"),
            "{found:?}"
        );
        // `assertEq(1, 1)` names nothing in scope and is skipped.
        assert!(found.len() == 1, "{found:?}");
    }

    #[test]
    fn contract_names_match_however_the_caller_spells_them() {
        let contracts = vec!["src/Vault.sol".to_string()];
        let found = TestOracle::mine_from_sources(&sources(), &contracts);
        assert!(!found.is_empty(), "a path has to match the bare name");
    }

    #[test]
    fn the_block_says_what_each_outcome_means() {
        let contracts = vec!["Vault".to_string()];
        let found = TestOracle::mine_from_sources(&sources(), &contracts);
        let relevant = relevant(&found, &contracts);
        let block = render_block(&relevant);
        assert!(block.contains("contradicts"), "{block}");
        assert!(block.contains("expected behaviour"), "{block}");
        assert!(block.contains("test/Vault.t.sol:3"), "{block}");
        assert!(render_block(&[]).is_empty());
    }
}
