//! Source of the external contracts a project calls but does not contain.
//!
//! A protocol that integrates a core it does not own — a CDP engine on top of a
//! shared accounting core, a vault against an Aave pool — reaches that core
//! through an interface, so the agent sees the call as opaque and has to guess
//! what it does. Guessed semantics are a common source of false positives: an
//! agent assumes a call credits collateral without debiting it, or that it skips
//! a solvency check it actually enforces, or invents a `require` that does not
//! exist in the dependency.
//!
//! This module loads the real source of those dependencies (operator-supplied
//! files or directories) and renders it as a ground-truth block for the agent's
//! system prompt. The block sits in the run-wide cached prefix next to the
//! resident project source, so it is paid for once per run, not once per link.

use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, WrapErr, eyre};
use knowdit_repo_model::project_facts::strip_comments;

/// Ceiling on the rendered block, in characters. The block is re-billed at the
/// cached-input rate on every call, so it must stay a few dependency contracts,
/// not a vendored library tree. A file that would cross the ceiling is listed
/// by name only.
pub const MAX_DEPENDENCY_CHARS: usize = 120_000;

/// One loaded dependency source file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyFile {
    /// Display label: the path as the operator gave it, relative where possible.
    pub label: String,
    /// Source with comments stripped (license headers and NatSpec are most of a
    /// dss file's bytes and none of its behaviour).
    pub source: String,
}

/// The loaded set. Empty when no `--gen-specs-dependency-source` was given.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DependencySources {
    pub files: Vec<DependencyFile>,
}

impl DependencySources {
    /// Load every `.sol` file named by `paths` (a file, or a directory searched
    /// recursively), sorted by path so the rendered prefix is stable across runs
    /// and the prompt cache keeps hitting.
    pub fn load(paths: &[PathBuf]) -> Result<Self> {
        let mut found: Vec<PathBuf> = Vec::new();
        for path in paths {
            if path.is_dir() {
                Self::collect_dir(path, &mut found)?;
            } else if path.is_file() {
                found.push(path.clone());
            } else {
                return Err(eyre!(
                    "dependency source {} does not exist",
                    path.display()
                ));
            }
        }
        found.sort();
        found.dedup();
        let mut files = Vec::new();
        for path in found {
            let raw = std::fs::read_to_string(&path)
                .wrap_err_with(|| format!("failed to read dependency source {}", path.display()))?;
            files.push(DependencyFile {
                label: path.display().to_string(),
                source: strip_comments(&raw)
                    .lines()
                    .filter(|line| !line.trim().is_empty())
                    .collect::<Vec<_>>()
                    .join("\n"),
            });
        }
        Ok(Self { files })
    }

    fn collect_dir(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        for entry in std::fs::read_dir(dir)
            .wrap_err_with(|| format!("failed to list dependency dir {}", dir.display()))?
        {
            let path = entry?.path();
            if path.is_dir() {
                Self::collect_dir(&path, out)?;
            } else if path.extension().and_then(|e| e.to_str()) == Some("sol") {
                out.push(path);
            }
        }
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// The ground-truth block for a system prompt, or an empty string when no
    /// dependency source was loaded. Files are included whole until
    /// [`MAX_DEPENDENCY_CHARS`]; the rest are named so the agent knows they
    /// exist and can read them with its file tools.
    pub fn render_block(&self) -> String {
        if self.files.is_empty() {
            return String::new();
        }
        let mut block = String::from(
            "# External dependency source (ground truth)\n\n\
             The project calls these contracts but does not contain them, so its own \
             source only shows their interfaces. Their behaviour is fixed by the source \
             below. Before a specification claims that a call into one of them reverts, \
             credits, debits, moves balances or checks a condition, find that code here \
             and reason from it. A `require`, revert reason or side effect that does not \
             appear in this source does not exist — do not build a specification on it. \
             In particular, re-derive every balance change of a dependency call from its \
             body (which storage slots it adds to and which it subtracts from) instead of \
             inferring it from the function's name.\n\n",
        );
        let mut used = block.len();
        let mut omitted: Vec<&str> = Vec::new();
        for file in &self.files {
            let section = format!("## {}\n```solidity\n{}\n```\n\n", file.label, file.source);
            if used + section.len() > MAX_DEPENDENCY_CHARS {
                omitted.push(&file.label);
                continue;
            }
            used += section.len();
            block.push_str(&section);
        }
        if !omitted.is_empty() {
            block.push_str(&format!(
                "Further dependency files exist but did not fit; read them with the file \
                 tools when a claim depends on them: {}\n\n",
                omitted.join(", ")
            ));
        }
        block
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("knowdit-dep-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).expect("create temp dir");
        dir
    }

    #[test]
    fn loads_directory_recursively_sorted_and_strips_comments() {
        let dir = tmp_dir("load");
        std::fs::write(
            dir.join("vat.sol"),
            "// SPDX-License-Identifier: AGPL\n/* header */\ncontract Vat {\n  // note\n  function grab() external {}\n}\n",
        )
        .expect("write");
        std::fs::write(dir.join("sub/dog.sol"), "contract Dog {}\n").expect("write");
        std::fs::write(dir.join("README.md"), "not solidity").expect("write");
        let deps = DependencySources::load(&[dir.clone()]).expect("load");
        assert_eq!(deps.files.len(), 2, "only .sol files");
        assert!(deps.files[0].label.ends_with("sub/dog.sol"), "sorted by path");
        let vat = &deps.files[1].source;
        assert!(vat.contains("function grab()"));
        assert!(!vat.contains("SPDX") && !vat.contains("header") && !vat.contains("note"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_path_is_an_error() {
        let err = DependencySources::load(&[PathBuf::from("/definitely/not/here.sol")]);
        assert!(err.is_err());
    }

    #[test]
    fn empty_set_renders_nothing_and_oversized_files_are_named() {
        assert_eq!(DependencySources::default().render_block(), "");
        let deps = DependencySources {
            files: vec![
                DependencyFile { label: "vat.sol".into(), source: "contract Vat {}".into() },
                DependencyFile { label: "huge.sol".into(), source: "x".repeat(MAX_DEPENDENCY_CHARS) },
            ],
        };
        let block = deps.render_block();
        assert!(block.contains("## vat.sol"));
        assert!(!block.contains("## huge.sol"));
        assert!(block.contains("did not fit") && block.contains("huge.sol"));
        assert!(block.contains("does not exist"), "carries the no-invented-require rule");
    }
}
