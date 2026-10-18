//! The project's own contract / state / function facts, in the shape the
//! invariant miner reasons over.
//!
//! Everything a hypothesis needs to be grounded in *this* project instead of in
//! a historical finding is already in the repo database: which contracts exist
//! and where, what storage they declare, what functions write and read which of
//! it. What was missing was one flat, owned snapshot that a non-database
//! consumer (the miner in `knowdit-audit`) can take by value and test without a
//! connection.
//!
//! The mapping from a function or state variable to its contract comes from the
//! stored relations (`contract_functions`, `contract_variable`) rather than from
//! line ranges: the same file can declare several contracts, and a range guess
//! would silently attribute a variable to the wrong one.

use std::collections::BTreeMap;

use color_eyre::{eyre::WrapErr, Result};
use sea_orm::EntityTrait;
use serde::{Deserialize, Serialize};

use crate::db::{
    contract as contract_model, contract_functions as contract_functions_model,
    contract_variable as contract_variable_model, function as function_model,
    function_state_variable as function_state_variable_model,
    state_variable as state_variable_model,
};

/// One contract declaration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContractFacts {
    pub id: i32,
    pub name: String,
    pub path: String,
    pub line: usize,
}

/// One state variable, attributed to the contract that declares it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateVarFacts {
    pub id: i32,
    pub name: String,
    pub type_name: String,
    pub path: String,
    pub line: usize,
    /// `None` when the storage graph has not been written for this project.
    pub contract: Option<String>,
}

/// One function body, with the storage it touches.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionFacts {
    pub id: i32,
    pub name: String,
    pub contract: Option<String>,
    pub path: String,
    pub line: usize,
    pub content: String,
    /// State-variable ids this function writes, and reads.
    pub writes: Vec<i32>,
    pub reads: Vec<i32>,
}

impl FunctionFacts {
    /// The body with comments removed, lowercased — the form the miner's
    /// substring rules match against. Comments are stripped so a rule cannot be
    /// satisfied by prose (a `// we never call _mint here` is not a mint).
    pub fn code(&self) -> String {
        let stripped = crate::project_facts::strip_comments(&self.content);
        stripped.to_lowercase()
    }
}

/// Everything the invariant miner needs about one project.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectFacts {
    pub contracts: Vec<ContractFacts>,
    pub state_variables: Vec<StateVarFacts>,
    pub functions: Vec<FunctionFacts>,
    /// `(derived, base)` pairs. A variable declared in a base and written by a
    /// derived contract is the *same* storage, not a mirror, and the miner
    /// needs the edges to tell those apart.
    pub inherits: Vec<(String, String)>,
}

/// Whether a source path is vendored code rather than the audited project.
/// The invariant miner skips these: an identity inside OpenZeppelin is true by
/// construction and reporting it would be reporting the library's own tests.
pub fn is_vendored_path(path: &str) -> bool {
    let path = path.trim_start_matches("./");
    path.starts_with("lib/")
        || path.contains("/lib/")
        || path.starts_with("node_modules/")
        || path.contains("/node_modules/")
        || path.starts_with("test/")
        || path.starts_with("tests/")
        || path.starts_with("script/")
        || path.starts_with("scripts/")
        || path.starts_with("out/")
        || path.starts_with("cache/")
}

impl ProjectFacts {
    /// Whether `child` is, or inherits from, `ancestor` (transitively).
    pub fn inherits_from(&self, child: &str, ancestor: &str) -> bool {
        if child == ancestor {
            return true;
        }
        let mut frontier: Vec<&str> = self
            .inherits
            .iter()
            .filter(|(derived, _)| derived == child)
            .map(|(_, base)| base.as_str())
            .collect();
        let mut visited: Vec<&str> = Vec::new();
        while let Some(current) = frontier.pop() {
            if current == ancestor {
                return true;
            }
            if visited.contains(&current) {
                continue;
            }
            visited.push(current);
            frontier.extend(
                self.inherits
                    .iter()
                    .filter(|(derived, _)| derived == current)
                    .map(|(_, base)| base.as_str()),
            );
        }
        false
    }

    /// Whether the two contracts are related by inheritance in either
    /// direction — one storage, two names.
    pub fn related_by_inheritance(&self, left: &str, right: &str) -> bool {
        self.inherits_from(left, right) || self.inherits_from(right, left)
    }

    pub fn contract_by_name(&self, name: &str) -> Option<&ContractFacts> {
        self.contracts
            .iter()
            .find(|candidate| candidate.name == name)
    }

    /// State variables declared by one contract, in declaration order.
    pub fn state_variables_of(&self, contract: &str) -> Vec<&StateVarFacts> {
        self.state_variables
            .iter()
            .filter(|variable| variable.contract.as_deref() == Some(contract))
            .collect()
    }

    /// Functions declared by one contract, in id order.
    pub fn functions_of(&self, contract: &str) -> Vec<&FunctionFacts> {
        self.functions
            .iter()
            .filter(|function| function.contract.as_deref() == Some(contract))
            .collect()
    }

    /// Name of the contract a state-variable id belongs to.
    pub fn contract_of_state_variable(&self, id: i32) -> Option<&str> {
        self.state_variables
            .iter()
            .find(|variable| variable.id == id)
            .and_then(|variable| variable.contract.as_deref())
    }

    /// Index of state variables by id.
    pub fn state_variable_by_id(&self, id: i32) -> Option<&StateVarFacts> {
        self.state_variables.iter().find(|variable| variable.id == id)
    }

    /// Contract names, in declaration order.
    pub fn contract_names(&self) -> Vec<&str> {
        self.contracts
            .iter()
            .map(|contract| contract.name.as_str())
            .collect()
    }
}

/// Read the flat project snapshot the invariant miner consumes.
///
/// Lives free of [`RepoDatabase`] because the connection field is private to
/// the `repo` module; `RepoDatabase::load_project_facts` is the thin wrapper.
pub(crate) async fn load(db: &sea_orm::DatabaseConnection) -> Result<ProjectFacts> {
    {
        let mut facts = ProjectFacts::default();
        let mut contract_name_by_id: BTreeMap<i32, String> = BTreeMap::new();

        let contracts = contract_model::Entity::find()
            .all(db)
            .await
            .wrap_err("failed to load contract rows")?;
        let mut contract_by_id: BTreeMap<i32, String> = BTreeMap::new();
        for contract in contracts {
            contract_by_id.insert(contract.id, contract.name.clone());
            contract_name_by_id.insert(contract.id, contract.name.clone());
            facts.contracts.push(ContractFacts {
                id: contract.id,
                name: contract.name,
                path: contract.relative_file_path,
                line: contract.start_line as usize,
            });
        }

        let variables = state_variable_model::Entity::find()
            .all(db)
            .await
            .wrap_err("failed to load state_variable rows")?;
        let mut variable_owner: BTreeMap<i32, String> = BTreeMap::new();
        for variable in variables {
            facts.state_variables.push(StateVarFacts {
                id: variable.id,
                name: variable.name,
                type_name: variable.type_name,
                path: variable.relative_file_path,
                line: variable.start_line as usize,
                contract: None,
            });
        }

        let owners = contract_variable_model::Entity::find()
            .all(db)
            .await
            .wrap_err("failed to load contract_variable rows")?;
        for owner in owners {
            if let Some(name) = contract_by_id.get(&owner.contract_id) {
                variable_owner.insert(owner.state_variable_id, name.clone());
            }
        }
        for variable in &mut facts.state_variables {
            variable.contract = variable_owner.get(&variable.id).cloned();
        }

        let inherit_edges = crate::db::contract_inherit::Entity::find()
            .all(db)
            .await
            .wrap_err("failed to load contract_inherit rows")?;
        for edge in inherit_edges {
            if let (Some(derived), Some(base)) = (
                contract_name_by_id.get(&edge.contract_id),
                contract_name_by_id.get(&edge.inherited_id),
            ) {
                facts
                    .inherits
                    .push((derived.clone(), base.clone()));
            }
        }

        let mut writes: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
        let mut reads: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
        let edges = function_state_variable_model::Entity::find()
            .all(db)
            .await
            .wrap_err("failed to load function_state_variable rows")?;
        for edge in edges {
            let bucket = if edge.is_write {
                writes.entry(edge.function_id).or_default()
            } else {
                reads.entry(edge.function_id).or_default()
            };
            bucket.push(edge.state_variable_id);
        }

        let mut function_owner: BTreeMap<i32, String> = BTreeMap::new();
        let owners = contract_functions_model::Entity::find()
            .all(db)
            .await
            .wrap_err("failed to load contract_functions rows")?;
        for owner in owners {
            if let Some(name) = contract_by_id.get(&owner.contract_id) {
                function_owner.insert(owner.function_id, name.clone());
            }
        }

        let functions = function_model::Entity::find()
            .all(db)
            .await
            .wrap_err("failed to load function rows")?;
        for function in functions {
            facts.functions.push(FunctionFacts {
                id: function.id,
                name: function.name,
                contract: function_owner.get(&function.id).cloned(),
                path: function.relative_file_path,
                line: function.start_line as usize,
                content: function.content.unwrap_or_default(),
                writes: writes.remove(&function.id).unwrap_or_default(),
                reads: reads.remove(&function.id).unwrap_or_default(),
            });
        }

        Ok(facts)
    }
}

/// Strip line and block comments, leaving the code (and string literals) as
/// written. Shared by the miner and by [`FunctionFacts::code`] so a rule and its
/// test cannot disagree about what counts as code.
pub fn strip_comments(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut chars = source.chars().peekable();
    let mut in_block = false;
    let mut in_line = false;
    let mut in_string: Option<char> = None;
    while let Some(current) = chars.next() {
        if in_line {
            if current == '\n' {
                in_line = false;
                out.push(current);
            }
            continue;
        }
        if in_block {
            if current == '*' && chars.peek() == Some(&'/') {
                chars.next();
                in_block = false;
            }
            continue;
        }
        if let Some(quote) = in_string {
            out.push(current);
            if current == '\\' {
                if let Some(escaped) = chars.next() {
                    out.push(escaped);
                }
                continue;
            }
            if current == quote {
                in_string = None;
            }
            continue;
        }
        match current {
            '/' if chars.peek() == Some(&'/') => {
                chars.next();
                in_line = true;
            }
            '/' if chars.peek() == Some(&'*') => {
                chars.next();
                in_block = true;
            }
            '"' | '\'' => {
                in_string = Some(current);
                out.push(current);
            }
            _ => out.push(current),
        }
    }
    out
}

/// One historical finding's text, joined for the rarity classifier. `patterns`
/// and `root_cause` carry the *how*, which is what a mechanism classifier needs.
pub fn finding_digest(
    title: &str,
    root_cause: &str,
    description: &str,
    patterns: &str,
    exploits: &str,
) -> String {
    [title, root_cause, description, patterns, exploits].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comments_are_stripped_but_string_literals_are_left_alone() {
        let source = "uint256 x; // _mint( - commented out\n/* _burn( */ y = a / b;\n\
                      string memory s = \"_mint(\";";
        let stripped = strip_comments(source);
        assert!(!stripped.contains("// _mint"), "{stripped}");
        assert!(!stripped.contains("/* _burn"), "{stripped}");
        assert!(stripped.contains("_mint("), "string literal survives: {stripped}");
        assert!(stripped.contains("y = a / b"), "{stripped}");
    }

    #[test]
    fn a_function_body_reports_its_code_form() {
        let function = FunctionFacts {
            id: 1,
            name: "deposit".to_string(),
            contract: Some("Vault".to_string()),
            path: "src/Vault.sol".to_string(),
            line: 10,
            content: "function deposit() external payable {\n  // _mint is not called here\n  \
                      uint256 assets = msg.value;\n  _mint(msg.sender, assets);\n}"
                .to_string(),
            writes: vec![],
            reads: vec![],
        };
        let code = function.code();
        assert!(code.contains("_mint(msg.sender"), "{code}");
        assert!(!code.contains("is not called here"), "{code}");
        assert!(code.contains("msg.value"), "{code}");
    }
}
