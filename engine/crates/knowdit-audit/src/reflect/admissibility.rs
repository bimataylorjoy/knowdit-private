//! Deterministic production-path admissibility of a synthesized proof.
//!
//! An audit program that pays for reproducible exploits accepts a finding
//! only when its proof exercises the *deployed* protocol: the deployed
//! code, its own storage, its own balances. A harness that deploys the
//! project's contracts into an empty EVM and mocks whatever they talk to
//! can produce a perfectly reproducible revert while proving nothing about
//! the system under audit — the state it violates was manufactured by the
//! harness itself.
//!
//! That distinction is not a judgement call, so it does not live in a
//! prompt. Everything here is a pure read of the committed harness source
//! against the project's own contract list, and the result names the exact
//! ways the reproduction fails the requirement. Callers use it to keep a
//! scaffolded reproduction out of the reportable body while still recording
//! it as an investigated candidate.
//!
//! The direction of each signal matters. A missed signal leaves an
//! unproven claim in the report; a spurious signal only demotes a finding
//! that an agent can still argue for. Both scanners therefore read
//! comment-stripped source: a harness that merely *mentions* a fork
//! cheatcode in prose must not pass the fork requirement.
//!
//! Forking is not enough on its own: a fork at head puts the proof on state
//! that keeps moving (oracle rounds, liquidity, a proxy upgraded since), so
//! whoever re-runs it cannot tell a reproduced defect from a changed chain.
//! Callers whose run resolved a block therefore pass
//! [`PocAdmissibilityPolicy::require_pinned_fork`] and a head fork is reported
//! as [`ScaffoldSignal::UnpinnedFork`]. A run with no block to offer leaves the
//! requirement off — the rule is only ever as strong as what the run knows.

use std::collections::BTreeSet;

/// One concrete way a harness fails to exercise the deployed system.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ScaffoldSignal {
    /// No fork of any live deployment appears in the harness: every
    /// contract under test was constructed fresh, so no observed state
    /// belongs to the deployed protocol.
    NoFork,
    /// The harness forks, but never pins the fork to a block. Everything the
    /// proof depends on — oracle rounds, pool liquidity, whether a proxy has
    /// been upgraded since — keeps moving, so the same harness can violate
    /// today and pass tomorrow and nobody re-running it can tell which
    /// happened. Raised only when the run resolved a block to pin to; a run
    /// that has no block to offer cannot demand one.
    UnpinnedFork,
    /// The harness replaces a project contract's response with a cheatcode
    /// instead of calling it. What the proof then exercises is the harness's own
    /// value, not the protocol's behaviour — so the defect it demonstrates is
    /// one the harness author wrote, not one the deployed code has.
    MockedProjectContract { contract: String },
    /// The harness deploys one of the project's own contracts itself
    /// (`new <ProjectContract>(`) instead of resolving the deployed
    /// instance at its real address.
    Redeployed { contract: String },
    /// The harness writes a contract's storage or code directly through a
    /// cheatcode, fabricating a state that no reachable transaction could
    /// have produced.
    FabricatedState { cheatcode: String },
}

impl ScaffoldSignal {
    /// Compact label for the inline advisory the harness agent reads.
    pub fn label(&self) -> String {
        match self {
            // Phrased as a property of the proof, not as a skipped
            // opportunity: the engine cannot tell "chose not to fork" from
            // "this engagement had no fork endpoint", so the signal must not
            // claim a deployment was there to be forked.
            Self::NoFork => {
                "does not exercise deployed state (the proof builds the world itself)".to_string()
            }
            Self::UnpinnedFork => {
                "forks the deployed system but never pins the fork to a block".to_string()
            }
            Self::MockedProjectContract { contract } => {
                format!("stubs project contract `{contract}` out with a mock cheatcode")
            }
            Self::Redeployed { contract } => {
                format!("deploys project contract `{contract}` itself")
            }
            Self::FabricatedState { cheatcode } => {
                format!("fabricates contract state with `{cheatcode}`")
            }
        }
    }
}

/// What the caller requires of a proof, over and above the standing rules.
///
/// Defaults to the standing rules alone, which is what a run with no resolved
/// fork block can honestly enforce.
#[derive(Debug, Clone, Copy, Default)]
pub struct PocAdmissibilityPolicy {
    /// A fork must be pinned to a block. Set by callers whose run resolved the
    /// block for this chain, because pinning is the difference between a proof
    /// somebody else can re-run and one that only ever held once.
    pub require_pinned_fork: bool,
}

/// What one harness source proves about the deployed system.
#[derive(Debug, Clone)]
pub struct PocAdmissibility {
    signals: BTreeSet<ScaffoldSignal>,
}

impl PocAdmissibility {
    /// Foundry's fork entry points. Any one of them means the harness
    /// resolved live state instead of building the world itself.
    const FORK_CHEATCODES: [&'static str; 5] = [
        "createSelectFork",
        "createFork",
        "selectFork",
        "rollFork",
        "activeFork",
    ];

    /// The subset whose second argument is the block: `createSelectFork(url,
    /// block)`, `createFork(url, block)`, `selectFork(id, block)`. A call to one
    /// of these with a single argument forks at head.
    const BLOCK_SELECTING_FORK_CALLS: [&'static str; 3] =
        ["createSelectFork", "createFork", "selectFork"];

    /// Cheatcodes that write contract storage / code. `vm.load` and the
    /// value-transfer helpers (`deal`, `prank`) are deliberately absent:
    /// they read state or fund an actor, neither of which substitutes for
    /// the protocol's own state.
    const FABRICATION_CHEATCODES: [&'static str; 2] = ["vm.store", "vm.etch"];

    /// Cheatcodes that replace a contract's response. Mocking a *dependency*
    /// (a token the project already fixes, a feed the program calls honest) is
    /// legitimate and stays unflagged; mocking one of the project's own
    /// contracts replaces the behaviour under audit, which is the one thing a
    /// proof of that behaviour cannot do.
    const MOCK_CHEATCODES: [&'static str; 5] = [
        "vm.mockCall",
        "vm.mockCalls",
        "vm.mockCallRevert",
        "vm.mockFunction",
        "vm.mockFunctionCall",
    ];

    /// Cheatcodes that deploy a contract from the project's own build output.
    /// Same failure as `new <ProjectContract>(`, different spelling: the
    /// harness puts its own copy of the code on chain and tests that.
    const DEPLOY_CODE_CHEATCODE: &'static str = "vm.deployCode";

    /// Classify one committed harness source under the standing rules alone.
    /// `is_project_contract` is the caller's own project-scope predicate — the
    /// harness layer holds a raw call graph, the review layer holds a
    /// `ProjectIndex`, and both must answer with the same notion of "this
    /// project's contract".
    pub fn of_harness(source: &str, is_project_contract: impl Fn(&str) -> bool) -> Self {
        Self::of_harness_with(source, is_project_contract, PocAdmissibilityPolicy::default())
    }

    /// Classify one committed harness source under the standing rules plus
    /// whatever the caller requires of this run.
    pub fn of_harness_with(
        source: &str,
        is_project_contract: impl Fn(&str) -> bool,
        policy: PocAdmissibilityPolicy,
    ) -> Self {
        let code = Self::strip_comments(source);
        let mut signals: BTreeSet<ScaffoldSignal> = BTreeSet::new();

        let forks: Vec<(String, usize)> = Self::fork_calls(&code);
        if forks.is_empty() {
            signals.insert(ScaffoldSignal::NoFork);
        } else if policy.require_pinned_fork && !Self::pins_a_block(&forks, &code) {
            signals.insert(ScaffoldSignal::UnpinnedFork);
        }

        for contract in Self::deployed_contracts(&code) {
            if is_project_contract(&contract) {
                signals.insert(ScaffoldSignal::Redeployed { contract });
            }
        }

        // The same failure by another route: `vm.deployCode("src/Vault.sol:Vault")`
        // puts the project's own build output on chain.
        for arguments in Self::argument_lists(&code, &[Self::DEPLOY_CODE_CHEATCODE]) {
            if let Some(contract) = Self::named_project_contract(arguments, &is_project_contract) {
                signals.insert(ScaffoldSignal::Redeployed { contract });
            }
        }

        // A mock aimed at a project contract rather than at a dependency.
        for arguments in Self::argument_lists(&code, &Self::MOCK_CHEATCODES) {
            if let Some(contract) = Self::named_project_contract(arguments, &is_project_contract) {
                signals.insert(ScaffoldSignal::MockedProjectContract { contract });
            }
        }

        for cheatcode in Self::FABRICATION_CHEATCODES {
            if Self::invokes(&code, cheatcode) {
                signals.insert(ScaffoldSignal::FabricatedState {
                    cheatcode: cheatcode.to_string(),
                });
            }
        }

        Self { signals }
    }

    /// Every fork cheatcode call in the source, with its top-level argument
    /// count — the count is what tells a head fork from a pinned one.
    fn fork_calls(code: &str) -> Vec<(String, usize)> {
        let mut out: Vec<(String, usize)> = Vec::new();
        for name in Self::FORK_CHEATCODES {
            let mut from = 0;
            while let Some(rel) = code[from..].find(name) {
                let at = from + rel;
                from = at + name.len();
                if let Some(args) = Self::call_arg_count(&code[from..]) {
                    out.push((name.to_string(), args));
                }
            }
        }
        out
    }

    /// Whether any fork call fixed the block it stands on: one of the
    /// block-selecting calls with two arguments, or a `rollFork(block)`.
    fn pins_a_block(forks: &[(String, usize)], code: &str) -> bool {
        let selects_a_block = forks.iter().any(|(name, args)| {
            *args >= 2 && Self::BLOCK_SELECTING_FORK_CALLS.contains(&name.as_str())
        });
        selects_a_block || Self::invokes(code, "rollFork")
    }

    /// Argument count of the call whose opening paren starts `after_name`,
    /// when what follows the name is a call. Depth-aware (nested calls count as
    /// one argument) and string-literal aware, so `f(a, g(b, c))` is two
    /// arguments and `f("a,b")` is one.
    fn call_arg_count(after_name: &str) -> Option<usize> {
        let mut chars = after_name.char_indices();
        let (_, first) = chars.find(|(_, c)| !c.is_whitespace())?;
        if first != '(' {
            return None;
        }
        // The opening paren is consumed; everything after it is inside the
        // argument list.
        let mut depth = 1usize;
        let mut commas = 0usize;
        let mut saw_any = false;
        let mut in_string: Option<char> = None;
        let mut escaped = false;
        for (_, c) in chars {
            if let Some(quote) = in_string {
                if escaped {
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == quote {
                    in_string = None;
                }
                continue;
            }
            match c {
                '"' | '\'' => {
                    in_string = Some(c);
                    saw_any = true;
                }
                '(' => {
                    depth += 1;
                    saw_any = true;
                }
                ')' => {
                    if depth == 1 {
                        return Some(if saw_any { commas + 1 } else { 0 });
                    }
                    depth -= 1;
                }
                ',' if depth == 1 => commas += 1,
                c if !c.is_whitespace() => saw_any = true,
                _ => {}
            }
        }
        None
    }

    /// True when the harness reproduces the defect against deployed state.
    pub fn is_admissible(&self) -> bool {
        self.signals.is_empty()
    }

    pub fn signals(&self) -> impl Iterator<Item = &ScaffoldSignal> {
        self.signals.iter()
    }

    /// One line per signal, for the inline advisory.
    pub fn signal_labels(&self) -> Vec<String> {
        self.signals.iter().map(ScaffoldSignal::label).collect()
    }

    /// The report-facing justification, phrased as the requirement it
    /// fails rather than as a workflow state. Prefixed to the review reason
    /// so the record reads as an investigation result, not a silent drop.
    pub fn clause(&self) -> String {
        let found = self.signal_labels().join("; ");
        // A head fork exercises the deployed protocol; what it does not do is
        // fix *which* deployed state the proof is about, and that is the
        // requirement it fails.
        if self.signals.len() == 1 && self.signals.contains(&ScaffoldSignal::UnpinnedFork) {
            return format!(
                "Not reportable as reproduced: the proof forks the deployed protocol but does not \
                 pin the chain state it stands on ({found}). A proof whose state keeps moving \
                 cannot be re-run by whoever triages it — the same harness can violate today and \
                 pass tomorrow, and the difference is not the code. Fork at the run's block and \
                 record it."
            );
        }
        format!(
            "Not reportable as reproduced: the proof is not against the deployed protocol \
             ({found}). Where a deployment is reachable, a reproducible proof against it is \
             required before a finding is reportable; this harness demonstrates the behaviour \
             in a synthetic deployment only."
        )
    }

    /// Drop comments so a fork cheatcode named in prose cannot satisfy the
    /// fork requirement. Shared with the fork-harness scan in
    /// [`crate::harness::solidity`], which asks the same question of a whole
    /// file — one implementation, so the two cannot disagree about what counts
    /// as code. Line and block comments are tracked across lines;
    /// string literals are left alone, which can only ever hide good news.
    pub fn strip_comments(source: &str) -> String {
        let mut out = String::with_capacity(source.len());
        let mut chars = source.chars().peekable();
        let mut in_block = false;
        while let Some(c) = chars.next() {
            if in_block {
                if c == '*' && chars.peek() == Some(&'/') {
                    chars.next();
                    in_block = false;
                }
                continue;
            }
            match c {
                '/' if chars.peek() == Some(&'/') => {
                    while let Some(&n) = chars.peek() {
                        if n == '\n' {
                            break;
                        }
                        chars.next();
                    }
                }
                '/' if chars.peek() == Some(&'*') => {
                    chars.next();
                    in_block = true;
                }
                _ => out.push(c),
            }
        }
        out
    }

    /// The arguments of every call to one of `names`, as written in the source.
    /// Used by the two rules that care what a cheatcode is aimed at — a mock is
    /// only a problem when it stubs a project contract, and a `deployCode` only
    /// one when its artifact is the project's own.
    ///
    /// The whole argument list, not just the first argument: the mocked target
    /// is usually held in a local (`vm.mockCall(address(vault), …)`), so the
    /// contract's name is what appears in the selector beside it.
    fn argument_lists<'a>(code: &'a str, names: &[&str]) -> Vec<&'a str> {
        let mut out = Vec::new();
        for name in names {
            let mut from = 0;
            while let Some(rel) = code[from..].find(name) {
                let at = from + rel;
                from = at + name.len();
                if let Some(arguments) = Self::parenthesised(&code[from..]) {
                    out.push(arguments);
                }
            }
        }
        out
    }

    /// The text between a call's parentheses, when what follows `after_name` is
    /// a call at all. Brackets and braces are tracked so a comma inside them
    /// cannot end the list.
    fn parenthesised(after_name: &str) -> Option<&str> {
        let open = after_name.find('(')?;
        let inner = &after_name[open + 1..];
        let mut depth = 1usize;
        let mut in_string: Option<char> = None;
        let mut escaped = false;
        for (index, c) in inner.char_indices() {
            if let Some(quote) = in_string {
                if escaped {
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == quote {
                    in_string = None;
                }
                continue;
            }
            match c {
                '"' | '\'' => in_string = Some(c),
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => {
                    if depth == 1 {
                        return Some(&inner[..index]);
                    }
                    depth -= 1;
                }
                _ => {}
            }
        }
        None
    }

    /// Which of the project's contracts these arguments name, if any. Read
    /// identifier by identifier, so only a real project contract name counts
    /// and a mock of a dependency stays unflagged.
    fn named_project_contract(
        arguments: &str,
        is_project_contract: &impl Fn(&str) -> bool,
    ) -> Option<String> {
        let mut token = String::new();
        for c in arguments.chars().chain(std::iter::once(' ')) {
            if c.is_alphanumeric() || c == '_' || c == '$' {
                token.push(c);
                continue;
            }
            if token.is_empty() {
                continue;
            }
            let named = is_project_contract(&token);
            if named {
                return Some(token);
            }
            token.clear();
        }
        None
    }

    /// True when `name` appears as a call — the identifier followed, after
    /// optional spaces, by `(`.
    fn invokes(code: &str, name: &str) -> bool {
        let mut from = 0;
        while let Some(rel) = code[from..].find(name) {
            let at = from + rel;
            let after = &code[at + name.len()..];
            if after.trim_start().starts_with('(') {
                return true;
            }
            from = at + name.len();
        }
        false
    }

    /// Every identifier the harness constructs with `new`. Array and bytes
    /// allocations (`new address[](n)`, `new bytes(32)`) are not calls into
    /// a contract, so the identifier must be followed by `(`.
    fn deployed_contracts(code: &str) -> BTreeSet<String> {
        let mut names: BTreeSet<String> = BTreeSet::new();
        let mut from = 0;
        while let Some(rel) = code[from..].find("new ") {
            let at = from + rel + "new ".len();
            let ident: String = code[at..]
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '$')
                .collect();
            if ident.is_empty() {
                from = at;
                continue;
            }
            from = at + ident.len();
            if code[from..].trim_start().starts_with('(') {
                names.insert(ident);
            }
        }
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The predicate a caller supplies: this project's own contracts.
    fn project(name: &str) -> bool {
        matches!(name, "OETHZapper" | "VaultCore" | "OUSD" | "AbstractHarvester")
    }

    /// A harness shaped like the ones the pipeline actually commits: real
    /// project contracts constructed into an empty EVM behind local mocks.
    const SCAFFOLDED: &str = r#"
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import {OETHZapper} from "contracts/zapper/OETHZapper.sol";

contract HarnessVault {
    function mint(uint256 amount) external { /* mocked accounting */ }
}

contract Test_4 is Test {
    OETHZapper internal zapper;
    function setUp() public {
        zapper = new OETHZapper(address(0), address(0), address(0), address(0));
        targetContract(address(zapper));
    }
}
"#;

    /// The shape a reportable reproduction needs: the deployed system is
    /// forked and its own instances are used, never constructed.
    const DEPLOYED: &str = r#"
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import {OETHZapper} from "contracts/zapper/OETHZapper.sol";

contract Test_9 is Test {
    OETHZapper internal constant ZAPPER = OETHZapper(0x1234);

    function setUp() public {
        vm.createSelectFork(vm.envString("MAINNET_RPC_URL"));
    }

    function test_exploit() public {
        vm.prank(address(0xBEEF));
        ZAPPER.deposit();
    }
}
"#;

    #[test]
    fn flags_harness_that_never_forks() {
        let verdict = PocAdmissibility::of_harness(SCAFFOLDED, project);
        assert!(!verdict.is_admissible());
        assert!(verdict.signals().any(|s| *s == ScaffoldSignal::NoFork));
    }

    #[test]
    fn flags_redeployment_of_a_project_contract() {
        let verdict = PocAdmissibility::of_harness(SCAFFOLDED, project);
        assert!(
            verdict.signals().any(|s| matches!(
                s,
                ScaffoldSignal::Redeployed { contract } if contract == "OETHZapper"
            )),
            "expected the constructed project contract to be named: {:?}",
            verdict.signal_labels()
        );
    }

    #[test]
    fn accepts_a_forked_harness_using_deployed_instances() {
        let verdict = PocAdmissibility::of_harness(DEPLOYED, project);
        assert!(
            verdict.is_admissible(),
            "unexpected signals: {:?}",
            verdict.signal_labels()
        );
    }

    #[test]
    fn fork_cheatcode_in_a_comment_does_not_satisfy_the_requirement() {
        let source = format!("{SCAFFOLDED}\n// could fork with vm.createSelectFork(RPC)\n");
        let verdict = PocAdmissibility::of_harness(&source, project);
        assert!(verdict.signals().any(|s| *s == ScaffoldSignal::NoFork));
    }

    #[test]
    fn block_comment_does_not_satisfy_the_requirement() {
        let source = format!("/* vm.createSelectFork(rpc) */\n{SCAFFOLDED}");
        let verdict = PocAdmissibility::of_harness(&source, project);
        assert!(verdict.signals().any(|s| *s == ScaffoldSignal::NoFork));
    }

    #[test]
    fn flags_state_fabrication_cheatcodes() {
        let source = format!("{SCAFFOLDED}\n// nothing\nfunction setUp() {{ vm.store(a, b, c); }}\n");
        let verdict = PocAdmissibility::of_harness(&source, project);
        assert!(verdict.signals().any(|s| matches!(
            s,
            ScaffoldSignal::FabricatedState { cheatcode } if cheatcode == "vm.store"
        )));
    }

    #[test]
    fn value_helpers_and_reads_are_not_fabrication() {
        let source = format!(
            "{DEPLOYED}\nfunction fund() external {{ vm.deal(address(this), 1 ether); \
             vm.load(address(1), bytes32(0)); }}\n"
        );
        let verdict = PocAdmissibility::of_harness(&source, project);
        assert!(verdict.is_admissible(), "{:?}", verdict.signal_labels());
    }

    #[test]
    fn mock_dependencies_and_array_allocations_are_not_redeployments() {
        let source = format!(
            "{DEPLOYED}\nfunction build() external {{ new MockUSDC(); new HarnessWETH(); \
             new address[](2); new bytes(32); }}\n"
        );
        let verdict = PocAdmissibility::of_harness(&source, project);
        assert!(verdict.is_admissible(), "{:?}", verdict.signal_labels());
    }

    #[test]
    fn clause_names_the_requirement_and_the_signals() {
        let clause = PocAdmissibility::of_harness(SCAFFOLDED, project).clause();
        assert!(clause.contains("Not reportable as reproduced"));
        assert!(clause.contains("OETHZapper"));
        assert!(clause.contains("does not exercise deployed state"));
        // The engine cannot tell "chose not to fork" from "no fork endpoint
        // was configured", so the clause must not assert a deployment it
        // cannot know about.
        assert!(
            clause.contains("Where a deployment is reachable"),
            "the requirement is stated conditionally: {clause}"
        );
        assert!(!clause.contains("does not fork the deployed"), "{clause}");
    }

    /// A run that resolved a block can demand a pinned fork; one that did not
    /// cannot, and the same harness must read differently in the two runs.
    fn pinning_required() -> PocAdmissibilityPolicy {
        PocAdmissibilityPolicy {
            require_pinned_fork: true,
        }
    }

    #[test]
    fn an_unpinned_fork_is_flagged_only_when_the_run_can_pin() {
        assert!(
            PocAdmissibility::of_harness(DEPLOYED, project).is_admissible(),
            "the standing rules accept a fork at head"
        );
        let verdict = PocAdmissibility::of_harness_with(DEPLOYED, project, pinning_required());
        assert!(!verdict.is_admissible());
        assert!(
            verdict
                .signals()
                .any(|signal| *signal == ScaffoldSignal::UnpinnedFork)
        );
        let clause = verdict.clause();
        assert!(clause.contains("does not pin the chain state"), "{clause}");
        assert!(
            !clause.contains("synthetic deployment"),
            "an unpinned fork is not a synthetic deployment: {clause}"
        );
    }

    #[test]
    fn a_fork_pinned_by_argument_or_by_roll_fork_passes() {
        let by_argument = DEPLOYED.replace(
            r#"vm.createSelectFork(vm.envString("MAINNET_RPC_URL"));"#,
            r#"vm.createSelectFork(vm.envString("MAINNET_RPC_URL"), 21_000_000);"#,
        );
        assert!(
            PocAdmissibility::of_harness_with(&by_argument, project, pinning_required())
                .is_admissible()
        );

        let by_roll = DEPLOYED.replace(
            r#"vm.createSelectFork(vm.envString("MAINNET_RPC_URL"));"#,
            "vm.createSelectFork(vm.envString(\"MAINNET_RPC_URL\"));\n        vm.rollFork(21_000_000);",
        );
        assert!(
            PocAdmissibility::of_harness_with(&by_roll, project, pinning_required()).is_admissible()
        );

        // A pinned fork still fails the standing rules when it redeploys.
        let pinned_but_redeployed = format!("{by_argument}\ncontract T {{ function f() external {{ new OETHZapper(); }} }}");
        assert!(
            !PocAdmissibility::of_harness_with(&pinned_but_redeployed, project, pinning_required())
                .is_admissible()
        );
    }

    #[test]
    fn argument_counting_reads_nesting_and_string_literals() {
        assert_eq!(PocAdmissibility::call_arg_count("()"), Some(0));
        assert_eq!(PocAdmissibility::call_arg_count("(x)"), Some(1));
        assert_eq!(PocAdmissibility::call_arg_count("  (url, block)"), Some(2));
        // Nested calls are one argument, not two.
        assert_eq!(PocAdmissibility::call_arg_count("(a, g(b, c))"), Some(2));
        // A comma inside a string literal is not a separator.
        assert_eq!(PocAdmissibility::call_arg_count(r#"("a,b")"#), Some(1));
        // Not a call at all.
        assert_eq!(PocAdmissibility::call_arg_count(" = 3;"), None);
        assert_eq!(PocAdmissibility::call_arg_count(" createSelectFork(url)"), None);
    }

    #[test]
    fn mocking_a_project_contract_is_a_signal_and_mocking_a_dependency_is_not() {
        let on_project = format!(
            "{DEPLOYED}\nfunction m() external {{ vm.mockCall(address(vault), abi.encodeCall(VaultCore.totalAssets, ()), abi.encode(1)); }}\n"
        );
        let verdict = PocAdmissibility::of_harness(&on_project, project);
        assert!(!verdict.is_admissible());
        assert!(verdict.signals().any(|signal| matches!(
            signal,
            ScaffoldSignal::MockedProjectContract { contract } if contract == "VaultCore"
        )));
        assert!(
            verdict.clause().contains("stubs project contract"),
            "{:?}",
            verdict.signal_labels()
        );

        // A mock aimed at a dependency the project does not own is how one
        // keeps a real fork quiet — not a scaffolded proof.
        let on_dependency = format!(
            "{DEPLOYED}\nfunction m() external {{ vm.mockCall(address(0xUSDC), abi.encodeCall(IERC20.balanceOf, (address(this))), abi.encode(1)); }}\n"
        );
        assert!(
            PocAdmissibility::of_harness(&on_dependency, project).is_admissible(),
            "{:?}",
            PocAdmissibility::of_harness(&on_dependency, project).signal_labels()
        );

        // The rule reads code, not prose.
        let in_comment =
            format!("{DEPLOYED}\n// vm.mockCall(address(vault), abi.encode(1));\n");
        assert!(PocAdmissibility::of_harness(&in_comment, project).is_admissible());
    }

    #[test]
    fn deploying_a_project_contract_from_build_output_is_a_redeployment() {
        let source = format!(
            "{DEPLOYED}\nfunction d() external {{ vm.deployCode(\"src/VaultCore.sol:VaultCore\"); }}\n"
        );
        let verdict = PocAdmissibility::of_harness(&source, project);
        assert!(!verdict.is_admissible());
        assert!(verdict.signals().any(|signal| matches!(
            signal,
            ScaffoldSignal::Redeployed { contract } if contract == "VaultCore"
        )));

        // An artifact that is not the project's own stays unflagged.
        let dependency =
            format!("{DEPLOYED}\nfunction d() external {{ vm.deployCode(\"lib/solmate/ERC20.sol:ERC20\"); }}\n");
        assert!(
            PocAdmissibility::of_harness(&dependency, project).is_admissible(),
            "a vendored dependency is not the system under audit"
        );
    }
}
