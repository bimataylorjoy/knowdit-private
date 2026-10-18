//! Oracles: what a harness has to measure to prove a specification.
//!
//! Until a specification declared its oracle, every harness proved the same
//! thing — "`post_attack` does not hold" — and that is the wrong proof for most
//! of what a bounty pays for. A theft is shown by someone ending up richer while
//! someone else ends up poorer, measured in one run; a rounding defect by an
//! error bound that is exceeded or that favours the caller; a miscomputation by
//! disagreement with a reference; a freeze by a legitimate call that stops
//! going through. The reference engagement's false positives were exactly the
//! gap between those: states that looked broken with nobody richer.
//!
//! This module turns the declared oracle into harness instructions for the
//! chosen mode, decides the mode itself when the run asks for `auto` (a path
//! that has to be searched goes to the fuzzing handler, a known path to the
//! deterministic PoC), and checks — from the harness source, deterministically —
//! that a violation was produced by a harness that actually measures what the
//! oracle names.

use regex::Regex;

use crate::reflect::admissibility::PocAdmissibility;
use crate::types::{AuditSpecification, OracleKind, OracleSpec, SearchMode};

use super::HarnessKind;
use super::utils::POC_VIOLATION_MARKER;

/// Oracle guidance and checks for one specification.
pub(super) struct OracleGuidance<'a> {
    oracle: Option<&'a OracleSpec>,
}

impl<'a> OracleGuidance<'a> {
    pub(super) fn of(spec: &'a AuditSpecification) -> Self {
        Self {
            oracle: spec.oracle.as_ref(),
        }
    }

    fn kind(&self) -> OracleKind {
        self.oracle
            .map(|oracle| oracle.kind)
            .unwrap_or(OracleKind::StateInvariant)
    }

    /// The mode a spec runs in when the run asked for `auto`: searching (a
    /// `fuzz` oracle, or one whose violation lives in an input space —
    /// differential and bounds) goes to the handler, everything else to the
    /// deterministic PoC. An explicit mode is honoured as given.
    pub(super) fn resolve_mode(&self, requested: HarnessKind) -> HarnessKind {
        if requested != HarnessKind::Auto {
            return requested;
        }
        let searches = self.oracle.is_some_and(|oracle| oracle.search == SearchMode::Fuzz)
            || matches!(self.kind(), OracleKind::Differential | OracleKind::Bounds);
        if searches {
            HarnessKind::Handler
        } else {
            HarnessKind::Poc
        }
    }

    /// The prompt section telling the harness agent how to encode this
    /// oracle in `mode`. Empty for a state invariant, which both mode prompts
    /// already describe.
    pub(super) fn render(&self, mode: HarnessKind) -> String {
        let Some(oracle) = self.oracle else {
            return String::new();
        };
        if oracle.kind == OracleKind::StateInvariant {
            return String::new();
        }
        let handler = mode == HarnessKind::Handler;
        let fires = match handler {
            true => "the invariant assertion fails".to_string(),
            false => format!(
                "the test reverts with a reason starting `{POC_VIOLATION_MARKER}` — write it as \
                 `if (<condition>) revert(\"{POC_VIOLATION_MARKER} <what broke, with the numbers>\");`"
            ),
        };
        let measured = if oracle.measured.is_empty() {
            "(none declared)".to_string()
        } else {
            oracle.measured.join(", ")
        };
        let reference = oracle.reference.as_deref().unwrap_or("(none declared)");
        let tolerance = oracle.tolerance.as_deref().unwrap_or("0");
        let capital = oracle
            .capital
            .as_deref()
            .unwrap_or("what a flash loan of the measured asset on the fork would provide");
        let body = match oracle.kind {
            OracleKind::StateInvariant => String::new(),
            OracleKind::ValueDelta => format!(
                "This specification is proved by **value moving**, not by an odd state.\n\n\
                 1. Pick one actor address and do everything as that actor (`vm.startPrank`).\n\
                 2. Fund it with the starting capital: {capital}. Use `deal` for tokens and \
                 `vm.deal` for native value, and count that capital in the starting value — \
                 profit is measured net of it.\n\
                 3. Before the sequence, record the actor's holdings of every measured asset \
                 ({measured}) with `balanceOf` / `.balance`, and record the victim metric \
                 ({reference}).\n\
                 4. After the sequence, record both again. Convert every asset to one unit of \
                 account (the fork's own oracle or pool price for that asset, or 1:1 when there is \
                 only one asset) — never a price the harness makes up.\n\
                 5. The violation is: actor net value rose by more than {tolerance} **and** the \
                 victim metric fell. {fires}.\n\
                 A run where only the actor's own balance moves, or where the victim metric does \
                 not fall, proves nothing about theft — do not assert on it.{}",
                if handler {
                    "\n\nIn handler mode, keep the start value in a ghost variable set in \
                     `setUp()`, route every handler entry point through the same actor, and \
                     check the condition in an `invariant_…` function so the fuzzer searches \
                     for the sequence."
                } else {
                    ""
                }
            ),
            OracleKind::Differential => format!(
                "This specification is proved by **disagreement with a reference**.\n\n\
                 - Target: {measured}.\n\
                 - Reference: {reference}. If it is a formula, implement it in the test with \
                 wider arithmetic (`Math.mulDiv` or 512-bit style) so the reference itself does \
                 not round; if it is an address, call that deployed version on the fork; if it \
                 is a sibling function, call it with the same inputs and state.\n\
                 - Feed both the same inputs; {}\n\
                 - The violation is `|target − reference| > {tolerance}`. {fires}, naming both \
                 values.",
                if handler {
                    "take the inputs as handler parameters and clamp them with `bound(...)` to \
                     the ranges the protocol actually accepts, so the fuzzer explores them."
                } else {
                    "use the concrete inputs where the specification says they disagree, and \
                     add a `testFuzz_…` variant over the same ranges."
                }
            ),
            OracleKind::Bounds => format!(
                "This specification is proved by **a numeric error beyond its bound**.\n\n\
                 - Functions: {measured}.\n\
                 - Compute the error the claim is about — round-trip loss (in then out), the \
                 difference between two orderings, a monotonicity step — for inputs clamped with \
                 `bound(...)` to realistic ranges, including the edges (1 wei, the first \
                 depositor, a near-empty pool, max supply).\n\
                 - The violation is an error above {tolerance}, or any rounding that ends in the \
                 caller's favour (they get back more than they put in). {fires}, with the input \
                 and the error in the message."
            ),
            OracleKind::Liveness => format!(
                "This specification is proved by **a legitimate call that stops succeeding**.\n\n\
                 - Operations that must keep working: {measured}.\n\
                 - Show first, before the sequence, that a legitimate caller (not the actor) can \
                 perform them with realistic inputs.\n\
                 - After the sequence, perform them again as that caller inside `try … catch` \
                 (or a low-level call whose `success` you check).\n\
                 - The violation is the call failing (or returning materially less) *only \
                 because of* the sequence. {fires}, including the revert reason you caught."
            ),
        };
        format!(
            "\n\n# Oracle for this specification: `{}` ({})\n\n{body}\n",
            oracle.kind.as_str(),
            oracle.search.as_str()
        )
    }

    /// What the harness fails to measure, when a violation it produced cannot
    /// be the oracle's. Read from comment-stripped source, so prose saying
    /// "we check balanceOf" does not count. `None` when the harness encodes
    /// the oracle (or the oracle is a state invariant, which the mode's own
    /// checks cover).
    pub(super) fn gap(&self, harness_source: &str) -> Option<String> {
        let oracle = self.oracle?;
        let code = PocAdmissibility::strip_comments(harness_source);
        let lower = code.to_lowercase();
        match oracle.kind {
            OracleKind::StateInvariant => None,
            OracleKind::ValueDelta => {
                let measurements = lower.matches("balanceof(").count() + lower.matches(".balance").count();
                (measurements < 2).then(|| {
                    "the oracle is `value_delta`, but the harness does not read balances before \
                     and after the sequence (`balanceOf` / `.balance` at least twice) — a \
                     violation that never measured value cannot show anyone ended up richer"
                        .to_string()
                })
            }
            OracleKind::Differential => {
                let reference = oracle.reference.as_deref().unwrap_or("");
                let address = Regex::new(r"0x[0-9a-fA-F]{40}")
                    .expect("static address pattern compiles")
                    .find(reference)
                    .map(|m| m.as_str().to_lowercase());
                match address {
                    Some(address) if !lower.contains(&address) => Some(format!(
                        "the oracle is `differential` against the deployed reference {address}, \
                         which the harness never calls"
                    )),
                    _ => None,
                }
            }
            OracleKind::Bounds => {
                let fuzzed = lower.contains("bound(")
                    || Regex::new(r"function\s+(testfuzz_\w*|test\w*)\s*\(\s*u?int")
                        .expect("static fuzz-signature pattern compiles")
                        .is_match(&lower);
                (!fuzzed).then(|| {
                    "the oracle is `bounds`, but the harness never varies its inputs (no \
                     `bound(...)`, no fuzzed test parameter) — one hand-picked input shows a \
                     number, not that the bound is exceeded"
                        .to_string()
                })
            }
            OracleKind::Liveness => {
                let observes_failure = lower.contains("try ")
                    || lower.contains("(bool success")
                    || lower.contains("(bool ok");
                (!observes_failure).then(|| {
                    "the oracle is `liveness`, but the harness never observes a legitimate call \
                     failing (`try … catch` or a checked low-level call) — a revert of the \
                     harness's own call is not a freeze"
                        .to_string()
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec_with(kind: OracleKind, search: SearchMode) -> AuditSpecification {
        AuditSpecification {
            oracle: Some(OracleSpec {
                kind,
                measured: vec!["WETH".to_string()],
                reference: Some("other holders' redeemable WETH".to_string()),
                tolerance: Some("1e9 wei".to_string()),
                capital: None,
                search,
                note: String::new(),
            }),
            ..Default::default()
        }
    }

    #[test]
    fn auto_mode_searches_only_when_the_oracle_needs_it() {
        let fixed = spec_with(OracleKind::ValueDelta, SearchMode::Fixed);
        let fuzz = spec_with(OracleKind::ValueDelta, SearchMode::Fuzz);
        let bounds = spec_with(OracleKind::Bounds, SearchMode::Fixed);
        let none = AuditSpecification::default();
        assert_eq!(OracleGuidance::of(&fixed).resolve_mode(HarnessKind::Auto), HarnessKind::Poc);
        assert_eq!(OracleGuidance::of(&fuzz).resolve_mode(HarnessKind::Auto), HarnessKind::Handler);
        assert_eq!(OracleGuidance::of(&bounds).resolve_mode(HarnessKind::Auto), HarnessKind::Handler);
        assert_eq!(OracleGuidance::of(&none).resolve_mode(HarnessKind::Auto), HarnessKind::Poc);
        // An explicit mode wins.
        assert_eq!(OracleGuidance::of(&fuzz).resolve_mode(HarnessKind::Poc), HarnessKind::Poc);
    }

    #[test]
    fn value_delta_needs_before_and_after_balances() {
        let spec = spec_with(OracleKind::ValueDelta, SearchMode::Fixed);
        let guidance = OracleGuidance::of(&spec);
        let blind = "function test_x() public { vault.redeem(1); // balanceOf(a) balanceOf(b)\n revert(\"KNOWDIT_POST_ATTACK_REACHED: x\"); }";
        assert!(guidance.gap(blind).is_some());
        let measured = "uint256 a = weth.balanceOf(me); vault.redeem(1); uint256 b = weth.balanceOf(me);";
        assert!(guidance.gap(measured).is_none());
        assert!(guidance.render(HarnessKind::Poc).contains("value_delta"));
    }

    #[test]
    fn bounds_and_liveness_checks() {
        let bounds = spec_with(OracleKind::Bounds, SearchMode::Fixed);
        assert!(OracleGuidance::of(&bounds).gap("function test_a() public { f(1); }").is_some());
        assert!(OracleGuidance::of(&bounds).gap("function testFuzz_a(uint256 x) public { x = bound(x, 1, 10); }").is_none());
        let live = spec_with(OracleKind::Liveness, SearchMode::Fixed);
        assert!(OracleGuidance::of(&live).gap("vault.withdraw(1);").is_some());
        assert!(OracleGuidance::of(&live).gap("try vault.withdraw(1) {} catch {}").is_none());
    }
}
