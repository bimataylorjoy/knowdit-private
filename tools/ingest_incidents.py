#!/usr/bin/env python3
"""Admit the scraped incident corpus into the Historical KG.

Run against a **copy** of `data/historical.db`: the KG's consumers refuse a
database whose `finding_link_status` is incomplete, and those rows are written
by the global linker, not here. Ingesting into the production file before the
linker has run would take the KG offline for every consumer.

What an incident becomes, field by field:

* **one project per incident** (`project` + `project_platform` with a
  `dhl-<slug>` platform id), so a later audit of a real target can be matched
  against the protocol by name;
* **one semantic per mechanism** (`semantic_node`), reused across incidents —
  this is the pattern-library shape the mapper consumes, and it keeps the node
  count bounded instead of inventing a semantic per incident;
* **one finding per incident** (`audit_finding`) carrying what the sources
  actually said: the hand-written root cause or the source's technique label,
  the loss, the chain, the mechanism, and the PoC path as the exploit reference.
  Nothing is invented to fill a field;
* the in-project links (`semantic_finding_link`), the project joins
  (`project_finding`, `project_semantic`), and — crucially — a `pending_semantic`
  row per new semantic so the global linker claims the work.

The LLM cost of the KG is in the linker and the original extraction. This script
writes rows; it does not call a model, and it does not claim to have read
anything the sources did not say.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "data" / "incidents" / "ingest_selection.json"
# Target KG. Defaults to the scratch copy; pass `--db` to write into a
# standalone incident KG instead (the shape `merge-kg --from` wants, so the
# dedup-merge sees only the additions rather than a copy of production).
DB = Path(sys.argv[sys.argv.index("--db") + 1]) if "--db" in sys.argv else ROOT / "data" / "historical-incidents.db"
SELECTION = (
    Path(sys.argv[sys.argv.index("--selection") + 1])
    if "--selection" in sys.argv
    else ROOT / "data" / "incidents" / "ingest_selection.json"
)
# A later wave re-ranks the whole pool, so it contains the earlier wave. The
# slugs already admitted live in the earlier KG; skip them here or the merge
# re-adds findings the target already has.
SKIP_SLUGS_FROM = (
    Path(sys.argv[sys.argv.index("--skip-slugs-from") + 1])
    if "--skip-slugs-from" in sys.argv
    else None
)

# Loss bands, with one deliberate asymmetry: an incident whose loss the sources
# do not state is graded Medium, not Low. The corpus exists partly to fix the
# severity prior the contest data got wrong (Low outnumbered High there), and a
# silent default of Low would re-teach exactly that inversion. Unknown is its
# own statement, and the finding's description says which it was.
SEVERITY_BANDS = [(10_000_000, "High"), (0, "Medium")]

# mechanism -> (category from the existing 14, semantic name, definition, description)
SEMANTICS: dict[str, tuple[str, str, str, str]] = {
    "oracle-manipulation": (
        "Dexes",
        "Spot Value Read From a Single Manipulable Source",
        "A protocol quantity (collateral value, share price, swap output, liquidation threshold) is derived from a price that one transaction can move: a pool's spot reserves, a single venue's slot0, or a feed without a manipulation guard.",
        "Incidents reproduce the same shape: the attacker moves the price inside one transaction (often flash-loan funded), then consumes the inflated value through a protocol entry point before the price reverts.",
    ),
    "reentrancy": (
        "Others",
        "State Read Across an External Call Before It Is Committed",
        "A function gives control to an external address while its own accounting is still provisional, so the callee can re-enter and act on the stale view.",
        "Read-only and cross-contract variants dominate production incidents: the re-entered call does not have to change state itself, it only has to observe a state the protocol has not finished writing.",
    ),
    "access-control": (
        "Others",
        "Privileged Effect Reachable Without the Privilege",
        "A state-changing path that is meant to be restricted can be reached by an unprivileged caller — through a missing check, an unprotected initializer, a public helper, or a role that is granted too broadly.",
        "The incident corpus is dominated by this class; in production it is usually a single missing guard on a function that moves value or reconfigures the system.",
    ),
    "initialization": (
        "Others",
        "Initializer Callable After Deployment",
        "An initializer (or a storage flag that gates one) can be called again after the contract is live, letting an attacker re-establish ownership, parameters, or implementation.",
        "The canonical form is an uninitialized or re-initializable proxy, including transient-storage flags that do not survive the way the code assumes.",
    ),
    "share-accounting": (
        "Yield",
        "Claim Value Computed Against a Stale or Manipulable Asset Base",
        "Shares (or debt, or LP units) are priced from an asset total the attacker can influence or that lags reality, so minting and redeeming settle at inconsistent rates.",
        "The attacker either inflates the asset base before minting or deflates it before redeeming; the protocol's own conversion function is the thing that lies.",
    ),
    "donation-attack": (
        "Yield",
        "Balance Donation Changes the Accounting of Existing Holders",
        "A raw token transfer into the contract moves a quantity the accounting reads directly, so the per-holder claim changes without any deposit or withdrawal.",
        "First-depositor and skim-style variants are the same defect: internal ledger and real balance are not the same number, and the code assumes they are.",
    ),
    "flash-loan": (
        "Others",
        "Capital Without Collateral Turns a Slow Attack Into One Transaction",
        "A control that looks sufficient because the attacker would need capital is defeated by uncollateralized same-transaction borrowing.",
        "Incidents use flash loans as an amplifier rather than a root cause: the underlying defect is still a missing check, a manipulable price, or a governance rule reachable with borrowed voting power.",
    ),
    "governance-attack": (
        "Others",
        "Governance Power Acquired Inside One Transaction",
        "Voting power (or proposal execution rights) can be acquired, borrowed, or double-counted within the window in which it is used.",
        "Includes flash-loan voting, proposal execution against a stale snapshot, and timelock bypasses.",
    ),
    "cross-chain-replay": (
        "Cross Chain",
        "Message or Proof Accepted Without Binding It to Its Origin",
        "A bridge or cross-domain handler validates a message against state that the attacker can influence or replay, so a forged or repeated message is treated as authentic.",
        "The failure is usually in what the message is bound to — chain id, sender, nonce, processed-set membership — rather than in the cryptography.",
    ),
    "signature-replay": (
        "Others",
        "Signature Valid More Than Once, or For More Than Intended",
        "A signed authorization is not bound to a nonce, a domain, a chain, or a single use, so it can be submitted again or repurposed.",
        "Malleability and missing per-signer nonce tracking are the two common forms; both end with the same effect as a stolen key.",
    ),
    "liquidation-accounting": (
        "Lending",
        "Liquidation Settles Against the Wrong Collateral Value",
        "The check that decides whether a position may be liquidated uses a quantity that does not match the debt or the collateral actually at stake.",
        "Outcomes range from bad debt the protocol absorbs to positions that cannot be liquidated at all.",
    ),
    "accounting-drift": (
        "Others",
        "Two Internal Totals That Are Supposed to Agree Stop Agreeing",
        "A tracked total (assets, liabilities, supply, reserves) diverges from the quantity it mirrors because one of the two update paths is incomplete, skipped, or double counted.",
        "This is the class the invariant miner targets directly, and production incidents are where the drifted total becomes withdrawable value.",
    ),
    "rounding-precision": (
        "Others",
        "Integer Rounding Favouring the Caller",
        "A division or truncation lets the caller round a value in their own favour, repeatedly or at scale.",
        "Small per-operation effects compound into real losses when the operation is repeatable or sized by an attacker-controlled amount.",
    ),
    "slippage-mev": (
        "Dexes",
        "Trade Submitted Without a Binding Bound on Its Outcome",
        "An execution path passes no minimum output, or a minimum the caller controls, so the transaction can be settled at any price the next block allows.",
        "Front-running and sandwiching are the exploitation; the defect is the missing binding of the operation to its quoted outcome.",
    ),
    "fee-accounting": (
        "Yield",
        "Fees Taken From a Base That Does Not Match the Yield",
        "A fee is computed on a quantity that includes principal, unrealized value, or another party's share.",
        "The result is a slow leak of one cohort's value to another, or fees that exceed the yield they are meant to come from.",
    ),
    "upgrade-storage": (
        "Others",
        "Upgrade Path Writes Where It Should Read",
        "A proxy upgrade, storage-slot assignment, or delegatecall target can be influenced so the implementation or a critical slot is replaced.",
        "Also covers storage collisions and reinitialization through the upgrade entry point.",
    ),
    "liquidity-pool-logic": (
        "Dexes",
        "Pool Accounting That Does Not Match the Real Balances",
        "Reserves, invariants, or accumulated fees tracked by the pool diverge from the tokens the pool actually holds, so swaps settle against a fiction.",
        "Includes invariant mis-updates, fee growth applied to the wrong side, and reserve syncs that can be triggered by an outsider.",
    ),
    "lending-logic": (
        "Lending",
        "Debt or Interest Accrual That the Position Cannot Repay",
        "Borrow, repay, or index accrual lets a position take on debt that is not reflected in the collateral check.",
        "The consequence is either free borrowing or a position the protocol must socialize.",
    ),
    "token-logic": (
        "Others",
        "Assumption About a Token That the Token Does Not Honour",
        "The protocol assumes a token behaves like a plain ERC20 while that token rebases, takes a fee, or calls back on transfer.",
        "The exploit is in the gap between the assumed transfer semantics and the actual ones.",
    ),
    "permit-signature": (
        "Others",
        "Permit or ERC1271 Path That Does Not Match the Real Signer",
        "A meta-transaction or signature-validation path accepts an authorization the signer did not intend.",
        "Covers chain-id reuse, contract-signature path confusion, and validation against the wrong digest.",
    ),
}


# The KG enforces one taxonomy row per finding (`audit_finding_category`), so a
# writer has to assign a class. The existing 45 are SWC-style and cover the
# classic classes; where an incident class has no honest equivalent there, a new
# row is added under the source's own label rather than forcing a wrong match.
# `(mechanism, existing finding_category id)` when there is a real match.
TAXONOMY_EXISTING: dict[str, int] = {
    "access-control": 3,          # Caller Not Checked
    "initialization": 8,          # Initializing Method without Permission Check
    "reentrancy": 38,             # Reentrancy Vulnerability without ETH Transfer
    "rounding-precision": 19,     # Inappropriate Integer Division before Multiplication
    "signature-replay": 26,       # Lack of Proper Signature Verification
    "permit-signature": 26,
    "slippage-mev": 24,           # Transaction Order Dependency
    "upgrade-storage": 15,        # Unsafe Delegatecall
}

# `mechanism -> (category, name)` for the classes the 45 do not name.
#
# `category` is NOT free text: it validates against the KG's
# `VulnerabilityCategory` enum (Access Control, Arithmetic, Block Manipulation,
# Cryptographic, Denial of Services, Reentrancy, Storage & Memory), and the
# linker rejects the whole database over an unknown value. So the source's own
# label goes in `name`, which is free text, and the enum slot takes the closest
# honest fit — the mismatch between a 7-value enum and how incidents are
# actually classified is a fact about the schema, not something to paper over
# with a wrong label.
TAXONOMY_NEW: dict[str, tuple[str, str]] = {
    "oracle-manipulation": ("Block Manipulation", "Price Read From a Manipulable Source"),
    "share-accounting": ("Arithmetic", "Incorrect Share Accounting"),
    "donation-attack": ("Arithmetic", "Balance Donation Changes Accounting"),
    "flash-loan": ("Access Control", "Flash-Loan-Amplified Attack Path"),
    "governance-attack": ("Access Control", "Governance Power Acquired Within One Transaction"),
    "cross-chain-replay": ("Cryptographic", "Message or Proof Not Bound to Its Origin"),
    "liquidation-accounting": ("Arithmetic", "Liquidation or Debt Accounting Mismatch"),
    "lending-logic": ("Arithmetic", "Debt or Interest Accrual Mismatch"),
    "accounting-drift": ("Arithmetic", "Divergent Internal Totals"),
    "fee-accounting": ("Arithmetic", "Fee Computed on the Wrong Base"),
    "liquidity-pool-logic": ("Arithmetic", "Pool Accounting Diverges From Real Balances"),
    "token-logic": ("Access Control", "Token Semantics Not Honoured"),
}


def severity_for(amount: int) -> str:
    for floor, severity in SEVERITY_BANDS:
        if amount >= floor:
            return severity
    return "Medium"


def main() -> int:
    if not DB.exists():
        print(f"[ingest] {DB} does not exist — copy the production KG there first:", file=sys.stderr)
        print(f"         cp data/historical.db {DB}", file=sys.stderr)
        return 2
    selection = json.loads(SELECTION.read_text())
    if SKIP_SLUGS_FROM is not None and SKIP_SLUGS_FROM.exists():
        earlier = sqlite3.connect(f"file:{SKIP_SLUGS_FROM}?mode=ro", uri=True)
        skip = {
            row[0].removeprefix("dhl-")
            for row in earlier.execute(
                "select platform_id from project_platform where platform_id like 'dhl-%'"
            )
        }
        before = len(selection)
        selection = [incident for incident in selection if incident["slug"] not in skip]
        print(f"[ingest] skipping {before - len(selection)} incident(s) already in the earlier wave")
    connection = sqlite3.connect(DB)
    cursor = connection.cursor()

    def next_id(table: str) -> int:
        return (cursor.execute(f"select coalesce(max(id), 0) from {table}").fetchone()[0]) + 1

    # Idempotent: a second run replaces this script's own rows.
    inserted_slugs = {
        row[0]
        for row in cursor.execute(
            "select platform_id from project_platform where platform_id like 'dhl-%'"
        )
    }
    if inserted_slugs:
        print(f"[ingest] {len(inserted_slugs)} dhl project(s) already present; replacing them")
        cursor.execute(
            "delete from project_finding where project_id in "
            "(select project_id from project_platform where platform_id like 'dhl-%')"
        )
        cursor.execute(
            "delete from project_semantic where project_id in "
            "(select project_id from project_platform where platform_id like 'dhl-%')"
        )
        cursor.execute(
            "delete from project where id in "
            "(select project_id from project_platform where platform_id like 'dhl-%')"
        )
        cursor.execute("delete from project_platform where platform_id like 'dhl-%'")

    categories = {name: ident for ident, name in cursor.execute("select id, name from category")}
    semantic_ids: dict[str, int] = {}
    semantic_link_ids: dict[str, tuple[int, int]] = {}

    mechanisms_used = {row["mechanism"] for row in selection}
    for mechanism in sorted(mechanisms_used):
        category, name, definition, description = SEMANTICS.get(
            mechanism,
            (
                "Others",
                f"Unclassified Incident Pattern ({mechanism})",
                "A production incident whose mechanism the sources label but do not describe in enough detail to generalize.",
                "Kept as its own semantic so it is visible rather than silently merged into a neighbouring class.",
            ),
        )
        semantic_id = next_id("semantic_node")
        cursor.execute(
            "insert into semantic_node (id, name, definition, description, category) values (?, ?, ?, ?, ?)",
            (semantic_id, name, definition, description, category),
        )
        semantic_ids[mechanism] = semantic_id
        # The retro-link queue: the global linker claims work from here.
        cursor.execute("insert into pending_semantic (semantic_node_id) values (?)", (semantic_id,))

    # Class vocabulary: reuse the existing rows where they fit, add the rest.
    taxonomy_ids: dict[str, int] = {}
    for mechanism in sorted(mechanisms_used):
        if mechanism in TAXONOMY_EXISTING:
            taxonomy_ids[mechanism] = TAXONOMY_EXISTING[mechanism]
            continue
        category, name = TAXONOMY_NEW.get(
            mechanism, ("Arithmetic", f"Unclassified Incident Class ({mechanism})")
        )
        existing = cursor.execute(
            "select id from finding_category where name = ?", (name,)
        ).fetchone()
        if existing:
            taxonomy_ids[mechanism] = existing[0]
            continue
        taxonomy_id = next_id("finding_category")
        definition = SEMANTICS.get(mechanism, ("", "", "", ""))[2]
        cursor.execute(
            "insert into finding_category (id, category, name, description) values (?, ?, ?, ?)",
            (taxonomy_id, category, name, definition[:400]),
        )
        taxonomy_ids[mechanism] = taxonomy_id
    new_categories = sum(
        1 for mechanism in mechanisms_used if mechanism not in TAXONOMY_EXISTING
    )
    print(f"[ingest] taxonomy: {len(taxonomy_ids)} class(es) assigned "
          f"({new_categories} of them new rows)")

    project_ids: dict[str, int] = {}
    findings = 0
    for incident in selection:
        slug = incident["slug"]
        project_id = next_id("project")
        cursor.execute(
            "insert into project (id, name, status) values (?, ?, 'completed')",
            (project_id, f"{incident['name']} (incident)"),
        )
        cursor.execute(
            "insert into project_platform (id, project_id, platform_id) values (?, ?, ?)",
            (next_id("project_platform"), project_id, f"dhl-{slug}"),
        )
        project_ids[slug] = project_id

        semantic_id = semantic_ids[incident["mechanism"]]
        category = SEMANTICS.get(incident["mechanism"], ("Others",))[0]
        if category in categories:
            cursor.execute(
                "insert or ignore into project_category (project_id, category_id) values (?, ?)",
                (project_id, categories[category]),
            )

        finding_id = next_id("audit_finding")
        title = f"{incident['name']} — {(incident.get('root_cause') or incident.get('technique') or incident['mechanism'])}"
        root_cause = (
            incident.get("root_cause")
            or incident.get("technique")
            or f"{incident['mechanism']} (labelled by the incident sources)"
        )
        facts = [
            f"Incident: {incident['name']} ({incident['month']}).",
            f"Loss: ${incident['amount_usd']:,}." if incident["amount_usd"] else "Loss: not stated by the sources.",
            f"Chain: {incident.get('chain')}." if incident.get("chain") else None,
            f"Source classification: {incident.get('classification')} / {incident.get('technique')}."
            if incident.get("classification")
            else None,
            f"Reference: {incident['source_url']}" if incident.get("source_url") else None,
        ]
        description = " ".join(fact for fact in facts if fact)
        patterns = " | ".join(
            filter(
                None,
                [
                    f"mechanism: {incident['mechanism']}",
                    incident.get("technique"),
                    incident.get("classification"),
                ],
            )
        )
        exploit_refs = [f"PoC: {incident['poc_path']}"]
        if incident.get("poc_contracts"):
            exploit_refs.append("attacked: " + ", ".join(incident["poc_contracts"]))
        if incident.get("poc_comments"):
            exploit_refs.append("|".join(incident["poc_comments"][:3]))
        cursor.execute(
            "insert into audit_finding (id, title, severity, root_cause, description, patterns, exploits) "
            "values (?, ?, ?, ?, ?, ?, ?)",
            (
                finding_id,
                title[:300],
                severity_for(incident["amount_usd"]),
                root_cause,
                description,
                patterns,
                " ".join(exploit_refs),
            ),
        )
        cursor.execute(
            "insert into semantic_finding_link (semantic_node_id, audit_finding_id, strength, evidence) "
            "values (?, ?, 'High', ?)",
            (
                semantic_id,
                finding_id,
                "incident link: the finding is a production instance of this mechanism",
            ),
        )
        cursor.execute(
            "insert into audit_finding_category (audit_finding_id, finding_category_id) values (?, ?)",
            (finding_id, taxonomy_ids[incident["mechanism"]]),
        )
        cursor.execute(
            "insert into project_finding (project_id, audit_finding_id) values (?, ?)",
            (project_id, finding_id),
        )
        cursor.execute(
            "insert into project_semantic (project_id, semantic_node_id) values (?, ?)",
            (project_id, semantic_id),
        )
        findings += 1
        semantic_link_ids[slug] = (semantic_id, finding_id)

    connection.commit()
    print(f"[ingest] {findings} incident finding(s) and {len(semantic_ids)} semantic(s) written to {DB.name}")
    print(f"[ingest] pending_semantic queue now holds "
          f"{cursor.execute('select count(*) from pending_semantic').fetchone()[0]} row(s)")
    print("[ingest] next: run the global linker, which writes finding_link_status:")
    print(f"         knowdit learn --database-url sqlite://{DB} link --link-concurrency 4 --until-all-linked")
    connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
