#!/usr/bin/env python3
"""Admit the Immunefi competition-finding corpus into a standalone Historical KG.

Same writer contract as `ingest_incidents.py` (taxonomy required, enum-valid
categories, one semantic per mechanism), with the source-specific differences:

* **severity comes from the report itself** (`[SC - Critical]`), not from a loss
  band — this corpus is judged, so its own tier is the honest value;
* **root cause comes from the report body** (the reviewer's own prose) when the
  body was fetched;
* **`exploits` points at the published report**, which is where a reader can see
  the reproduction rather than a PoC file path;
* the project is the *competition* the finding ran in (`alchemix`), so a later
  audit of that protocol can be matched against it by name.

Run against a fresh KG created with `learn init-db`.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

# Reuse the class vocabulary and mechanism→semantic mapping: one language for
# both corpora, or the rarity index would be comparing two taxonomies.
from ingest_incidents import SEMANTICS, TAXONOMY_EXISTING, TAXONOMY_NEW  # noqa: E402

DEFAULT_SELECTION = ROOT / "data" / "incidents" / "immunefi_selection.json"
DEFAULT_DB = ROOT / "data" / "immunefi-kg.db"

SEVERITY_BY_TIER = {
    "Critical": "High",
    "High": "High",
    "Medium": "Medium",
    "Low": "Low",
    "Insight": "Low",
    "None": "Medium",
}


def main() -> int:
    args = sys.argv[1:]
    selection_path = Path(args[args.index("--selection") + 1]) if "--selection" in args else DEFAULT_SELECTION
    db_path = Path(args[args.index("--db") + 1]) if "--db" in args else DEFAULT_DB
    if not db_path.exists():
        print(f"[ingest-imm] {db_path} missing — run `knowdit learn init-db` first", file=sys.stderr)
        return 2

    selection = json.loads(selection_path.read_text())
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    def next_id(table: str) -> int:
        return cursor.execute(f"select coalesce(max(id), 0) from {table}").fetchone()[0] + 1

    mechanisms = {row["mechanism"] for row in selection}
    semantic_ids: dict[str, int] = {}
    for mechanism in sorted(mechanisms):
        category, name, definition, description = SEMANTICS.get(
            mechanism,
            (
                "Others",
                f"Unclassified Bounty Pattern ({mechanism})",
                "A judged bounty finding whose mechanism the sources label but do not describe enough to generalize.",
                "Kept as its own semantic so it stays visible instead of merging into a neighbour.",
            ),
        )
        semantic_id = next_id("semantic_node")
        cursor.execute(
            "insert into semantic_node (id, name, definition, description, category) values (?, ?, ?, ?, ?)",
            (semantic_id, name, definition, description, category),
        )
        semantic_ids[mechanism] = semantic_id
        cursor.execute("insert into pending_semantic (semantic_node_id) values (?)", (semantic_id,))

    taxonomy_ids: dict[str, int] = {}
    for mechanism in sorted(mechanisms):
        if mechanism in TAXONOMY_EXISTING:
            taxonomy_ids[mechanism] = TAXONOMY_EXISTING[mechanism]
            continue
        category, name = TAXONOMY_NEW.get(
            mechanism, ("Arithmetic", f"Unclassified Bounty Class ({mechanism})")
        )
        existing = cursor.execute("select id from finding_category where name = ?", (name,)).fetchone()
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

    categories = {name: ident for ident, name in cursor.execute("select id, name from category")}
    findings = 0
    for finding in selection:
        slug = f"{finding['project']}-{finding['finding_id']}"
        project_id = next_id("project")
        cursor.execute(
            "insert into project (id, name, status) values (?, ?, 'completed')",
            (project_id, f"{finding['project']} (immunefi competition)"),
        )
        cursor.execute(
            "insert into project_platform (id, project_id, platform_id) values (?, ?, ?)",
            (next_id("project_platform"), project_id, f"immunefi-{slug}"),
        )
        category_name = SEMANTICS.get(finding["mechanism"], ("Others",))[0]
        if category_name in categories:
            cursor.execute(
                "insert or ignore into project_category (project_id, category_id) values (?, ?)",
                (project_id, categories[category_name]),
            )

        semantic_id = semantic_ids[finding["mechanism"]]
        finding_id = next_id("audit_finding")
        title = f"{finding['project']} — {finding['title']}"
        body = (finding.get("body") or "").strip()
        root_cause = body.split("\n")[0][:400] if body else finding["title"]
        description_parts = [
            f"Judged bounty finding: {finding['project']} ({finding['severity']}).",
            f"Report: {finding['url']}",
            f"Mechanism: {finding['mechanism']}.",
        ]
        if body:
            description_parts.append(body[:1200])
        exploit_refs = [f"report: {finding['url']}"]
        if finding.get("has_poc"):
            exploit_refs.append("the report discusses a proof of concept")
        cursor.execute(
            "insert into audit_finding (id, title, severity, root_cause, description, patterns, exploits) "
            "values (?, ?, ?, ?, ?, ?, ?)",
            (
                finding_id,
                title[:300],
                SEVERITY_BY_TIER.get(finding["severity"], "Medium"),
                root_cause,
                " ".join(description_parts),
                f"mechanism: {finding['mechanism']} | tier: {finding['severity']}",
                " ".join(exploit_refs),
            ),
        )
        cursor.execute(
            "insert into audit_finding_category (audit_finding_id, finding_category_id) values (?, ?)",
            (finding_id, taxonomy_ids[finding["mechanism"]]),
        )
        cursor.execute(
            "insert into semantic_finding_link (semantic_node_id, audit_finding_id, strength, evidence) "
            "values (?, ?, 'High', ?)",
            (semantic_id, finding_id, "bounty link: judged instance of this mechanism"),
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

    connection.commit()
    print(f"[ingest-imm] {findings} bounty finding(s), {len(semantic_ids)} semantic(s) → {db_path.name}")
    print(f"[ingest-imm] pending_semantic: {cursor.execute('select count(*) from pending_semantic').fetchone()[0]}")
    connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
