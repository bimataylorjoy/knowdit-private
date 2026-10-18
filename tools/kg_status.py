#!/usr/bin/env python3
"""Show the production Historical KG enrichment status.

Reads data/historical.db and prints the metrics that matter for the
"KG quality → novel vuln discovery" loop:

  - projects learned vs dataset size
  - findings by severity and top-level category (signal vs noise)
  - semantics by DeFi category (domain coverage)
  - global link completion (finding_link_status) — the transfer engine
"""
from __future__ import annotations

import glob
import os
import sqlite3
import sys

DB = os.path.join(os.path.dirname(__file__), "..", "data", "historical.db")
DATASET_AUDITS = os.path.join(os.path.dirname(__file__), "..", "data", "datasets", "audits")


def main() -> int:
    con = sqlite3.connect(DB)
    cur = con.cursor()

    def q(sql):
        cur.execute(sql)
        return cur.fetchall()

    dataset_count = len(glob.glob(os.path.join(DATASET_AUDITS, "*.json")))
    learned = q("SELECT COUNT(*) FROM project")[0][0]
    findings = q("SELECT COUNT(*) FROM audit_finding")[0][0]
    semantics = q("SELECT COUNT(*) FROM semantic_node")[0][0]
    linked = q("SELECT COUNT(*) FROM finding_link_status")[0][0]

    print("== KG status ==")
    print(f"  projects:        {learned}/{dataset_count} learned")
    print(f"  audit_findings:  {findings}")
    print(f"  semantic_nodes:  {semantics}")
    print(f"  global linked:   {linked}/{findings} ({linked / max(findings, 1) * 100:.0f}%)")

    print("\n== findings by severity ==")
    for r in q(
        "SELECT severity, COUNT(*) FROM audit_finding GROUP BY severity ORDER BY 2 DESC"
    ):
        print(f"  {r[0]:8s} {r[1]}")

    print("\n== findings by top-level category ==")
    for r in q(
        """SELECT fc.category, COUNT(*) FROM audit_finding af
           JOIN audit_finding_category afc ON afc.audit_finding_id = af.id
           JOIN finding_category fc ON fc.id = afc.finding_category_id
           GROUP BY fc.category ORDER BY 2 DESC"""
    ):
        print(f"  {r[0]:20s} {r[1]}")

    print("\n== semantic nodes by DeFi category ==")
    for r in q(
        "SELECT category, COUNT(*) FROM semantic_node GROUP BY category ORDER BY 2 DESC"
    ):
        print(f"  {r[0]:20s} {r[1]}")

    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
