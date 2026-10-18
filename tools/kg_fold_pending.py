#!/usr/bin/env python3
"""Fold the semantics waiting in `pending_semantic` into their canonical twins.

Why this exists
---------------
Phase 2 of the KG learn/merge path is the canonicalisation step: a freshly
extracted semantic is compared against the existing canonicals and, when it is a
variant of one, it is *folded* into it. `pending_semantic` is that queue. The
LLM implementation of phase 2 (merge-kg / workflow learn) persists nothing until a
phase completes, so an interrupted run leaves the queue unprocessed.

This script does the same decision deterministically and only where the evidence
is name identity outside case/punctuation, which needs no model. It mirrors the
engine's own write path (knowdit-kg/src/db.rs, import_merged_canonicals):

  * the leaf node is KEPT -- folding is a pointer, not a delete. The linker
    renders a folded leaf as a variant sub-entry of its canonical
    (link.rs, semantic_merge_appended_notes).
  * one `semantic_merge` row per target: from=leaf, to=canonical, plus the
    leaf's description as the appended note.
  * provenance grows on the canonical: every project that knows the leaf also
    learns the canonical (project_semantic), so the canonical is reachable from
    every project that contributed a variant.
  * the canonical's `name` and `definition` are identity and are never touched.
  * `semantic_finding_link` is NOT rewired: findings keep pointing at the leaf
    they were matched against, and resolution happens through `semantic_merge`.

What it deliberately does NOT do
--------------------------------
It does not consolidate pre-existing name-duplicate groups among the base KG's
own nodes. Those came from independent learn
runs and are reported, not silently rewritten -- collapsing them changes the
candidate rendering for every finding in the KG and deserves its own measured
pass.

Usage:
    python3 tools/kg_fold_pending.py --db data/historical-merged.db            # dry run
    python3 tools/kg_fold_pending.py --db data/historical-merged.db --apply
"""
import argparse
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path


def norm(text: str) -> str:
    """Identity key: case, punctuation and spacing are not identity."""
    text = re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--apply", action="store_true", help="write; without it, dry run")
    ap.add_argument("--backup-dir", type=Path, default=Path("data/backups"))
    args = ap.parse_args()

    if not args.db.exists():
        print(f"db not found: {args.db}", file=sys.stderr)
        return 1

    con = sqlite3.connect(f"file:{args.db}?mode=rw" if args.apply else f"file:{args.db}?mode=ro",
                          uri=True)
    con.execute("pragma foreign_keys=on")

    nodes = con.execute("select id, name, definition, description, category from semantic_node").fetchall()
    leaf_ids = [r[0] for r in con.execute("select semantic_node_id from pending_semantic order by semantic_node_id")]
    if not leaf_ids:
        print("pending_semantic is empty; nothing to fold")
        return 0

    by_key: dict[str, list[int]] = {}
    for nid, name, *_ in nodes:
        by_key.setdefault(norm(name), []).append(nid)

    # A canonical is chosen, never invented, and exactly ONE per identity group:
    # picking per-variant instead would pair two variants into each other
    # (A→B while B→A), which is a cycle rather than a fold. The member the corpus
    # actually uses most is the canonical -- it is the one the linker already
    # resolves candidates against, so it carries the most edges and the richest
    # rendering. Ties go to the lowest id so the result never depends on row order.
    def link_count(nid: int) -> int:
        return con.execute("select count(*) from semantic_finding_link where semantic_node_id=?",
                           (nid,)).fetchone()[0]

    meta = {nid: (name, definition, description, category) for nid, name, definition, description, category in nodes}
    pending_set = set(leaf_ids)

    folds: list[tuple[int, int, str]] = []   # leaf, canonical, appended note
    accepted: list[int] = []                 # leaves alone in their group: canonical themselves
    report: list[tuple[str, int, int, int]] = []  # key, group size, canonical, links
    for key, members in sorted(by_key.items()):
        mine = [m for m in members if m in pending_set]
        if not mine:
            continue
        canonical = max(members, key=lambda m: (link_count(m), -m))
        if len(members) > 1:
            report.append((key, len(members), canonical, link_count(canonical)))
        for leaf in mine:
            if leaf == canonical:
                accepted.append(leaf)
            else:
                folds.append((leaf, canonical, (meta[leaf][3] or meta[leaf][1] or "").strip()))

    # A fold target that is itself a source would nest variants indefinitely and
    # make resolution order-dependent; the group-canonical rule makes it
    # impossible, and this asserts it stayed that way.
    sources = {leaf for leaf, _, _ in folds}
    assert not (sources & {c for _, c, _ in folds}), "cycle in fold plan"

    print(f"queue: {len(leaf_ids)} pending semantics")
    print(f"  fold into a group canonical : {len(folds)}")
    print(f"  stay canonical themselves   : {len(accepted)}")
    print(f"\nidentity groups touched ({len(report)}) -- one canonical each:")
    for key, size, canonical, links in report:
        folded = [f"sem-{leaf}" for leaf, c, _ in folds if c == canonical]
        print(f"    '{key[:58]}' size={size} canonical=sem-{canonical} (links {links}) ← {', '.join(folded)}")
    for leaf in accepted:
        print(f"    sem-{leaf} '{meta[leaf][0][:60]}' (alone in its group)")

    pre_existing = [v for k, v in by_key.items() if len(v) > 1 and not (set(v) & pending_set)]
    print(f"\nreported, NOT changed: {len(pre_existing)} pre-existing name-duplicate groups among the "
          f"base KG's own nodes ({sum(len(v) - 1 for v in pre_existing)} surplus)")

    if not args.apply:
        print("\ndry run: nothing written (pass --apply to write)")
        return 0

    args.backup_dir.mkdir(parents=True, exist_ok=True)
    backup = args.backup_dir / f"{args.db.stem}.before-fold-{time.strftime('%Y%m%d-%H%M%S')}.db"
    shutil.copy2(args.db, backup)
    print(f"\nbackup: {backup}")

    con.execute("begin immediate")
    try:
        with con:
            for leaf, canonical, note in folds:
                for project_id, in con.execute(
                        "select project_id from project_semantic where semantic_node_id=?", (leaf,)):
                    con.execute(
                        "insert or ignore into project_semantic (project_id, semantic_node_id) values (?,?)",
                        (project_id, canonical))
                exists = con.execute(
                    "select 1 from semantic_merge where from_semantic_id=? and to_semantic_id=?",
                    (leaf, canonical)).fetchone()
                if not exists:
                    con.execute(
                        "insert into semantic_merge (from_semantic_id, to_semantic_id, appended_description) "
                        "values (?,?,?)", (leaf, canonical, note))
            # The queue is a to-do list: a folded leaf is resolved by its merge row,
            # a leaf with no twin is resolved by being its own canonical.
            for leaf in leaf_ids:
                con.execute("delete from pending_semantic where semantic_node_id=?", (leaf,))
    except Exception:
        con.rollback()
        raise

    remaining = con.execute("select count(*) from pending_semantic").fetchone()[0]
    merges = con.execute("select count(*) from semantic_merge").fetchone()[0]
    findings = con.execute("select count(*) from audit_finding").fetchone()[0]
    statuses = con.execute("select count(*) from finding_link_status").fetchone()[0]
    print(f"written: {len(folds)} merge rows | semantic_merge now {merges}")
    print(f"queue now: {remaining}")
    print(f"integrity: {con.execute('pragma integrity_check').fetchone()[0]}")
    print(f"untouched: findings={findings} statuses={statuses} nodes={con.execute('select count(*) from semantic_node').fetchone()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
