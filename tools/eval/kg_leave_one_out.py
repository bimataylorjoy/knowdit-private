#!/usr/bin/env python3
"""Copy a Historical KG with one or more projects left out.

Used to evaluate Knowdit on a project the KG was learned from without handing
the pipeline its own answers. For each left-out project P it removes:

  * every audit finding attached to P (with its semantic links, categories,
    link status and merge rows), including canonical findings other projects'
    variants were folded into: their text is P's, so they go too;
  * every semantic node only P carries (with its functions, links and merges);
  * P's rows on semantics it shares with other projects, and the semantic
    function rows on those shared nodes that point at P's own source files;
  * the project row and its platform / category rows.

The result is strictly weaker than the original KG. Other projects' variants
merged into a removed finding lose their canonical row, so a left-out
evaluation is conservative, never flattering.

Usage:
  kg_leave_one_out.py SRC_DB DST_DB --project NAME|PLATFORM_ID[=SOURCE_DIR] [--project ...]
SOURCE_DIR (optional) is P's source tree, used to match shared-node functions.
"""

import argparse
import os
import shutil
import sqlite3
import sys


def ids(db, sql, *args):
    return [row[0] for row in db.execute(sql, args).fetchall()]


def in_list(values):
    return "(" + ",".join(str(int(v)) for v in values) + ")" if values else "(NULL)"


def source_paths(root):
    paths = set()
    if not root:
        return paths
    for base, _, files in os.walk(root):
        if "node_modules" in base or "/lib/" in base + "/":
            continue
        for name in files:
            if name.endswith(".sol"):
                rel = os.path.relpath(os.path.join(base, name), root)
                paths.add(rel)
    return paths


def leave_out(db, name, root):
    # A platform id (e.g. c4-456) disambiguates projects sharing a name.
    project = ids(db, "select project_id from project_platform where platform_id = ?", name) or ids(
        db, "select id from project where name = ?", name
    )
    if len(project) != 1:
        sys.exit(f"project {name!r}: expected one row, found {len(project)}")
    pid = project[0]
    report = {"project": name, "id": pid}

    findings = ids(db, "select audit_finding_id from project_finding where project_id = ?", pid)
    f = in_list(findings)
    report["findings"] = len(findings)
    report["finding_links"] = db.execute(
        f"delete from semantic_finding_link where audit_finding_id in {f}").rowcount
    db.execute(f"delete from audit_finding_category where audit_finding_id in {f}")
    db.execute(f"delete from finding_link_status where audit_finding_id in {f}")
    report["finding_merges"] = db.execute(
        f"delete from finding_merge where from_finding_id in {f} or to_finding_id in {f}").rowcount
    db.execute(f"delete from project_finding where audit_finding_id in {f}")
    db.execute(f"delete from audit_finding where id in {f}")

    semantics = ids(db, "select semantic_node_id from project_semantic where project_id = ?", pid)
    only = [
        s for s in semantics
        if not ids(db, "select project_id from project_semantic where semantic_node_id = ? and project_id != ?", s, pid)
    ]
    shared = [s for s in semantics if s not in only]
    s = in_list(only)
    report["semantics_removed"] = len(only)
    report["semantics_shared"] = len(shared)
    report["semantic_links"] = db.execute(
        f"delete from semantic_finding_link where semantic_node_id in {s}").rowcount
    db.execute(f"delete from semantic_function where semantic_node_id in {s}")
    db.execute(f"delete from semantic_merge where from_semantic_id in {s} or to_semantic_id in {s}")
    db.execute(f"delete from pending_semantic where semantic_node_id in {s}")
    db.execute(f"delete from project_semantic where semantic_node_id in {s}")
    db.execute(f"delete from semantic_node where id in {s}")

    paths = source_paths(root)
    removed_functions = 0
    if paths and shared:
        for row_id, path in db.execute(
            f"select id, contract_path from semantic_function where semantic_node_id in {in_list(shared)}"
        ).fetchall():
            if path in paths:
                db.execute("delete from semantic_function where id = ?", (row_id,))
                removed_functions += 1
    report["shared_functions_removed"] = removed_functions

    db.execute("delete from project_semantic where project_id = ?", (pid,))
    db.execute("delete from project_category where project_id = ?", (pid,))
    db.execute("delete from project_platform where project_id = ?", (pid,))
    db.execute("delete from project where id = ?", (pid,))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--project", action="append", required=True, help="NAME or platform id (c4-456), optionally =SOURCE_DIR")
    args = parser.parse_args()

    if os.path.exists(args.dst):
        sys.exit(f"{args.dst} exists; refusing to overwrite")
    source = sqlite3.connect(args.src)
    target = sqlite3.connect(args.dst)
    source.backup(target)
    source.close()

    target.execute("pragma foreign_keys = off")
    for spec in args.project:
        name, _, root = spec.partition("=")
        report = leave_out(target, name, root or None)
        print(" ".join(f"{k}={v}" for k, v in report.items()))
    target.commit()

    dangling = target.execute(
        "select count(*) from semantic_finding_link l left join audit_finding f on f.id = l.audit_finding_id "
        "left join semantic_node n on n.id = l.semantic_node_id where f.id is null or n.id is null"
    ).fetchone()[0]
    if dangling:
        sys.exit(f"{dangling} dangling semantic_finding_link row(s) left")
    target.execute("vacuum")
    target.close()


if __name__ == "__main__":
    main()
