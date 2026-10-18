#!/usr/bin/env python3
"""Targeted reasoning-recall runs: spec generation on the links that aim at a
known defect, and nothing else.

Recall splits into aiming (does the planned queue reach the defect? free, see
`knowdit workflow eval-recall aim`) and reasoning (given links that reach it,
does the agent specify the defect?). This driver measures the second half at a
fraction of a full run's cost: per case it copies the project database (so the
mapper is reused and nothing is written back), runs streamloop in only-spec
mode focused on the case's disclosed locations with a small link budget, and
leaves one output folder per case slug for `eval-recall score`.

Usage:
  run_reasoning.py --cases SET.json --runs RUNS.txt --kg sqlite:///KG.db --out DIR
                   [--links 5] [--parallel 2] [--agents 2] [--case ID ...]

RUNS.txt holds one `<case-id>[,<case-id>...]=<repo database>@<project root>` per
line (the same syntax as `eval-recall aim --run`).
"""

import argparse
import concurrent.futures
import datetime
import json
import os
import shutil
import subprocess
import sys

ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "engine", "target", "release", "knowdit")


def load_runs(path):
    runs = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cases, rest = line.split("=", 1)
        database, root = rest.rsplit("@", 1)
        for case in cases.split(","):
            runs[case.strip()] = (database, root)
    return runs


def focus_of(case):
    focus = []
    for location in case["locations"]:
        if location.get("function"):
            focus.append(f"{location['contract']}.{location['function']}")
        else:
            focus.append(location["contract"])
    return focus


def slug(case_id):
    # Mirrors DisclosedCase::slug so `eval-recall score` finds the folders.
    return "".join(c if (c.isascii() and c.isalnum()) or c in "-_" else "-" for c in case_id)


def run_case(case, database, root, args, log):
    name = slug(case["id"])
    db_dir = os.path.join(args.out, "db")
    copy = os.path.join(db_dir, f"{name}.sqlite3")
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(database + suffix):
            shutil.copyfile(database + suffix, copy + suffix)
    command = [
        ENGINE, "workflow", "streamloop",
        "-p", f"{name}:{root}",
        "--repo-database", copy,
        "--database-url", args.kg,
        "-o", os.path.join(args.out, "results", name),
        "--only-spec", "--gen-specs-regenerate",
        "--gen-specs-max-links", str(args.links),
        "-d", str(args.agents),
    ]
    for focus in focus_of(case):
        command += ["--gen-specs-focus", focus]
    started = datetime.datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(args.out, "logs", f"{name}.log"), "w") as out:
        env = dict(os.environ, RUST_LOG=os.environ.get("RUST_LOG", "info,sqlx=warn"))
        code = subprocess.call(["timeout", str(args.timeout)] + command, stdout=out, stderr=subprocess.STDOUT, env=env)
    log(f"{started} {case['id']} exit={code}")
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--kg", required=True, help="KG database url, as streamloop --database-url takes it")
    parser.add_argument("--out", required=True)
    parser.add_argument("--links", type=int, default=5, help="links per case (after focusing)")
    parser.add_argument("--parallel", type=int, default=2, help="cases at once")
    parser.add_argument("--agents", type=int, default=2, help="streamloop -d per case")
    parser.add_argument("--timeout", type=int, default=7200, help="seconds per case")
    parser.add_argument("--case", action="append", default=[], help="only these case ids")
    args = parser.parse_args()

    cases = json.load(open(args.cases))["cases"]
    runs = load_runs(args.runs)
    selected = [c for c in cases if (not args.case or c["id"] in args.case)]
    missing = [c["id"] for c in selected if c["id"] not in runs]
    if missing:
        sys.exit(f"no --runs entry for: {', '.join(missing)}")
    for sub in ("db", "results", "logs"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    driver = open(os.path.join(args.out, "logs", "driver.log"), "a")

    def log(line):
        driver.write(line + "\n")
        driver.flush()
        print(line, flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(run_case, c, *runs[c["id"]], args, log) for c in selected]
        failed = sum(1 for f in futures if f.result() != 0)
    log(f"DONE {len(selected)} case(s), {failed} failed")


if __name__ == "__main__":
    main()
