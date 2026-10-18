#!/usr/bin/env python3
"""Build a high-quality Code4rena audit dataset for the Knowdit `learn c4` pipeline.

Data sources (authoritative):
  - Report list:  https://code4rena.com/reports  (SSR HTML embeds all report entries
    {contest: real C4 id, slug: canonical report slug, title, date})
  - Report body:  https://code4rena.com/reports/<slug>  (markdown-extractable)
  - Source repo:  github.com/code-423n4/<repo> extracted from the report page

Dataset layout (matches knowdit-project/src/c4.rs):
    <out>/audits/<contestId>.json   AuditMeta {contestId,title,slug,startTime,endTime,details}
    <out>/contracts/<contestId>/    Solidity sources (scope filters vendored dirs)
    <out>/reports/<contestId>.md    markdown audit report (findings source for learn)

Quality bar:
  - newest N contests with published reports (fresh KG)
  - must have non-vendored .sol source + a report containing [H-xx]/[M-xx] findings
  - resumable: skips ids already in audits/
"""
from __future__ import annotations

import argparse
import html as ihtml
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

C4_REPORTS_BASE = "https://code4rena.com/reports"
ORG = "code-423n4"
REPORT_ENTRY_RE = re.compile(
    r'\\"contest\\":(\d+),\\"date\\":\\"([0-9-]+)\\",\\"findings\\":\\"([^\\]*)\\",'
    r'\\"slug\\":\\"([^\\]+)\\",\\"sponsor\\":\\"([^\\]*)\\",\\"title\\":\\"([^\\]*)\\"'
)
VENDORED_PARTS = {"lib", "test", "tests", "script", "scripts", "node_modules",
                  "deploy", "deployments", "mocks", "mock", "out", "artifacts",
                  "cache", ".git", "src/vendor", "forge-std"}


def http_get(url: str, timeout: int = 90, retries: int = 4) -> str:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}")
                return r.read().decode("utf-8", errors="ignore")
        except Exception as e:
            last = e
            time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"fetch failed {url}: {last}")


def load_report_entries(cache: Path) -> list[dict]:
    """Parse the reports index once and cache it."""
    if cache.exists():
        return json.loads(cache.read_text())
    html = http_get(f"{C4_REPORTS_BASE}")
    entries = []
    seen = {}
    for m in REPORT_ENTRY_RE.finditer(html):
        e = {"contest": int(m.group(1)), "date": m.group(2),
             "findings": m.group(3), "slug": m.group(4),
             "sponsor": m.group(5), "title": m.group(6)}
        seen.setdefault(e["contest"], e)
    entries = sorted(seen.values(), key=lambda e: e["date"], reverse=True)
    cache.write_text(json.dumps(entries, indent=1))
    print(f"[*] report entries cached: {len(entries)}", flush=True)
    return entries


def repo_url_from_report(html: str) -> str | None:
    """Find the canonical audit repo github.com/code-423n4/<name> in a report page."""
    pats = set(re.findall(rf"https://github\.com/{ORG}/([A-Za-z0-9_.-]+)", html))
    # prefer a bare repo link (not blob/archive paths) and the one matching report slug best
    for name in pats:
        if any(x in name for x in (".png", ".jpg", ".svg", "archive", "releases")):
            continue
        return f"https://github.com/{ORG}/{name}"
    return None


def extract_report_markdown(html_text: str) -> str | None:
    body = html_text[html_text.find("<body"):] if "<body" in html_text else html_text
    body = re.sub(r"<script.*?</script>", "", body, flags=re.S)
    body = re.sub(r"<style.*?</style>", "", body, flags=re.S)
    body = re.sub(r"</(h[1-6]|p|li|pre|div|tr|table|blockquote|ol|ul)>", "\n", body)
    body = re.sub(r"<br\s*/?>", "\n", body)
    body = re.sub(r"<[^>]+>", "", body)
    body = ihtml.unescape(body)
    lines, prev = [], False
    for l in body.split("\n"):
        l = l.rstrip()
        if not l.strip():
            if prev:
                continue
            prev = True
        else:
            prev = False
        lines.append(l)
    text = "\n".join(lines)
    start = -1
    for sc in ["Findings & Analysis Report", "Findings and Analysis Report",
               "High Risk Findings", "# Findings"]:
        idx = text.find(sc)
        if idx >= 0:
            start = idx
            break
    if start < 0:
        return None
    end = text.find("Disclaimer", start)
    if end < 0:
        end = text.find("Related Contests", start)
    seg = text[start:end] if end > start else text[start:]
    if len(seg.strip()) < 1500:
        return None
    if not re.search(r"\[(H|M)-\d+\]", seg):
        return None
    return seg


def has_solidity_source(contract_dir: Path) -> bool:
    if not contract_dir.is_dir():
        return False
    for root, dirs, files in os.walk(contract_dir):
        rel = Path(root).relative_to(contract_dir)
        dirs[:] = [d for d in dirs if d not in VENDORED_PARTS and not d.startswith(".")]
        if any(f.endswith(".sol") for f in files):
            return True
    return False


def clone_repo(url: str, dest: Path) -> bool:
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", "--single-branch", url, str(dest)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=900)
        return True
    except Exception as e:
        print(f"    [clone] failed {url}: {str(e)[:100]}", flush=True)
        shutil.rmtree(dest, ignore_errors=True)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/datasets"))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--entries-cache", type=Path, default=Path("data/datasets/.c4_report_entries.json"))
    args = ap.parse_args()

    out = args.out
    (out / "audits").mkdir(parents=True, exist_ok=True)
    (out / "contracts").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    done_ids = {int(p.stem) for p in (out / "audits").glob("*.json")}
    print(f"[*] already have {len(done_ids)} contests", flush=True)

    entries = load_report_entries(args.entries_cache)
    # newest first (already sorted desc by date)
    good = 0
    for e in entries:
        if good >= args.limit:
            break
        cid = e["contest"]
        if cid in done_ids:
            continue
        slug = e["slug"]
        print(f"[{good+1}/{args.limit}] {slug} (contest={cid}, report {e['date']})", flush=True)

        # 1. report content
        try:
            html_text = http_get(f"{C4_REPORTS_BASE}/{slug}")
        except Exception as ex:
            print(f"    [skip] report fetch: {ex}", flush=True)
            continue
        md = extract_report_markdown(html_text)
        if not md:
            print("    [skip] report has no H/M findings", flush=True)
            continue

        # 2. source repo
        repo = repo_url_from_report(html_text)
        if not repo:
            print("    [skip] no repo url in report page", flush=True)
            continue
        cdir = out / "contracts" / str(cid)
        if not clone_repo(repo, cdir):
            continue
        shutil.rmtree(cdir / ".git", ignore_errors=True)
        if not has_solidity_source(cdir):
            print("    [skip] no solidity source", flush=True)
            shutil.rmtree(cdir, ignore_errors=True)
            continue

        (out / "reports" / f"{cid}.md").write_text(md, encoding="utf-8")
        meta = {"contestId": cid, "title": e["title"], "slug": slug,
                "startTime": None, "endTime": None, "details": f"sponsor: {e['sponsor']}"}
        (out / "audits" / f"{cid}.json").write_text(json.dumps(meta), encoding="utf-8")
        good += 1
        print(f"    [OK] {good}/{args.limit}  report={len(md)//1024}KB", flush=True)

    print(f"\n[done] collected {good} contests", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
