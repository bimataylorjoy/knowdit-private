#!/usr/bin/env python3
"""Scrape Immunefi's published competition findings — the bounty-judged corpus.

Why this source and not another contest corpus: the knowledge graph is already
built from C4 findings, which is a corpus of *audit-contest* bugs. Immunefi's
competitions are judged by the platform that pays bounties, with the severity
vocabulary a program actually uses (`Critical` / `High` / `Medium` / `Low` /
`Insight`), and their published reports discuss impact in the program's own
terms. That is the prior this pipeline is meant to serve.

`reports.immunefi.com` is a GitBook site, and GitBook publishes an LLM index at
`/llms.txt`: one line per finding with an id, the severity in the title, the bug
title, and a `.md` URL. That index is the inventory; each page's markdown is
fetchable directly, so the body (root cause, impact, PoC discussion) comes along
without HTML scraping.

Output:
    data/incidents/immunefi_findings.json    every indexed finding
    data/incidents/immunefi_selection.json   the subset proposed for the KG
    data/incidents/IMMUNEFI_SUMMARY.md       the numbers
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "incidents"

INDEX_URL = "https://reports.immunefi.com/llms.txt"
BASE = "https://reports.immunefi.com"

MAX_INGEST = int(os.environ.get("IMMUNEFI_MAX_INGEST", "400"))
MAX_BODY_FETCHES = int(os.environ.get("IMMUNEFI_MAX_BODIES", "400"))
MAX_PER_MECHANISM_SHARE = 0.25
BODY_CHARS = 4000

# The title carries the class: `30555 - [SC - Low] Precision loss when …`.
ENTRY = re.compile(r"^- \[(.+?)\]\((https://reports\.immunefi\.com/([^/)]+)/(\d+)-([^)]+?)\.md)\)")
SEVERITY = re.compile(r"\[(?:SC|WEB|BC)\s*-\s*(Critical|High|Medium|Low|Insight|None)\]", re.I)

# `Insight` is Immunefi's "worth noting, not a bug" tier: it never pays, so it
# enters only as context.
PAYABLE = {"critical", "high", "medium"}

# Same mechanism vocabulary as the incident scraper: one language for both
# corpora, so the rarity index can compare them.
MECHANISM_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("oracle-manipulation", ("oracle", "spot price", "price manipulation", "twap", "stale price", "pricefeed")),
    ("reentrancy", ("reentran", "read-only")),
    ("access-control", ("access control", "authorization", "unauthorized", "permissionless", "missing check", "onlyowner", "privilege")),
    ("initialization", ("initiali", "reinitializ", "uninitialized")),
    ("flash-loan", ("flash loan", "flashloan")),
    ("share-accounting", ("share", "exchange rate", "convertto", "mint", "redeem", "inflate", "insolven", "nav")),
    ("donation-attack", ("donation", "donate", "balance inflation", "first deposit", "skim")),
    ("rounding-precision", ("precision loss", "rounding", "truncat", "round down", "round up")),
    ("signature-replay", ("signature", "replay", "permit", "nonce", "ecrecover")),
    ("cross-chain-replay", ("bridge", "cross-chain", "crosschain", "message", "l1", "l2", "sequencer")),
    ("liquidation-accounting", ("liquidation", "collateral", "bad debt", "health factor")),
    ("governance-attack", ("governance", "proposal", "voting", "timelock", "quorum", "vealcx")),
    ("accounting-drift", ("accounting", "invariant", "double count", "drift", "insolven", "totalassets", "totalsupply")),
    ("slippage-mev", ("slippage", "sandwich", "front-run", "frontrun", "mev", "minout", "minamount")),
    ("fee-accounting", ("fee", "commission", "royalt", "interest")),
    ("upgrade-storage", ("upgrade", "proxy", "storage slot", "delegatecall", "implementation", "storage collision")),
    ("liquidity-pool-logic", ("pool", "amm", "liquidity", "swap", "reserve")),
    ("lending-logic", ("borrow", "repay", "debt", "loan", "interest rate")),
    ("token-logic", ("erc20", "erc721", "erc777", "fee-on-transfer", "rebasing", "blacklist", "hook")),
    ("denial-of-service", ("dos", "denial of service", "grief", "revert", "block gas", "freeze")),
    ("permanent-freeze", ("permanent freeze", "permanently frozen", "stuck funds", "locked funds")),
]


def http_get(url: str, timeout: int = 60, retries: int = 3) -> str:
    last: Exception | None = None
    for _ in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "knowdit-immunefi-scraper/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as error:  # noqa: BLE001 - retried
            last = error
    raise RuntimeError(f"GET {url} failed: {last}")


def mechanism_of(text: str) -> str:
    lowered = text.lower()
    best: tuple[str, int] | None = None
    for slug, hints in MECHANISM_HINTS:
        hits = sum(1 for hint in hints if hint in lowered)
        if hits and (best is None or hits > best[1]):
            best = (slug, hits)
    return best[0] if best else "unclassified"


def parse_index(text: str) -> list[dict]:
    findings: list[dict] = []
    for line in text.splitlines():
        if not line.startswith("- ["):
            continue
        match = ENTRY.match(line)
        if not match:
            continue
        title, url, project, finding_id, slug = match.groups()
        # GitBook escapes punctuation in link text: `\\[SC - Critical\\]`. Unescape
        # before matching, or every severity reads as None.
        title = title.replace("\\", "").replace("_", " ").strip()
        severity_match = SEVERITY.search(title)
        severity = severity_match.group(1).capitalize() if severity_match else "None"
        # The human part of the title, without the id/severity scaffolding.
        clean = re.sub(r"^\d+\s*-\s*", "", title)
        clean = SEVERITY.sub("", clean).strip(" -[]")
        findings.append(
            {
                "project": project,
                "finding_id": finding_id,
                "severity": severity,
                "title": clean,
                "slug": slug,
                "url": url,
                "mechanism": mechanism_of(clean),
                "payable": severity.lower() in PAYABLE,
            }
        )
    return findings


def fetch_body(finding: dict) -> dict:
    try:
        body = http_get(finding["url"], timeout=45, retries=2)
    except RuntimeError:
        return {}
    text = re.sub(r"^#.*$", "", body, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # The report's own root-cause/impact language, and any PoC it links.
    return {
        "body": text[:BODY_CHARS],
        "has_poc": bool(re.search(r"poc|proof of concept|forge|foundry|test\.sol|```solidity", text, re.I)),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[immunefi] index …")
    findings = parse_index(http_get(INDEX_URL))
    print(f"[immunefi] {len(findings)} finding(s) indexed")
    if not findings:
        return 1

    projects = Counter(finding["project"] for finding in findings)
    severities = Counter(finding["severity"] for finding in findings)
    mechanism_counts = Counter(finding["mechanism"] for finding in findings)
    total = len(findings)
    for finding in findings:
        count = mechanism_counts[finding["mechanism"]]
        finding["mechanism_rarity"] = round(
            math.log((1 + total) / (1 + count)) / math.log(1 + total), 3
        )
        finding["mechanism_count"] = count

    # Ranking: rarity first (the rare classes are what the C4 corpus lacks),
    # then severity, then whether the report discusses a PoC. A Critical that
    # repeats a class we already have 300 of does not deserve the budget.
    severity_rank = {"Critical": 3.0, "High": 2.0, "Medium": 1.2, "Low": 0.6, "Insight": 0.3, "None": 0.3}
    findings.sort(
        key=lambda finding: (
            finding["mechanism_rarity"]
            * severity_rank.get(finding["severity"], 0.3)
        ),
        reverse=True,
    )

    per_mechanism_cap = max(1, int(MAX_INGEST * MAX_PER_MECHANISM_SHARE))
    selection: list[dict] = []
    taken: Counter = Counter()
    for finding in findings:
        if len(selection) >= MAX_INGEST:
            break
        if taken[finding["mechanism"]] >= per_mechanism_cap:
            continue
        taken[finding["mechanism"]] += 1
        selection.append(finding)

    for finding in selection[:MAX_BODY_FETCHES]:
        finding.update(fetch_body(finding))
    with_body = sum(1 for finding in selection if finding.get("body"))
    with_poc = sum(1 for finding in selection if finding.get("has_poc"))

    (OUT / "immunefi_findings.json").write_text(
        json.dumps(findings, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "immunefi_selection.json").write_text(
        json.dumps(selection, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Immunefi competition findings",
        "",
        f"Scraped {datetime.now(timezone.utc).strftime('%Y-%m-%d')} from "
        "`reports.immunefi.com/llms.txt` (GitBook's own index) plus each finding's markdown.",
        "",
        f"- findings indexed: **{len(findings)}** across **{len(projects)} projects**",
        f"- severity mix: "
        + ", ".join(f"{name} {count}" for name, count in severities.most_common()),
        f"- proposed for ingestion: **{len(selection)}** "
        f"({with_body} with the report body fetched, {with_poc} discussing a PoC)",
        "",
        "## Mechanisms proposed (rarest first)",
        "",
        "| mechanism | in selection | in corpus |",
        "| --- | --- | --- |",
    ]
    for mechanism, count in Counter(f["mechanism"] for f in selection).most_common():
        lines.append(f"| {mechanism} | {count} | {mechanism_counts[mechanism]} |")
    lines += ["", "## Severity mix of the selection", "",
              "| severity | count |", "| --- | --- |"]
    for severity, count in Counter(f["severity"] for f in selection).most_common():
        lines.append(f"| {severity} | {count} |")
    lines += ["", "## Projects with the most selected findings", "",
              "| project | selected | in corpus |", "| --- | --- |"]
    selected_projects = Counter(f["project"] for f in selection)
    for project, count in selected_projects.most_common(15):
        lines.append(f"| {project} | {count} | {projects[project]} |")
    (OUT / "IMMUNEFI_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[immunefi] wrote {OUT/'immunefi_findings.json'}")
    print(f"[immunefi] selection: {len(selection)} ({with_poc} with PoC discussion)")
    print("[immunefi] severities:", Counter(f["severity"] for f in selection).most_common(5))
    print("[immunefi] mechanisms:", Counter(f["mechanism"] for f in selection).most_common(6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
