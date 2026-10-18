#!/usr/bin/env python3
"""Scrape the incident/exploit corpus into a reviewable dataset.

Sources, and why each one:

* **DeFiHackLabs repo tree** (`SunWeb3Sec/DeFiHackLabs`) — the inventory. 900+
  Foundry PoCs under `src/test/<YYYY-MM>/<Name>_exp.sol`, one per real incident,
  each reproducible. This is the falsification recipe the contest corpus never
  had: what a finding looked like as an executable attack.
* **DeFiHackLabs README** — hand-written one-line root causes for the recent
  window ("missing pairDelta update in leverage=0 margin borrow freezes AMM
  quote"), which is the mechanism clause in a human's words.
* **DefiLlama's hacks API** (`api.llama.fi/hacks`) — the economic half: loss in
  USD, chain, and the source's own classification/technique labels.
* **The PoC source itself** — for incidents with no README line, the file still
  names the vulnerable contract and usually carries the exploit's intent in
  comments.

Curation is deliberate:

* key-compromise / frontend / infrastructure incidents are dropped (no code
  defect to find);
* size gates: an incident is kept when a PoC exists or the loss is large;
* the proposed ingest list is ranked by `rarity x amount`, rarity being the
  mechanism's frequency in this corpus — so the KG gains the classes that
  actually cost money rather than 700 more "improper access control" rows.

Outputs (durable, reviewable, no database writes):
    data/incidents/incidents.json           every incident, all fields
    data/incidents/ingest_selection.json    the curated subset for the KG
    data/incidents/SUMMARY.md               the numbers, for a human
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

TREE_URL = "https://api.github.com/repos/SunWeb3Sec/DeFiHackLabs/git/trees/main?recursive=1"
RAW = "https://raw.githubusercontent.com/SunWeb3Sec/DeFiHackLabs/main/"
README_URL = RAW + "README.md"
LLAMA_URL = "https://api.llama.fi/hacks"

BIG_LOSS_USD = 1_000_000
# Cap per wave, overridable so a later wave can take the remainder without
# editing the script: MAX_INGEST=848 python3 tools/scraper_incidents.py
MAX_INGEST = int(os.environ.get("MAX_INGEST", "250"))
MAX_POC_FETCHES = int(os.environ.get("MAX_POC_FETCHES", "900"))
# No single mechanism may take more than this share of one wave. The corpus is
# dominated by a few classes; a wave ranked purely by rarity x amount still
# ends up mostly "improper access control" because that is what exists.
MAX_PER_MECHANISM_SHARE = 0.25

DROP_CLASSIFICATIONS = {
    "key compromise",
    "frontend & infrastructure",
    "phishing",
    "rugpull",
    "exit scam",
}

# Mechanism vocabulary. This is the layer the contest corpus under-represents, so
# the table is written from incident language rather than from judge language:
# the words here are the ones post-mortems use.
MECHANISM_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("oracle-manipulation", ("oracle", "price manipulation", "spot price", "stale price", "twap", "pricefeed", "chainlink")),
    ("reentrancy", ("reentrancy", "reentrant", "read-only reentrancy", "callback reenter")),
    ("access-control", ("access control", "accesscontrol", "authorization", "missing owner", "unprotected", "permissionless", "anyone can", "onlyowner", "privilege")),
    ("initialization", ("initiali", "setup", "transient storage", "uninitialized", "reinitializ")),
    ("flash-loan", ("flash loan", "flashloan", "flash-loan")),
    ("share-accounting", ("share", "exchange rate", "convertto", "vault accounting", "inflate", "nav", "mint", "redeem")),
    ("donation-attack", ("donation", "donate", "balance inflation", "first deposit", "skim")),
    ("rounding-precision", ("rounding", "precision", "truncat", "decimals", "round down", "round up")),
    ("signature-replay", ("signature", "replay", "permit", "ecrecover", "nonce", "malleab")),
    ("cross-chain-replay", ("bridge", "cross-chain", "crosschain", "message replay", "validator", "merkle proof", "l1", "l2")),
    ("liquidation-accounting", ("liquidation", "collateral", "bad debt", "health factor", "liquidat")),
    ("governance-attack", ("governance", "proposal", "voting", "timelock", "quorum")),
    ("accounting-drift", ("accounting", "invariant", "double count", "drift", "delta", "totalassets", "totalsupply", "ledger")),
    ("slippage-mev", ("slippage", "sandwich", "front-run", "frontrun", "mev", "minamount", "minout")),
    ("fee-accounting", ("fee", "commission", "royalt", "interest accrual", "borrow index")),
    ("upgrade-storage", ("upgrade", "proxy", "storage slot", "delegatecall", "implementation")),
    ("liquidity-pool-logic", ("pool", "amm", "liquidity", "swap", "reserve", "curve", "constant product")),
    ("lending-logic", ("borrow", "repay", "debt", "loan", "interest")),
    ("token-logic", ("erc20", "erc777", "transfer hook", "fee-on-transfer", "rebasing", "blacklist")),
    ("permit-signature", ("eip-2612", "permit(", "isvalidsignature", "erc1271")),
]

NAME_CLEAN = re.compile(r"[^a-z0-9]")
POC_NAME = re.compile(r"^src/test/(\d{4}-\d{2})/(.+?)(?:_exp\d?|_exploit|_hack)\.sol$")
TEST_CONTRACT = re.compile(r"contract\s+(Test\d*_\w+|\w*_exp\w*|\w+Exploit\w*)\s+is")
INTERFACE = re.compile(r"interface\s+(I[A-Z]\w+)")
COMMENT_HINT = re.compile(r"//\s*(.{12,180})")


def http_get(url: str, timeout: int = 90, retries: int = 3) -> str:
    last: Exception | None = None
    for _ in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "knowdit-incident-scraper/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as error:  # noqa: BLE001 - retried
            last = error
    raise RuntimeError(f"GET {url} failed: {last}")


def normalise(name: str) -> str:
    return NAME_CLEAN.sub("", name.lower())


def inventory() -> list[dict]:
    """Every PoC in the repo, as `(date, name, path)`."""
    tree = json.loads(http_get(TREE_URL))
    incidents: list[dict] = []
    for entry in tree.get("tree", []):
        path = entry.get("path", "")
        if entry.get("type") != "blob":
            continue
        match = POC_NAME.match(path)
        if not match:
            continue
        month, name = match.groups()
        incidents.append(
            {
                "name": name.replace("_", " "),
                "slug": name,
                "date": f"{month}-01",
                "month": month,
                "poc_path": path,
            }
        )
    incidents.sort(key=lambda incident: incident["month"], reverse=True)
    return incidents


def readme_causes() -> dict[str, str]:
    """`{normalised incident name: one-line root cause}` from the README."""
    text = http_get(README_URL)
    causes: dict[str, str] = {}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            body = stripped[4:].strip()
            if body.lower().startswith("lost"):
                continue
            match = re.match(r"^(?:\d{8}\s+)?(.+?)\s+-\s+(.+)$", body)
            if match:
                current = normalise(match.group(1))
                causes[current] = match.group(2).strip()
        elif current and stripped and not stripped.startswith("#"):
            # A wrapped continuation line only if the heading was cut off.
            if causes[current].endswith(("and", "the", "to", "of", "a", "with", "in")):
                causes[current] = f"{causes[current]} {stripped}"
            else:
                current = None
    return causes


def llama_index() -> list[dict]:
    rows = json.loads(http_get(LLAMA_URL))
    for row in rows:
        row["_key"] = normalise(str(row.get("name") or ""))
    return [row for row in rows if row["_key"]]


def match_llama(incident: dict, rows: list[dict]) -> dict | None:
    """Containment match on the protocol name, because the two sources disagree
    on spelling and on how much of a protocol name to keep."""
    key = normalise(incident["slug"])
    if len(key) < 3:
        return None
    best: tuple[int, dict] | None = None
    for row in rows:
        other = row["_key"]
        if other == key:
            return row
        if key in other or other in key:
            overlap = min(len(key), len(other))
            if best is None or overlap > best[0]:
                best = (overlap, row)
    return best[1] if best and best[0] >= 4 else None


def fetch_poc(path: str) -> dict:
    """The PoC's own evidence: the vulnerable contract it attacks, and the
    comments its author left about why the attack works."""
    try:
        source = http_get(RAW + path, timeout=60, retries=2)
    except RuntimeError:
        return {}
    interfaces = sorted(set(INTERFACE.findall(source)))
    comments = [
        comment.strip()
        for comment in COMMENT_HINT.findall(source)
        if not comment.lower().startswith(("spdx", "forge", "deploy", "tx", "https"))
    ]
    return {
        "poc_contracts": interfaces[:8],
        "poc_comments": comments[:6],
        "poc_bytes": len(source),
    }


def mechanism_of(text: str) -> str:
    lowered = text.lower()
    best: tuple[str, int] | None = None
    for slug, hints in MECHANISM_HINTS:
        hits = sum(1 for hint in hints if hint in lowered)
        if hits and (best is None or hits > best[1]):
            best = (slug, hits)
    return best[0] if best else "unclassified"


def amount_usd(row: dict | None) -> int:
    if not row:
        return 0
    try:
        return int(float(row.get("amount")))
    except (TypeError, ValueError):
        return 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[scrape] inventory (git tree) …")
    inventory_rows = inventory()
    print(f"[scrape] {len(inventory_rows)} PoC(s) in the repo")
    print("[scrape] README root causes …")
    causes = readme_causes()
    print(f"[scrape] {len(causes)} hand-written root cause(s)")
    print("[scrape] DefiLlama hacks index …")
    llama = llama_index()
    print(f"[scrape] {len(llama)} indexed incident(s)")

    joined = 0
    incidents: list[dict] = []
    for row in inventory_rows:
        key = normalise(row["slug"])
        cause = causes.get(key)
        llama_row = match_llama(row, llama)
        if llama_row:
            joined += 1
        technique = str((llama_row or {}).get("technique") or "")
        classification = str((llama_row or {}).get("classification") or "")
        text = " ".join(filter(None, [cause or "", technique, classification, row["slug"]]))
        incidents.append(
            {
                **row,
                "root_cause": cause,
                "amount_usd": amount_usd(llama_row),
                "chain": (llama_row or {}).get("chain"),
                "classification": classification or None,
                "technique": technique or None,
                "target_type": (llama_row or {}).get("targetType"),
                "llama_id": (llama_row or {}).get("defillamaId"),
                "source_url": (llama_row or {}).get("source") or None,
                "mechanism": mechanism_of(text),
                "poc_contracts": [],
                "poc_comments": [],
            }
        )

    frequency = Counter(incident["mechanism"] for incident in incidents)
    total = max(len(incidents), 1)
    for incident in incidents:
        count = frequency[incident["mechanism"]]
        incident["mechanism_rarity"] = round(
            math.log((1 + total) / (1 + count)) / math.log(1 + total), 3
        )
        incident["mechanism_count"] = count
        droppable = (incident["classification"] or "").lower() in DROP_CLASSIFICATIONS
        usable = bool(incident.get("poc_path")) or incident["amount_usd"] >= BIG_LOSS_USD
        incident["ingest_eligible"] = bool(not droppable and usable)

    eligible = [incident for incident in incidents if incident["ingest_eligible"]]
    # What makes an entry usable downstream is not its size alone: the PoC is
    # the falsification recipe and the hand-written root cause is the mechanism
    # clause. A big loss with neither is a headline, not a finding.
    eligible.sort(
        key=lambda incident: incident["mechanism_rarity"]
        * math.log(2 + min(incident["amount_usd"] or 1, 500_000_000))
        * (1.0 + 0.5 * bool(incident.get("poc_path")))
        * (1.0 + 0.3 * bool(incident.get("root_cause"))),
        reverse=True,
    )
    per_mechanism_cap = max(1, int(MAX_INGEST * MAX_PER_MECHANISM_SHARE))
    selection: list[dict] = []
    taken: Counter = Counter()
    for incident in eligible:
        if len(selection) >= MAX_INGEST:
            break
        if taken[incident["mechanism"]] >= per_mechanism_cap:
            continue
        taken[incident["mechanism"]] += 1
        selection.append(incident)

    # Fetch the PoC source for the incidents actually proposed: the vulnerable
    # contract names and the author's comments are what makes an entry usable
    # downstream, and fetching files nobody will ingest is work for nobody.
    for incident in selection[:MAX_POC_FETCHES]:
        incident.update(fetch_poc(incident["poc_path"]))

    (OUT / "incidents.json").write_text(
        json.dumps(incidents, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "ingest_selection.json").write_text(
        json.dumps(selection, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    mechanisms = Counter(incident["mechanism"] for incident in selection)
    with_cause = sum(1 for incident in selection if incident.get("root_cause"))
    with_poc_text = sum(1 for incident in selection if incident.get("poc_bytes"))
    lines = [
        "# Incident corpus",
        "",
        f"Scraped {datetime.now(timezone.utc).strftime('%Y-%m-%d')} from the DeFiHackLabs repo "
        "(inventory + PoCs), its README, and DefiLlama's hacks API.",
        "",
        f"- PoCs in the repository: **{len(incidents)}**",
        f"- joined to a DefiLlama row (loss, chain, technique): **{joined}**",
        f"- with a hand-written root cause (README): **{sum(1 for i in incidents if i.get('root_cause'))}**",
        f"- eligible after the key-compromise / size gate: **{len(eligible)}**",
        f"- proposed for ingestion: **{len(selection)}** "
        f"({with_cause} with a README root cause, {with_poc_text} with PoC source fetched)",
        "",
        "## Mechanisms proposed for ingestion",
        "",
        "| mechanism | count |",
        "| --- | --- |",
    ]
    for mechanism, count in mechanisms.most_common(24):
        lines.append(f"| {mechanism} | {count} |")
    lines += [
        "",
        "## Largest losses in the selection",
        "",
        "| protocol | month | loss USD | mechanism | root cause |",
        "| --- | --- | --- | --- | --- |",
    ]
    for incident in sorted(selection, key=lambda i: i["amount_usd"] or 0, reverse=True)[:15]:
        loss = f"{incident['amount_usd']:,}" if incident["amount_usd"] else "?"
        cause = (incident.get("root_cause") or "—")[:70]
        lines.append(
            f"| {incident['name'][:28]} | {incident['month']} | {loss} | "
            f"{incident['mechanism']} | {cause} |"
        )
    lines += ["", "## Rarest mechanisms in the selection", ""]
    for mechanism, count in sorted(mechanisms.items(), key=lambda pair: pair[1])[:12]:
        lines.append(f"- `{mechanism}` — {count}")
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[scrape] wrote {OUT/'incidents.json'}")
    print(f"[scrape] selection: {len(selection)} of {len(eligible)} eligible")
    print("[scrape] mechanisms:", mechanisms.most_common(6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
