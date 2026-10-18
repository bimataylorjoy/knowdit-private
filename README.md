# Knowdit

Knowledge-graph driven smart contract auditing. Knowdit learns reusable
*semantic ↔ vulnerability* links from historical audits, maps a target
project's own semantics onto that knowledge graph, and drives LLM agents to
turn each match into a concrete, falsifiable audit specification for the
target code. Every candidate is held to the rules of the engagement it is
meant for (a bug bounty program or an audit competition) before any further
work is spent on it.

This repository contains the engine, the precomputed historical knowledge
graphs and the tooling used to build and extend them.

## Contents

- [Overview](#overview)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Build](#build)
- [Configuration](#configuration)
- [Quick start](#quick-start)
- [Pipeline](#pipeline)
- [Engagement scope file](#engagement-scope-file)
- [Historical knowledge graph](#historical-knowledge-graph)
- [Tooling](#tooling)
- [Evaluation](#evaluation)
- [Operational notes](#operational-notes)
- [Attribution and license](#attribution-and-license)

## Overview

A run works in four layers:

1. **Historical knowledge.** Past audit reports, bounty disclosures and
   on-chain incidents are distilled into DeFi *semantics* (what a piece of code
   does) linked to the *findings* that broke that behaviour elsewhere.
2. **Project semantics.** The target repository is profiled, its call graph is
   built from source, and an agent extracts the project's own semantics.
3. **Mapping.** The knowledge mapper matches project semantics against the
   historical graph. Each `(project semantic, historical semantic, finding)`
   triple is a *link*, ranked by strength.
4. **Specification.** For each link, a tool-using agent reads the target code
   and either commits an `AuditSpecification` (preconditions, call sequence,
   expected post-state, impact) or abandons the link with a code-grounded
   reason. Optional downstream stages synthesize Foundry harnesses, fuzz them
   and triage the results.

Engagement awareness runs through every layer: payable impacts, the program's
severity table, out-of-scope rules, trusted roles and known issues are loaded
from a scope file and enforced when a specification is committed.

Beyond the upstream framework, this distribution adds:

- engagement scope files with enforceable exclusions, a per-candidate LLM
  scope judge and program severity tables (bug bounty and audit-competition
  triage profiles);
- known-issue import from prior audit reports (Markdown, text, PDF; Code4rena,
  Sherlock, Spearbit, Trail of Bits, Guardian and Dedaub formats, plus
  free-form notes);
- target hypotheses mined from the project's own invariants, an LLM invariant
  miner for structurally novel bugs, and grounding in external dependency
  source;
- a streamed orchestrator with resumable runs, skipping of links concluded on
  an earlier pass, a secondary LLM endpoint, and automatic stop on provider
  quota exhaustion;
- additional knowledge-graph corpora (bounty competition findings, real-world
  incidents) and a recall evaluation against publicly disclosed findings.

## Repository layout

```
.
├── engine/              Rust workspace (the `knowdit` binary and its crates)
│   └── crates/
│       ├── knowdit/             CLI and workflow orchestrators
│       ├── knowdit-audit/       extraction, mapping, specification, scope, harness, reflection
│       ├── knowdit-kg/          historical knowledge-graph learning, linking and merging
│       ├── knowdit-repo-model/  per-project database
│       ├── knowdit-sol/         Solidity tree-sitter / AST extraction
│       └── …
├── data/
│   ├── historical.db            validated historical knowledge graph (Code4rena corpus)
│   ├── immunefi-kg.db           bounty competition findings corpus
│   ├── incidents-kg.db          real-world incident corpus (wave 1)
│   ├── incidents-kg2.db         real-world incident corpus (wave 2)
│   ├── incidents/               scraped source datasets and summaries
│   └── HISTORICAL_KG.md
├── tools/               dataset scrapers, KG ingest and maintenance, evaluation drivers
├── .env.example
└── LICENSE
```

Build outputs, per-project databases, run artifacts, logs and raw datasets are
not tracked (see `.gitignore`).

## Requirements

- Rust (stable, edition 2024) and Cargo
- [Foundry](https://book.getfoundry.sh/) (`forge`, `cast`) on `PATH`; harness
  stages need a `forge` that can build the target repository
- An OpenAI-compatible chat-completions endpoint with tool calling
- Python 3.10+ for the scripts under `tools/`

## Build

```bash
cd engine
cargo build --release -p knowdit
./target/release/knowdit --help
```

Run the test suite with:

```bash
cargo test --workspace
```

## Configuration

Copy `.env.example` to `.env` (or export the variables in your shell):

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_URL` | Base URL of an OpenAI-compatible endpoint |
| `OPENAI_API_MODEL` | Model id |
| `OPENAI_API_KEY` | API key |
| `OPENAI_BILLING_CAP` | Spending cap in USD for one process (`0` disables it) |
| `LLM_REASONING_EFFORT` | Reasoning effort sent with each request; accepted values depend on the provider |
| `LLM_STREAM` | Stream responses; recommended behind proxies with short idle timeouts |
| `LLM_FULL_MODEL_NAME` | Send the model id verbatim, including a `vendor/` prefix |
| `LLM_CONTEXT_WINDOW` | Context window for models the client does not recognise |
| `LLM_MAX_COMPLETION_TOKENS` | Per-response output cap |

A secondary endpoint (for links and mapper workers) is configured with the
same variables under the `SECONDARY_` prefix; see `knowdit workflow streamloop --help`.

## Quick start

1. **Validate the historical knowledge graph.**

   ```bash
   ./engine/target/release/knowdit learn \
     --database-url "sqlite:///$PWD/data/historical.db" validate-db
   ```

2. **Write an engagement scope file** for the target (see
   [Engagement scope file](#engagement-scope-file)), for example
   `target/knowdit-scope.json`.

3. **Run the streamed pipeline** in specification-only mode:

   ```bash
   ./engine/target/release/knowdit workflow streamloop \
     -p my-target:/path/to/target-repo \
     --repo-database runs/my-target/knowdit.sqlite3 \
     --database-url "sqlite:///$PWD/data/historical.db" \
     --scope-config /path/to/target-repo/knowdit-scope.json \
     --output-folder runs/my-target \
     --only-spec \
     --gen-specs-max-links 500 \
     --gen-specs-max-agent-steps 60 \
     --stream-link-concurrency 4 --map-concurrency 4 \
     --stream-resume-skip-abandoned \
     --gen-specs-scope-judge
   ```

   The first run on a repository builds the project profile, call graph,
   semantics and mapping; later runs reuse them and resume where the previous
   run stopped.

4. **Review the output.** Committed specifications are stored in the project
   database (`specification` table) and written under
   `runs/my-target/specifications/`; a summary for every processed link is
   written under `runs/my-target/summaries/`.

## Pipeline

`workflow streamloop` chains the stages below and processes links as a stream,
so downstream work starts as soon as a specification is committed:

| Stage | Command (standalone) | Output |
| --- | --- | --- |
| Project profile | `agentic profile` | domain summary, subsystems, core components |
| Call graph | `static solidity …` | contracts, functions, call edges |
| Semantic extraction | `agentic extract-semantics` | project semantics |
| Knowledge mapping | `agentic map-semantics` | ranked links into the historical graph |
| Specification | `agentic gen-specs` | `AuditSpecification` rows or abandon reasons |
| Harness and fuzzing | `agentic fuzz` | Foundry harnesses and runs |
| Reflection and regeneration | `agentic reflect`, `agentic regen` | triage verdicts, regenerated harnesses |

Related workflows:

- `workflow review-findings` turns validated findings into a client-facing report;
- `workflow import-known-issues` converts prior audit reports into scope known issues;
- `workflow judge-specs` re-runs the scope judge over existing specifications (read-only);
- `workflow record-verdict` records a program's decision on a submitted finding so later runs inherit it;
- `workflow learn` and `workflow merge-kg` admit new projects or whole graphs into the historical knowledge graph;
- `workflow eval-recall` scores runs against publicly disclosed findings.

Useful switches:

- `--only-spec` stops after specification; harness work is left to a reviewer.
- `--gen-specs-target-hypotheses` / `--gen-specs-invariant-llm` add hypotheses
  derived from the target's own invariants in addition to knowledge-graph links.
- `--gen-specs-dependency-source <path>` grounds agents in the real source of
  external contracts the project integrates.
- `--include-lib-sources` treats `lib/` as project source for repositories that
  keep their implementation there.

## Engagement scope file

The scope file (JSON or TOML; `knowdit-scope.json` at the repository root is
picked up automatically) transcribes the program's rules. Links whose project
semantic touches no in-scope asset are dropped before any agent runs, and a
specification that the program's own rules exclude is refused at commit.

```json
{
  "program": "Immunefi: Example Protocol",
  "primacy": "rules",
  "chain_id": 1,
  "assets": [
    { "name": "Vault", "chain_id": 1, "address": "0x…", "value_usd": null }
  ],
  "impacts_in_scope": ["direct_theft_of_funds", "permanent_freeze", "protocol_insolvency", "temporary_freeze"],
  "severity_table": [
    { "tier": "critical", "impacts": ["direct_theft_of_funds", "permanent_freeze", "protocol_insolvency"], "min_value_usd": 50000 },
    { "tier": "high", "impacts": ["temporary_freeze"] }
  ],
  "out_of_scope": [
    { "id": "example:oracle-data", "title": "Incorrect data supplied by third-party oracles", "source": "https://…" }
  ],
  "use_platform_defaults": true,
  "requires_unprivileged_actor": true,
  "allow_privileged_impacts": false,
  "excluded_contracts": ["Mock*", "*Test*"],
  "known_issues": [
    {
      "id": "audit-2025:M-01",
      "title": "Rounding in share conversion favours the depositor",
      "contracts": ["Vault"],
      "match_any": ["convertToShares rounding"],
      "enforce": true,
      "summary": "…"
    }
  ],
  "assumptions": ["The owner multisig and the operator role are trusted."]
}
```

Field notes:

- `primacy`: `impact` weighs any demonstrated impact; `rules` pays only the
  impacts listed in `impacts_in_scope`.
- Impact classes: `direct_theft_of_funds`, `permanent_freeze`,
  `temporary_freeze`, `protocol_insolvency`, `protocol_dos`, `griefing`.
- `severity_table` maps impact classes onto the program's own tiers, with an
  optional value-at-risk floor per tier.
- `known_issues` entries with `enforce: true` refuse a specification whose
  contracts match and whose text contains one of the `match_any` needles; keep
  needles specific to the defect. Entries with `enforce: false` are context for
  the agents and the novelty judge.
- `assumptions` are rendered into every agent prompt.

## Historical knowledge graph

| Database | Corpus | Projects | Findings |
| --- | --- | ---: | ---: |
| `data/historical.db` | Code4rena audit reports | 66 | 1,887 |
| `data/immunefi-kg.db` | bounty competition findings | 800 | 800 |
| `data/incidents-kg.db` | on-chain incidents, wave 1 | 250 | 250 |
| `data/incidents-kg2.db` | on-chain incidents, wave 2 | 371 | 371 |

`historical.db` is the validated baseline: every finding is globally linked.
The additional corpora are standalone graphs that can be merged into a copy of
the baseline with `workflow merge-kg`. Treat the databases as versioned binary
artifacts and always merge into a copy. Details are in
[`data/HISTORICAL_KG.md`](data/HISTORICAL_KG.md).

## Tooling

| Script | Purpose |
| --- | --- |
| `tools/scraper_c4.py` | Build a Code4rena dataset (reports, findings, source repositories) for `learn c4` |
| `tools/scraper_immunefi.py` | Scrape published Immunefi competition findings |
| `tools/scraper_incidents.py` | Build the incident dataset from DeFiHackLabs and DefiLlama |
| `tools/ingest_immunefi.py`, `tools/ingest_incidents.py` | Admit a scraped corpus into a standalone knowledge graph |
| `tools/learn_c4.sh` | Learn the Code4rena dataset into the historical graph |
| `tools/link_kg.sh` | Run the global cross-project linker until every finding is linked |
| `tools/kg_fold_pending.py` | Deterministically fold pending semantics into their canonical twins |
| `tools/kg_status.py` | Print coverage and linking status of a knowledge graph |

The shell scripts read the endpoint settings from the environment and refuse to
start without them.

## Evaluation

`tools/eval/` contains the recall case sets (`immunefi-disclosed.json`,
`c4-auditeval-small.json`) and two drivers: `kg_leave_one_out.py` builds a
knowledge graph without the case's own project, and `run_reasoning.py` runs
specification generation only on the links that aim at a disclosed defect. `workflow eval-recall` checks
each disclosed case out at the commit before its fix and scores a run by how
far it got toward the disclosed defect (contract, function, mechanism, proof).

## Operational notes

- **Resume.** Runs are resumable. With `--stream-resume-skip-abandoned`, links
  concluded on an earlier pass are skipped; links that failed on a provider
  error or ran out of agent steps are run again.
- **Provider quota.** The run stops on credit or quota exhaustion, and after
  several consecutive links fail before their first step, instead of draining
  the queue against a dead endpoint.
- **Concurrency.** `--stream-link-concurrency` and `--map-concurrency` set
  parallel links and mapper workers; size them to the endpoint's throughput.
- **Vendored code.** `lib/` and `node_modules/` are treated as vendored. Keep
  dependencies managed elsewhere (for example a `dependencies/` folder) out of
  the analysed tree, or the extractor treats them as project source.
- **Cost.** Every stage reports usage per scope; `OPENAI_BILLING_CAP` stops a
  process at a hard spending limit.

## Attribution and license

Knowdit was created by Ziqiao Kong and released as the research artifact of
[*Knowdit: Knowledge + Audit*](https://arxiv.org/abs/2603.26270)
([upstream repository](https://github.com/abortfuzz/knowdit)). This repository
is a modified version of that work.

Licensed under the GNU General Public License, version 2 only
(`GPL-2.0-only`). See [`LICENSE`](LICENSE).
