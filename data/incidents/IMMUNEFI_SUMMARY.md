# Immunefi competition findings

Scraped 2026-09-21 from `reports.immunefi.com/llms.txt` (GitBook's own index) plus each finding's markdown.

- findings indexed: **2687** across **45 projects**
- severity mix: None 1447, Insight 438, Critical 242, Low 237, High 176, Medium 147
- proposed for ingestion: **400** (400 with the report body fetched, 307 discussing a PoC)

## Mechanisms proposed (rarest first)

| mechanism | in selection | in corpus |
| --- | --- | --- |
| unclassified | 100 | 1338 |
| share-accounting | 54 | 224 |
| denial-of-service | 38 | 168 |
| governance-attack | 34 | 67 |
| signature-replay | 27 | 127 |
| slippage-mev | 20 | 55 |
| liquidity-pool-logic | 19 | 73 |
| fee-accounting | 19 | 113 |
| lending-logic | 15 | 49 |
| cross-chain-replay | 14 | 64 |
| liquidation-accounting | 13 | 130 |
| access-control | 10 | 49 |
| accounting-drift | 10 | 54 |
| upgrade-storage | 6 | 68 |
| rounding-precision | 6 | 31 |
| initialization | 5 | 24 |
| oracle-manipulation | 4 | 12 |
| permanent-freeze | 2 | 9 |
| donation-attack | 2 | 10 |
| flash-loan | 2 | 2 |

## Severity mix of the selection

| severity | count |
| --- | --- |
| Critical | 195 |
| Medium | 89 |
| High | 70 |
| Low | 44 |
| Insight | 2 |

## Projects with the most selected findings

| project | selected | in corpus |
| --- | --- |
| alchemix | 106 | 181 |
| movement-labs-attackathon | 44 | 123 |
| shardeum-core | 40 | 52 |
| flare-fassets-or-mainnet-audit-comp | 38 | 120 |
| zerolend | 29 | 62 |
| yeet | 28 | 118 |
| stacks-i-attackathon | 13 | 32 |
| shardeum-core-iii | 11 | 36 |
| swaylend_iop | 10 | 25 |
| term-structure-institutional_iop | 9 | 13 |
| shardeum-core-ii | 8 | 15 |
| anvil | 7 | 17 |
| puffer-finance | 7 | 58 |
| iop-paradex | 6 | 34 |
| lombard | 6 | 20 |
