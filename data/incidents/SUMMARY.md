# Incident corpus

Scraped 2026-09-21 from the DeFiHackLabs repo (inventory + PoCs), its README, and DefiLlama's hacks API.

- PoCs in the repository: **870**
- joined to a DefiLlama row (loss, chain, technique): **430**
- with a hand-written root cause (README): **115**
- eligible after the key-compromise / size gate: **848**
- proposed for ingestion: **621** (112 with a README root cause, 621 with PoC source fetched)

## Mechanisms proposed for ingestion

| mechanism | count |
| --- | --- |
| unclassified | 212 |
| access-control | 94 |
| share-accounting | 84 |
| oracle-manipulation | 75 |
| liquidity-pool-logic | 46 |
| reentrancy | 40 |
| cross-chain-replay | 9 |
| slippage-mev | 9 |
| signature-replay | 8 |
| initialization | 8 |
| accounting-drift | 8 |
| flash-loan | 6 |
| governance-attack | 5 |
| lending-logic | 5 |
| liquidation-accounting | 5 |
| fee-accounting | 3 |
| rounding-precision | 1 |
| upgrade-storage | 1 |
| donation-attack | 1 |
| token-logic | 1 |

## Largest losses in the selection

| protocol | month | loss USD | mechanism | root cause |
| --- | --- | --- | --- | --- |
| Bybit | 2025-02 | 1,400,000,000 | unclassified | — |
| Euler | 2023-03 | 197,000,000 | share-accounting | — |
| NomadBridge | 2022-08 | 190,000,000 | cross-chain-replay | — |
| Beanstalk | 2022-04 | 181,000,000 | flash-loan | — |
| Qubit | 2022-01 | 80,000,000 | unclassified | — |
| Uranium | 2021-04 | 57,200,000 | share-accounting | — |
| HedgeyFinance | 2024-04 | 44,700,000 | unclassified | — |
| PAID | 2021-03 | 37,300,000 | unclassified | — |
| Mono | 2021-11 | 31,400,000 | liquidity-pool-logic | — |
| Spartan | 2021-05 | 30,500,000 | share-accounting | — |
| Parity kill | 2017-11 | 30,000,000 | initialization | — |
| Parity first hack | 2017-07 | 30,000,000 | initialization | — |
| Grim | 2021-12 | 30,000,000 | reentrancy | — |
| TransitSwap | 2022-10 | 28,900,000 | liquidity-pool-logic | — |
| Penpiexyzio | 2024-09 | 27,000,000 | reentrancy | — |

## Rarest mechanisms in the selection

- `rounding-precision` — 1
- `upgrade-storage` — 1
- `donation-attack` — 1
- `token-logic` — 1
- `fee-accounting` — 3
- `governance-attack` — 5
- `lending-logic` — 5
- `liquidation-accounting` — 5
- `flash-loan` — 6
- `signature-replay` — 8
- `initialization` — 8
- `accounting-drift` — 8
