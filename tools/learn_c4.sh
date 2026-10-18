#!/usr/bin/env bash
# Learn the C4 dataset into the production Historical KG.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="$ROOT/engine"
DATASET="$ROOT/data/datasets"
DB="sqlite://$ROOT/data/historical.db?mode=rwc"

: "${OPENAI_API_URL:?set OPENAI_API_URL (OpenAI-compatible endpoint)}"
: "${OPENAI_API_MODEL:?set OPENAI_API_MODEL}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
: "${OPENAI_BILLING_CAP:=100}"
: "${CONCURRENCY:=4}"

export OPENAI_API_URL OPENAI_API_MODEL OPENAI_API_KEY OPENAI_BILLING_CAP

echo "[learn] db=$DB model=$OPENAI_API_MODEL cap=\$$OPENAI_BILLING_CAP concurrency=$CONCURRENCY"
exec "$ENGINE/target/release/knowdit" learn --database-url "$DB" \
    c4 --c4-dir "$DATASET" --limit 100 --concurrency "$CONCURRENCY" --link
