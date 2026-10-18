#!/usr/bin/env bash
# Global cross-project linker for the production Historical KG.
#
# This is the step that turns "a finding happened in project X" into
# "this finding pattern can fire on semantic S" — the transferable
# semantic↔finding edges the streamloop mapper consumes. Until it runs,
# `finding_link_status` stays empty and every consumer (streamloop)
# refuses the DB as partial.
#
# Run it AFTER `learn c4` finishes, never concurrently: both write the same
# SQLite database.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="$ROOT/engine"
DB="sqlite://$ROOT/data/historical.db?mode=rwc"

: "${OPENAI_API_URL:?set OPENAI_API_URL (OpenAI-compatible endpoint)}"
: "${OPENAI_API_MODEL:?set OPENAI_API_MODEL}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
: "${OPENAI_BILLING_CAP:=100}"
: "${LINK_CONCURRENCY:=4}"

export OPENAI_API_URL OPENAI_API_MODEL OPENAI_API_KEY OPENAI_BILLING_CAP
export LLM_FULL_MODEL_NAME=true
export LLM_REASONING_EFFORT=none
export LLM_RETRY=15

if pgrep -f "knowdit learn" >/dev/null 2>&1; then
  echo "[link] ERROR: a 'knowdit learn' process is still running. Finish it first." >&2
  exit 1
fi

echo "[link] db=$DB model=$OPENAI_API_MODEL concurrency=$LINK_CONCURRENCY cap=\$$OPENAI_BILLING_CAP"
exec "$ENGINE/target/release/knowdit" learn --database-url "$DB" link \
  --link-concurrency "$LINK_CONCURRENCY" \
  --until-all-linked
