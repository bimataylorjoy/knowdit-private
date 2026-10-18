//! Knowledge Mapper agent (v2 — profile-aware, strength-scoring).
//!
//! Step 2 of the agentic auditing workflow. Given a project that
//! already has its extracted DeFi semantics + a generated
//! [`ProjectProfile`] stored in [`RepoDatabase`], this agent
//! fuzzy-matches each extracted semantic against the historical
//! knowledge graph and attaches a per-pair **strength** label
//! (High / Medium / Low) plus a free-form **evidence** rationale.
//! Downstream filters / sorts / budgets by strength to keep the
//! pipeline anchored to project-relevant historicals only.
//!
//! Pipeline:
//! 1. Read extracted semantics + the [`ProjectProfile`] from the repo
//!    database. The profile MUST exist (run `agentic profile` first);
//!    if absent we abort instead of guessing.
//! 2. Collect categories + `Others`, pull canonical historical
//!    semantics in those categories from the KG.
//! 3. Per batch of historicals, ask the LLM to emit one entry per
//!    matched `(extract, historical)` pair with strength + evidence,
//!    grounded in the project profile we inlined into the system
//!    prompt.
//! 4. Aggregate, run pass 2 (cross-domain widening for extracts that matched
//!    nothing in their own categories — opt-in via
//!    [`MapperOptions::widen_unmatched_categories`]) on any extracts
//!    still unmatched, persist semantics + findings + match edges
//!    via [`RepoDatabase::write_semantic_match_results`].

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use color_eyre::eyre::{Result, WrapErr};
use itertools::Itertools;
use knowdit_kg::agents::CanonicalWithChildren;
use knowdit_kg::category::DeFiCategory;
use knowdit_kg::db::HistoricalDatabase;
use knowdit_kg::learn::ExtractedSemantic;
use knowdit_kg_model::db::semantic_node;
use knowdit_repo_model::{
    HistoricalFindingChild, HistoricalLinkedFinding, HistoricalSemanticChild,
    HistoricalSemanticRecord, MatchStrength, ProjectProfile, RepoDatabase, SemanticMatch,
    SemanticMatchSet,
};
use llmy::client::client::LLM;
use serde::Deserialize;
use serde::Serialize;
use std::path::PathBuf;
use tokio::task::JoinSet;

const DEFAULT_BATCH_SIZE: usize = 16;
/// Default cap on raw children rendered per canonical in the matcher
/// prompt. Overridable per-run via [`MapperOptions::max_rendered_raw_children`]
/// (CLI `--map-max-raw-children`).
const DEFAULT_MAX_RENDERED_RAW_CHILDREN: usize = 5;

/// Minimum length of the LLM-emitted `evidence` string per strength
/// tier. The agent is required to emit one entry per candidate pair, so:
/// High/Medium must be substantive (40 chars); Low can be short
/// (15 chars) because the agent is required to emit per-candidate
/// and most candidates will be Low — over-tight evidence floor on Low
/// just inflates output token cost without adding signal.
const MIN_EVIDENCE_LEN_HIGH_MEDIUM: usize = 40;
const MIN_EVIDENCE_LEN_LOW: usize = 15;

/// Keep each mapper response small in both dimensions. `batch_size` controls
/// the historical side; this fixed chunk controls the extracted side.
const EXTRACT_BATCH_SIZE: usize = 8;

#[cfg(test)]
const MAPPER_CHECKPOINT_FILE: &str = ".knowdit-mapper-checkpoint.json";

/// Suffix appended to the repo database's file name to name its checkpoint.
/// The checkpoint belongs to one database: extract ids restart at 1 in every
/// project and the candidate ids come from the shared KG, so two projects
/// whose databases sat in one directory under a shared name could resume
/// each other's batches — and race on the temporary file while doing it.
const MAPPER_CHECKPOINT_SUFFIX: &str = ".mapper-checkpoint.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MapperCheckpoint {
    extracted_ids: Vec<i32>,
    pass1_candidate_ids: Vec<i32>,
    pass1_batch_size: usize,
    pass1_completed_batches: BTreeSet<String>,
    pass1_matches: Vec<SemanticMatch>,
    pass2_candidate_ids: Vec<i32>,
    pass2_batch_size: usize,
    pass2_completed_batches: BTreeSet<String>,
    pass2_matches: Vec<SemanticMatch>,
}

impl MapperCheckpoint {
    fn empty(extracted_ids: &[i32]) -> Self {
        Self {
            extracted_ids: extracted_ids.to_vec(),
            pass1_candidate_ids: Vec::new(),
            pass1_batch_size: 0,
            pass1_completed_batches: BTreeSet::new(),
            pass1_matches: Vec::new(),
            pass2_candidate_ids: Vec::new(),
            pass2_batch_size: 0,
            pass2_completed_batches: BTreeSet::new(),
            pass2_matches: Vec::new(),
        }
    }
}

struct MapperCheckpointStore {
    path: PathBuf,
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;

    fn pending_batch_indices(
        total_batches: usize,
        completed_batches: &BTreeSet<String>,
    ) -> Vec<usize> {
        (0..total_batches)
            .filter(|index| !completed_batches.contains(&index.to_string()))
            .collect()
    }

    #[test]
    fn checkpoint_round_trip_and_input_mismatch() {
        let root = std::env::temp_dir().join(format!(
            "knowdit-mapper-checkpoint-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before unix epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("create checkpoint test directory");

        let store = MapperCheckpointStore {
            path: root.join(MAPPER_CHECKPOINT_FILE),
        };
        let mut checkpoint = MapperCheckpoint::empty(&[1, 2]);
        checkpoint.pass1_candidate_ids = vec![10, 11];
        checkpoint.pass1_batch_size = 1;
        checkpoint.pass1_completed_batches.insert("0".to_string());
        checkpoint.pass1_matches.push(SemanticMatch {
            extract_id: 1,
            historical_id: 10,
            strength: MatchStrength::High,
            evidence: "same concrete mechanism".to_string(),
        });

        store.store(&checkpoint).expect("store checkpoint");
        let loaded = store.load(&[1, 2]).expect("load checkpoint");
        assert_eq!(loaded.pass1_candidate_ids, vec![10, 11]);
        assert!(loaded.pass1_completed_batches.contains("0"));
        assert_eq!(loaded.pass1_matches.len(), 1);
        assert!(store.load(&[1, 3]).is_none());

        store.remove();
        std::fs::remove_dir_all(root).expect("remove checkpoint test directory");
    }

    #[test]
    fn checkpoint_publish_does_not_leave_temporary_file() {
        let root = std::env::temp_dir().join(format!(
            "knowdit-mapper-checkpoint-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before unix epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("create checkpoint test directory");

        let store = MapperCheckpointStore {
            path: root.join(MAPPER_CHECKPOINT_FILE),
        };
        store
            .store(&MapperCheckpoint::empty(&[7]))
            .expect("store checkpoint");
        assert!(store.path.is_file());
        assert!(!root.join(".knowdit-mapper-checkpoint.json.tmp").exists());

        store.remove();
        std::fs::remove_dir_all(root).expect("remove checkpoint test directory");
    }

    #[test]
    fn databases_sharing_a_directory_get_their_own_checkpoint() {
        let dir = std::path::Path::new("/runs/db");
        let first = MapperCheckpointStore::beside(&dir.join("c4-486.sqlite3"));
        let second = MapperCheckpointStore::beside(&dir.join("c4-487.sqlite3"));
        assert_ne!(first.path, second.path);
        assert_eq!(first.path, dir.join(".c4-486.sqlite3.mapper-checkpoint.json"));
    }

    #[test]
    fn resume_only_schedules_unfinished_batches() {
        let completed = BTreeSet::from(["0".to_string(), "2".to_string()]);
        assert_eq!(pending_batch_indices(4, &completed), vec![1, 3]);
    }

    #[test]
    fn changed_batch_shape_invalidates_completed_batch_set() {
        let mut checkpoint = MapperCheckpoint::empty(&[1]);
        checkpoint.pass1_candidate_ids = vec![10, 11];
        checkpoint.pass1_batch_size = 2;
        checkpoint.pass1_completed_batches.insert("0".to_string());

        let current_candidates = vec![10, 12];
        let current_batch_size = 2;
        if checkpoint.pass1_candidate_ids != current_candidates
            || checkpoint.pass1_batch_size != current_batch_size
        {
            checkpoint.pass1_completed_batches.clear();
            checkpoint.pass1_matches.clear();
        }

        assert!(checkpoint.pass1_completed_batches.is_empty());
        assert!(checkpoint.pass1_matches.is_empty());
    }
}

impl MapperCheckpointStore {
    fn for_repo(repo: &RepoDatabase) -> Option<Self> {
        repo.path().map(|path| Self::beside(path))
    }

    fn beside(database: &std::path::Path) -> Self {
        let name = database
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        Self {
            path: database.with_file_name(format!(".{name}{MAPPER_CHECKPOINT_SUFFIX}")),
        }
    }

    fn load(&self, extracted_ids: &[i32]) -> Option<MapperCheckpoint> {
        let checkpoint = std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|raw| serde_json::from_str::<MapperCheckpoint>(&raw).ok())?;
        (checkpoint.extracted_ids == extracted_ids).then_some(checkpoint)
    }

    fn store(&self, checkpoint: &MapperCheckpoint) -> Result<()> {
        let temp = self.path.with_extension("json.tmp");
        let payload = serde_json::to_vec_pretty(checkpoint)
            .wrap_err("failed to serialize mapper checkpoint")?;
        std::fs::write(&temp, payload)
            .wrap_err("failed to write mapper checkpoint")?;
        std::fs::rename(&temp, &self.path)
            .wrap_err("failed to atomically publish mapper checkpoint")?;
        Ok(())
    }

    fn remove(&self) {
        if let Err(error) = std::fs::remove_file(&self.path)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(path = %self.path.display(), "failed to remove mapper checkpoint: {error}");
        }
    }
}

/// Tunables for one Knowledge Mapper pass.
#[derive(Debug, Clone)]
pub struct MapperOptions {
    /// Historical semantics presented per matching prompt.
    pub batch_size: usize,
    /// Max historical batches matched concurrently. Each batch is one
    /// LLM call, so this fans the matching prompts out over this many
    /// workers (async-channel + `JoinSet`, the same idiom as the fuzz
    /// runtime). `1` keeps the old strictly-sequential behaviour.
    pub concurrency: usize,
    /// Extra LLM endpoints, each with its own worker count. Their workers pull
    /// from the same batch queue as the primary's, so the load balances itself:
    /// whichever endpoint is free takes the next batch.
    pub extra_workers: Vec<(LLM, usize)>,
    /// Pre-rendered Markdown block describing the project's
    /// source language; verbatim-prepended to the mapper's
    /// system prompt. See [`crate::profile::ProfileOptions::language_prompt_prefix`]
    /// for the same dispatch convention.
    pub language_prompt_prefix: String,
    /// Extra historical categories to consider on top of the project's own
    /// extracted-semantic categories (which always include `Others`). Lets a
    /// caller widen the candidate pool when the project auto-classified too
    /// narrowly — e.g. a P2P-escrow project tagged only `Others` can add
    /// `Services` to surface signature/authorization historical knowledge.
    pub extra_categories: Vec<DeFiCategory>,
    /// Cap on merged raw children rendered per canonical in the matcher
    /// prompt. `0` suppresses the `merged_raw_examples` block entirely.
    /// Larger values give the LLM more concrete examples per cluster at the
    /// cost of prompt tokens. See [`DEFAULT_MAX_RENDERED_RAW_CHILDREN`].
    pub max_rendered_raw_children: usize,
    /// Run the second matching pass: re-match every extract that found nothing
    /// in its own categories against all remaining ones.
    ///
    /// Off by default. The pass exists to protect recall — an extract whose
    /// project was auto-classified into too narrow a category would otherwise
    /// get no link at all — and it does that by matching across domains, which
    /// is exactly where a link stops being about this project's mechanism. The
    /// knowledge graph's own motivating case (proportional-share accounting in
    /// an insurance pool and in an AMM) is a cross-domain match, so the pass
    /// can be worth running; but when the engagement is scoped to a program
    /// that pays for a named list of impacts on a named list of assets, a
    /// cross-domain analogy is the most expensive kind of candidate to chase:
    /// it costs a full pipeline and the impact it lands on is usually not on
    /// that list. Prefer `extra_categories` when the goal is to reach a known
    /// neighbouring domain; use this when the goal is to leave no semantic
    /// unmatched.
    pub widen_unmatched_categories: bool,
    /// Pass 1 matches each extract only against historical semantics of its
    /// own category, instead of against every category the project touches.
    /// Extracts and historicals are ordered by category so their chunks are
    /// (nearly) single-category, and a chunk pair that shares no category is
    /// never sent. On GMX this cut pass 1 from 3105 batches to a few hundred;
    /// on the earlier GMX tree only 8 of 85 High matches were cross-category.
    pub same_category_only: bool,
}

impl Default for MapperOptions {
    fn default() -> Self {
        Self {
            batch_size: DEFAULT_BATCH_SIZE,
            concurrency: 1,
            extra_workers: Vec::new(),
            language_prompt_prefix: String::new(),
            extra_categories: Vec::new(),
            max_rendered_raw_children: DEFAULT_MAX_RENDERED_RAW_CHILDREN,
            widen_unmatched_categories: false,
            same_category_only: false,
        }
    }
}

/// Knowledge Mapper agent.
///
/// Owns the project profile + language prompt prefix; `repo` /
/// `kg` / `llm` are method arguments to [`Self::run`] so this
/// struct has no lifetime params.
#[derive(Debug, Clone)]
pub struct KnowledgeMapper {
    profile: ProjectProfile,
    language_prompt_prefix: String,
}

impl KnowledgeMapper {
    pub fn new(profile: ProjectProfile, language_prompt_prefix: String) -> Self {
        Self {
            profile,
            language_prompt_prefix,
        }
    }

    /// Run one mapping pass and write the result into `repo`.
    pub async fn run(
        &self,
        repo: &RepoDatabase,
        kg: &HistoricalDatabase,
        llm: &LLM,
        options: &MapperOptions,
    ) -> Result<MapperOutcome> {
        let extracted = repo
            .load_project_semantics()
            .await
            .wrap_err("failed to load extracted project semantics")?;

        if extracted.is_empty() {
            tracing::warn!("Knowledge Mapper invoked with no extracted semantics; nothing to do");
            return Ok(MapperOutcome::default());
        }

        // Stable indices — must match `project_semantic.id` insertion order.
        let mut extracted_with_id: Vec<(i32, &ExtractedSemantic)> = extracted
            .iter()
            .enumerate()
            .map(|(i, sem)| ((i as i32) + 1, sem))
            .collect();
        if options.same_category_only {
            extracted_with_id.sort_by_key(|(_, sem)| sem.category.as_str());
        }

        let categories = collect_categories(&extracted, &options.extra_categories);
        if !options.extra_categories.is_empty() {
            tracing::info!(
                "Knowledge Mapper: --mapper-extra-categories added {}",
                options
                    .extra_categories
                    .iter()
                    .map(|c| c.as_str())
                    .join(", "),
            );
        }
        tracing::info!(
            "Knowledge Mapper considering {} historical categories: {}",
            categories.len(),
            categories.iter().map(|c| c.as_str()).join(", ")
        );

        let mut historical_semantics = kg
            .canonical_semantics_with_children_for_categories(&categories)
            .await
            .wrap_err("failed to load historical semantics from knowledge graph")?;
        if options.same_category_only {
            historical_semantics.sort_by(|a, b| a.canonical.category.cmp(&b.canonical.category));
        }

        if historical_semantics.is_empty() {
            tracing::warn!(
                "Knowledge Mapper found no historical semantics for the project's categories"
            );
            repo.write_semantic_match_results(&SemanticMatchSet::default())
                .await?;
            return Ok(MapperOutcome::default());
        }

        tracing::info!(
            "Mapping {} extracted semantic(s) against {} historical semantic(s) (batch_size={})",
            extracted_with_id.len(),
            historical_semantics.len(),
            options.batch_size
        );

        let extract_ids: BTreeSet<i32> = extracted_with_id.iter().map(|(id, _)| *id).collect();
        let system_prompt = build_system_prompt(&self.profile, &self.language_prompt_prefix);

        let checkpoint_store = MapperCheckpointStore::for_repo(repo);
        let extracted_ids: Vec<i32> = extracted_with_id.iter().map(|(id, _)| *id).collect();
        let mut checkpoint = checkpoint_store
            .as_ref()
            .and_then(|store| store.load(&extracted_ids));
        let mut checkpoint = checkpoint
            .take()
            .unwrap_or_else(|| MapperCheckpoint::empty(&extracted_ids));

        let pass1_candidate_ids: Vec<i32> = historical_semantics
            .iter()
            .map(|candidate| candidate.canonical.id)
            .collect();
        if checkpoint.pass1_candidate_ids != pass1_candidate_ids
            || checkpoint.pass1_batch_size != options.batch_size
            || checkpoint
                .pass1_completed_batches
                .iter()
                .any(|key| !key.contains(':'))
        {
            checkpoint.pass1_candidate_ids = pass1_candidate_ids;
            checkpoint.pass1_batch_size = options.batch_size;
            checkpoint.pass1_completed_batches.clear();
            checkpoint.pass1_matches.clear();
            checkpoint.pass2_candidate_ids.clear();
            checkpoint.pass2_batch_size = 0;
            checkpoint.pass2_completed_batches.clear();
            checkpoint.pass2_matches.clear();
        }

        let pass1_matches = batch_match(
            llm,
            options,
            &system_prompt,
            &extracted_with_id,
            &extract_ids,
            &historical_semantics,
            "pass1",
            &mut checkpoint,
            checkpoint_store.as_ref(),
        )
        .await?;

        let mut matches = pass1_matches;
        dedup_keep_strongest(&mut matches);

        // Pass 2: widen to remaining categories for any extracts still
        // unmatched after pass 1.
        let pass1_matched_ids: BTreeSet<i32> = matches.iter().map(|m| m.extract_id).collect();
        let unmatched_after_pass1: Vec<(i32, &ExtractedSemantic)> = extracted_with_id
            .iter()
            .filter(|(id, _)| !pass1_matched_ids.contains(id))
            .copied()
            .collect();

        let pass1_categories: BTreeSet<DeFiCategory> = categories.iter().copied().collect();
        let fallback_categories: Vec<DeFiCategory> = DeFiCategory::ALL
            .iter()
            .copied()
            .filter(|c| !pass1_categories.contains(c))
            .collect();

        let pass2_historicals = if unmatched_after_pass1.is_empty()
            || fallback_categories.is_empty()
        {
            Vec::new()
        } else if !options.widen_unmatched_categories {
            tracing::info!(
                "{} extract(s) matched nothing in their own categor(ies); cross-domain widening is \
                 off (--map-widen-unmatched-categories), so they get no link. Pass \
                 --mapper-extra-categories <Category> to reach a neighbouring domain instead of \
                 every domain.",
                unmatched_after_pass1.len(),
            );
            Vec::new()
        } else {
            tracing::info!(
                "Widening match for {} unmatched extract(s) across {} fallback categor(ies): {}",
                unmatched_after_pass1.len(),
                fallback_categories.len(),
                fallback_categories.iter().map(|c| c.as_str()).join(", ")
            );
            kg.canonical_semantics_with_children_for_categories(&fallback_categories)
                .await
                .wrap_err("failed to load fallback historical semantics from knowledge graph")?
        };

        let pass2_matches = if pass2_historicals.is_empty() {
            Vec::new()
        } else {
            tracing::info!(
                "Pass 2: matching {} unmatched extract(s) against {} fallback historical(s)",
                unmatched_after_pass1.len(),
                pass2_historicals.len(),
            );
            let pass2_extract_ids: BTreeSet<i32> =
                unmatched_after_pass1.iter().map(|(id, _)| *id).collect();
            let pass2_candidate_ids: Vec<i32> = pass2_historicals
                .iter()
                .map(|candidate| candidate.canonical.id)
                .collect();
            if checkpoint.pass2_candidate_ids != pass2_candidate_ids
                || checkpoint.pass2_batch_size != options.batch_size
                || checkpoint
                    .pass2_completed_batches
                    .iter()
                    .any(|key| !key.contains(':'))
            {
                checkpoint.pass2_candidate_ids = pass2_candidate_ids;
                checkpoint.pass2_batch_size = options.batch_size;
                checkpoint.pass2_completed_batches.clear();
                checkpoint.pass2_matches.clear();
            }
            batch_match(
                llm,
                options,
                &system_prompt,
                &unmatched_after_pass1,
                &pass2_extract_ids,
                &pass2_historicals,
                "pass2",
                &mut checkpoint,
                checkpoint_store.as_ref(),
            )
            .await?
        };

        let pass2_match_count = pass2_matches.len();
        matches.extend(pass2_matches);
        dedup_keep_strongest(&mut matches);

        let matched_extract_ids: BTreeSet<i32> = matches.iter().map(|m| m.extract_id).collect();
        let total_historical_pool = historical_semantics.len() + pass2_historicals.len();
        for (extract_id, sem) in &extracted_with_id {
            if !matched_extract_ids.contains(extract_id) {
                tracing::warn!(
                    "Extracted semantic '{}' (id={}) had no historical match across {} candidate(s) even after fallback widening",
                    sem.name,
                    extract_id,
                    total_historical_pool
                );
            }
        }

        let matched_historical_ids: Vec<i32> =
            matches.iter().map(|m| m.historical_id).unique().collect();

        let mut semantic_by_id: HashMap<i32, semantic_node::Model> = historical_semantics
            .into_iter()
            .map(|c| (c.canonical.id, c.canonical))
            .collect();
        for c in pass2_historicals {
            semantic_by_id.entry(c.canonical.id).or_insert(c.canonical);
        }

        let finding_pairs = if matched_historical_ids.is_empty() {
            Vec::new()
        } else {
            kg.findings_for_semantic_ids(&matched_historical_ids)
                .await
                .wrap_err("failed to load historical findings for matched semantics")?
        };

        // Folded children + their merge-edge notes, so the mirror carries each
        // canonical's merged variants (concrete deltas) for downstream rendering.
        let mut sem_children = kg
            .semantic_children_with_notes(&matched_historical_ids)
            .await
            .wrap_err("failed to load semantic children for matched canonicals")?;
        let finding_ids: Vec<i32> = finding_pairs
            .iter()
            .map(|l| l.finding.id)
            .unique()
            .collect();
        let mut finding_children = kg
            .finding_children_with_notes(&finding_ids)
            .await
            .wrap_err("failed to load finding children for matched canonicals")?;

        let mut findings_by_semantic: BTreeMap<i32, Vec<HistoricalLinkedFinding>> = BTreeMap::new();
        for linked in finding_pairs {
            // Attach children to the first link of each finding only — the write
            // path dedups child rows, and one edge insert per (child, canonical)
            // avoids a duplicate-key collision.
            let raw_children = finding_children
                .remove(&linked.finding.id)
                .unwrap_or_default()
                .into_iter()
                .map(|(finding, d, p, x)| HistoricalFindingChild {
                    finding,
                    appended_description: d,
                    appended_patterns: p,
                    appended_exploits: x,
                })
                .collect();
            findings_by_semantic
                .entry(linked.canonical_semantic_id)
                .or_default()
                .push(HistoricalLinkedFinding {
                    finding: linked.finding,
                    strength: linked.strength,
                    evidence: linked.evidence,
                    raw_children,
                    rendered_description: String::new(),
                    rendered_patterns: String::new(),
                    rendered_exploits: String::new(),
                });
        }

        let historicals: Vec<HistoricalSemanticRecord> = matched_historical_ids
            .iter()
            .filter_map(|id| {
                semantic_by_id.get(id).cloned().map(|semantic| {
                    let findings = findings_by_semantic.remove(id).unwrap_or_default();
                    let raw_children = sem_children
                        .remove(id)
                        .unwrap_or_default()
                        .into_iter()
                        .map(|(node, appended_description)| HistoricalSemanticChild {
                            node,
                            appended_description,
                        })
                        .collect();
                    HistoricalSemanticRecord {
                        semantic,
                        findings,
                        raw_children,
                        rendered_description: String::new(),
                    }
                })
            })
            .collect();

        let mut strength_counts = StrengthCounts::default();
        for m in &matches {
            strength_counts.bump(m.strength);
        }

        let outcome = MapperOutcome {
            extracted_count: extracted_with_id.len(),
            historical_considered: total_historical_pool,
            matched_pair_count: matches.len(),
            widened_match_pair_count: pass2_match_count,
            unmatched_extract_count: extracted_with_id.len() - matched_extract_ids.len(),
            historical_finding_count: historicals.iter().map(|r| r.findings.len()).sum(),
            strength_counts,
        };

        let set = SemanticMatchSet {
            historicals,
            matches,
        };
        repo.write_semantic_match_results(&set)
            .await
            .wrap_err("failed to persist mapper output to project database")?;

        if let Some(store) = &checkpoint_store {
            store.remove();
        }

        tracing::info!(
            "Knowledge Mapper wrote {} match edge(s) (High={}/Medium={}/Low={}; {} from widened fallback) covering {} historical semantic(s) and {} finding(s); {} extracted semantic(s) had no match",
            outcome.matched_pair_count,
            outcome.strength_counts.high,
            outcome.strength_counts.medium,
            outcome.strength_counts.low,
            outcome.widened_match_pair_count,
            set.historicals.len(),
            outcome.historical_finding_count,
            outcome.unmatched_extract_count,
        );

        Ok(outcome)
    }
}

/// Drive one batched matching pass.
async fn batch_match(
    llm: &LLM,
    options: &MapperOptions,
    system_prompt: &str,
    extracts: &[(i32, &ExtractedSemantic)],
    _valid_extract_ids: &BTreeSet<i32>,
    historicals: &[CanonicalWithChildren<semantic_node::Model>],
    label_prefix: &str,
    checkpoint: &mut MapperCheckpoint,
    checkpoint_store: Option<&MapperCheckpointStore>,
) -> Result<Vec<SemanticMatch>> {
    if extracts.is_empty() || historicals.is_empty() {
        return Ok(Vec::new());
    }

    let batch_size = options.batch_size.max(1);
    let concurrency = options.concurrency.max(1);
    let max_rendered_raw_children = options.max_rendered_raw_children;
    let extract_chunks: Vec<Vec<(i32, &ExtractedSemantic)>> = extracts
        .chunks(EXTRACT_BATCH_SIZE)
        .map(|chunk| chunk.to_vec())
        .collect();
    let historical_chunks: Vec<Vec<CanonicalWithChildren<semantic_node::Model>>> = historicals
        .chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect();
    let is_pass1 = label_prefix == "pass1";
    // Which (extract chunk, historical chunk) pairs to send. Same-category
    // pairing keeps a pair only when the two chunks share a category.
    let pair_wanted = |extract_idx: usize, historical_idx: usize| -> bool {
        if !(is_pass1 && options.same_category_only) {
            return true;
        }
        chunks_share_category(
            extract_chunks[extract_idx].iter().map(|(_, sem)| sem.category),
            historical_chunks[historical_idx].iter().map(|h| h.canonical.category),
        )
    };
    let all_pairs = extract_chunks.len() * historical_chunks.len();
    let total_jobs = (0..extract_chunks.len())
        .flat_map(|e| (0..historical_chunks.len()).map(move |h| (e, h)))
        .filter(|&(e, h)| pair_wanted(e, h))
        .count();
    if total_jobs != all_pairs {
        tracing::info!(
            "mapper {label_prefix}: same-category pairing sends {total_jobs} of {all_pairs} batch pair(s)"
        );
    }

    // Shared prompt text is cloned into each worker; per-job extract blocks
    // and candidate batches are sent through the queue.
    let system_prompt = Arc::new(system_prompt.to_string());

    // Job queue sized to hold every batch, so feeding it never blocks
    // on a slow worker; results come back on an mpsc the collector
    // drains.
    type Job = (
        String,
        String,
        BTreeSet<i32>,
        Vec<CanonicalWithChildren<semantic_node::Model>>,
        String,
    );
    let (tx, rx) = async_channel::bounded::<(String, Job)>(total_jobs + 1);
    let (out_tx, mut out_rx) = tokio::sync::mpsc::channel::<Result<(String, Vec<SemanticMatch>)>>(
        concurrency + 1,
    );

    let mut workers = JoinSet::new();
    // Each worker carries its fallback: none for the primary, the primary for
    // an extra endpoint. An extra worker whose batch fails hands that batch to
    // the primary and retires, so a flaky second provider cannot abort the pass.
    let worker_llms: Vec<(LLM, Option<LLM>)> = std::iter::repeat_n((llm.clone(), None), concurrency)
        .chain(options.extra_workers.iter().flat_map(|(extra, n)| {
            std::iter::repeat_n((extra.clone(), Some(llm.clone())), *n)
        }))
        .collect();
    if !options.extra_workers.is_empty() {
        tracing::info!(
            "mapper: {} worker(s) on the primary endpoint + {} on extra endpoint(s)",
            concurrency,
            worker_llms.len() - concurrency
        );
    }
    for (llm, fallback) in worker_llms {
        let rx = rx.clone();
        let out = out_tx.clone();
        let checkpointed = if is_pass1 {
            checkpoint.pass1_completed_batches.clone()
        } else {
            checkpoint.pass2_completed_batches.clone()
        };
        workers.spawn(async move {
            while let Ok((batch_key, (extract_block, job_system, job_extract_ids, batch, label))) =
                rx.recv().await
            {
                if checkpointed.contains(&batch_key) {
                    continue;
                }
                let mut res = run_one_batch(
                    &llm,
                    &job_system,
                    &extract_block,
                    &job_extract_ids,
                    &batch,
                    max_rendered_raw_children,
                    &label,
                )
                .await;
                let mut retire = false;
                if let (Err(err), Some(primary)) = (&res, fallback.as_ref()) {
                    tracing::warn!(
                        "{label}: extra-endpoint mapper worker failed ({err:#}); \
                         rerunning the batch on the primary endpoint and retiring the worker"
                    );
                    res = run_one_batch(
                        primary,
                        &job_system,
                        &extract_block,
                        &job_extract_ids,
                        &batch,
                        max_rendered_raw_children,
                        &label,
                    )
                    .await;
                    retire = true;
                }
                // A closed receiver means the collector hit an error
                // and bailed; stop pulling work.
                if out.send(res.map(|matches| (batch_key, matches))).await.is_err() || retire {
                    break;
                }
            }
        });
    }
    drop(out_tx);
    drop(rx);

    let completed_batches = if is_pass1 {
        &checkpoint.pass1_completed_batches
    } else {
        &checkpoint.pass2_completed_batches
    };
    for (extract_idx, extract_chunk) in extract_chunks.iter().enumerate() {
        let extract_block = render_extracted_block(extract_chunk);
        let extract_ids: BTreeSet<i32> = extract_chunk.iter().map(|(id, _)| *id).collect();
        for (historical_idx, batch) in historical_chunks.iter().enumerate() {
            let batch_key = format!("{extract_idx}:{historical_idx}");
            if completed_batches.contains(&batch_key) || !pair_wanted(extract_idx, historical_idx) {
                continue;
            }
            let label = format!(
                "knowledge-mapper {} batch {}/{} (extract {}/{})",
                label_prefix,
                historical_idx + 1,
                historical_chunks.len(),
                extract_idx + 1,
                extract_chunks.len()
            );
            tx.send((
                batch_key,
                (
                    extract_block.clone(),
                    system_prompt.as_ref().clone(),
                    extract_ids.clone(),
                    batch.clone(),
                    label,
                ),
            ))
            .await
            .expect("mapper workers kept alive");
        }
    }
    drop(tx);

    let mut matches: Vec<SemanticMatch> = if is_pass1 {
        checkpoint.pass1_matches.clone()
    } else {
        checkpoint.pass2_matches.clone()
    };
    let mut first_err: Option<color_eyre::eyre::Error> = None;
    // A batch the model keeps answering with no parseable JSON is skipped, not
    // fatal: its pairs get no match, and the pass goes on. Only a billing-cap
    // stop, or failures too many to be one bad prompt, end the pass.
    let max_failed = (total_jobs / 20).max(3);
    let mut failed = 0usize;
    while let Some(res) = out_rx.recv().await {
        match res {
            Ok((batch_key, batch_matches)) => {
                matches.extend(batch_matches.clone());
                if is_pass1 {
                    checkpoint.pass1_completed_batches.insert(batch_key.clone());
                    checkpoint.pass1_matches = matches.clone();
                } else {
                    checkpoint.pass2_completed_batches.insert(batch_key.clone());
                    checkpoint.pass2_matches = matches.clone();
                }
                if let Some(store) = checkpoint_store {
                    store.store(checkpoint)?;
                    tracing::debug!(
                        pass = %label_prefix,
                        batch = %batch_key,
                        "knowledge-mapper checkpoint persisted"
                    );
                }
            }
            Err(err) if !crate::spec::is_billing_exhausted(&err) && failed < max_failed => {
                failed += 1;
                tracing::warn!(
                    "knowledge-mapper {label_prefix}: a batch failed and is skipped, its pairs \
                     get no match ({failed}/{max_failed} allowed): {err:#}"
                );
            }
            Err(err) => {
                // Abort the remaining batches — a billing stop, or failures
                // past the cap — without waiting on in-flight LLM calls.
                first_err = Some(err);
                break;
            }
        }
    }
    if first_err.is_some() {
        workers.abort_all();
    }
    drop(out_rx);
    while let Some(joined) = workers.join_next().await {
        match joined {
            Ok(()) => {}
            Err(err) if err.is_cancelled() => {}
            Err(err) => tracing::error!("knowledge-mapper worker task panicked: {err:#}"),
        }
    }
    if let Some(err) = first_err {
        return Err(err);
    }
    if failed > 0 {
        tracing::warn!(
            "knowledge-mapper {label_prefix}: {failed} batch(es) skipped after repeated failures; \
             their (extract, historical) pairs got no match"
        );
    }

    Ok(matches)
}

/// Whether an extract chunk and a historical chunk share at least one
/// category — the pairs [`MapperOptions::same_category_only`] keeps.
fn chunks_share_category(
    extract_categories: impl IntoIterator<Item = DeFiCategory>,
    historical_categories: impl IntoIterator<Item = DeFiCategory>,
) -> bool {
    let extract_categories: BTreeSet<DeFiCategory> = extract_categories.into_iter().collect();
    historical_categories
        .into_iter()
        .any(|category| extract_categories.contains(&category))
}

/// Match one batch of historical semantics against the (pre-rendered)
/// extract block: one LLM call + the per-record validation /
/// forced-per-candidate-emit accounting. Returns the surviving
/// [`SemanticMatch`] rows; an `Err` aborts the whole mapping pass.
async fn run_one_batch(
    llm: &LLM,
    system_prompt: &str,
    extracted_block: &str,
    valid_extract_ids: &BTreeSet<i32>,
    batch: &[CanonicalWithChildren<semantic_node::Model>],
    max_rendered_raw_children: usize,
    label: &str,
) -> Result<Vec<SemanticMatch>> {
    let user_prompt = build_user_prompt(extracted_block, batch, max_rendered_raw_children);

    let response: MatchResponse = llm
        .prompt_json_with_retry(system_prompt, &user_prompt, Some("mapper"), None, None)
        .await?;

    let valid_historical_ids: BTreeSet<i32> = batch.iter().map(|c| c.canonical.id).collect();

    // Forced per-candidate emit: every
    // (extract_id × historical_id) pair shown in this batch should
    // appear in the response. We don't bounce the response on missing
    // coverage (no agent re-loop on the JSON path right now), but we
    // log a warning so calibration runs can see when the model is
    // sneakily omitting Low pairs.
    let expected_pairs: BTreeSet<(i32, i32)> = valid_extract_ids
        .iter()
        .flat_map(|eid| valid_historical_ids.iter().map(move |hid| (*eid, *hid)))
        .collect();
    let mut emitted_pairs: BTreeSet<(i32, i32)> = BTreeSet::new();
    let mut matches: Vec<SemanticMatch> = Vec::new();

    for record in response.matches {
        if !valid_historical_ids.contains(&record.historical_id) {
            tracing::warn!(
                "{label}: dropping match for unknown historical id {}",
                record.historical_id
            );
            continue;
        }
        if !valid_extract_ids.contains(&record.extract_id) {
            tracing::warn!(
                "{label}: dropping match for unknown extract id {}",
                record.extract_id
            );
            continue;
        }
        let Some(strength) = MatchStrength::parse(&record.strength) else {
            tracing::warn!(
                "{label}: dropping (extract={}, historical={}): unparseable strength '{}'",
                record.extract_id,
                record.historical_id,
                record.strength
            );
            continue;
        };
        let evidence = record.evidence.trim().to_string();
        let floor = match strength {
            MatchStrength::High | MatchStrength::Medium => MIN_EVIDENCE_LEN_HIGH_MEDIUM,
            MatchStrength::Low => MIN_EVIDENCE_LEN_LOW,
        };
        if evidence.len() < floor {
            tracing::warn!(
                "{label}: dropping (extract={}, historical={}, strength={}): evidence shorter than {} chars: {:?}",
                record.extract_id,
                record.historical_id,
                strength,
                floor,
                evidence
            );
            continue;
        }
        emitted_pairs.insert((record.extract_id, record.historical_id));
        matches.push(SemanticMatch {
            extract_id: record.extract_id,
            historical_id: record.historical_id,
            strength,
            evidence,
        });
    }

    let missing: Vec<(i32, i32)> = expected_pairs.difference(&emitted_pairs).copied().collect();
    if !missing.is_empty() {
        tracing::warn!(
            "{label}: model scored only {} of the {} (extract, historical) pair(s) it was shown. \
             Every pair must get a verdict — an omitted pair is silently treated as no match, so \
             a real match can be lost here. Usually means the batch was too large for the model \
             to answer exhaustively; lower --map-batch-size if it keeps happening. First few \
             unscored: {:?}",
            emitted_pairs.len(),
            expected_pairs.len(),
            &missing[..missing.len().min(5)],
        );
    }

    Ok(matches)
}

/// Drop duplicate `(extract_id, historical_id)` rows; on collision
/// keep the higher-strength entry (so a Pass-2 retry that re-emits
/// the same pair at a different strength doesn't downgrade Pass 1).
fn dedup_keep_strongest(matches: &mut Vec<SemanticMatch>) {
    let mut by_key: BTreeMap<(i32, i32), SemanticMatch> = BTreeMap::new();
    for m in matches.drain(..) {
        let key = (m.extract_id, m.historical_id);
        by_key
            .entry(key)
            .and_modify(|existing| {
                if m.strength.rank() > existing.strength.rank() {
                    *existing = m.clone();
                }
            })
            .or_insert(m);
    }
    matches.extend(by_key.into_values());
    // Stable sort by (extract_id, historical_id) so downstream snapshots
    // are deterministic.
    matches.sort_by(|a, b| (a.extract_id, a.historical_id).cmp(&(b.extract_id, b.historical_id)));
}

/// Summary statistics for one mapper pass.
#[derive(Debug, Clone, Default)]
pub struct MapperOutcome {
    pub extracted_count: usize,
    pub historical_considered: usize,
    pub matched_pair_count: usize,
    /// Subset of `matched_pair_count` rescued by the widened fallback.
    pub widened_match_pair_count: usize,
    pub unmatched_extract_count: usize,
    pub historical_finding_count: usize,
    pub strength_counts: StrengthCounts,
}

#[derive(Debug, Clone, Default)]
pub struct StrengthCounts {
    pub high: usize,
    pub medium: usize,
    pub low: usize,
}

impl StrengthCounts {
    fn bump(&mut self, s: MatchStrength) {
        match s {
            MatchStrength::High => self.high += 1,
            MatchStrength::Medium => self.medium += 1,
            MatchStrength::Low => self.low += 1,
        }
    }
}

fn collect_categories(
    extracted: &[ExtractedSemantic],
    extra: &[DeFiCategory],
) -> Vec<DeFiCategory> {
    let mut cats: BTreeSet<DeFiCategory> = extracted.iter().map(|s| s.category).collect();
    cats.insert(DeFiCategory::Others);
    cats.extend(extra.iter().copied());
    cats.into_iter().collect()
}

fn render_extracted_block(extracted: &[(i32, &ExtractedSemantic)]) -> String {
    let mut out = String::new();
    for (id, sem) in extracted {
        out.push_str(&format!(
            "- extract_id={} | category={} | name=\"{}\"\n  definition: {}\n  description: {}\n",
            id,
            sem.category.as_str(),
            sem.name,
            sem.definition.trim(),
            sem.description.trim()
        ));
    }
    out
}

fn render_historical_block(
    batch: &[CanonicalWithChildren<semantic_node::Model>],
    max_rendered_raw_children: usize,
) -> String {
    let mut out = String::new();
    for entry in batch {
        let canonical = &entry.canonical;
        out.push_str(&format!(
            "- historical_id={} | category={} | name=\"{}\"\n  definition: {}\n  description: {}\n",
            canonical.id,
            canonical.category.as_str(),
            canonical.name,
            canonical.definition.trim(),
            canonical.description.trim()
        ));
        if !entry.raw_children.is_empty() && max_rendered_raw_children > 0 {
            let total = entry.raw_children.len();
            let shown = total.min(max_rendered_raw_children);
            out.push_str(&format!(
                "  merged_raw_examples ({} of {}; canonical may match if any align):\n",
                shown, total
            ));
            for raw in entry.raw_children.iter().take(shown) {
                out.push_str(&format!(
                    "    * raw \"{}\" — {}\n",
                    raw.name,
                    raw.description.trim().replace('\n', " ")
                ));
            }
            if total > shown {
                out.push_str(&format!(
                    "    * (+{} more merged raw children not shown)\n",
                    total - shown
                ));
            }
        }
    }
    out
}

fn build_user_prompt(
    extracted_block: &str,
    batch: &[CanonicalWithChildren<semantic_node::Model>],
    max_rendered_raw_children: usize,
) -> String {
    format!(
        "## Extracted project semantics\n{extracted}\n\
         \n## Historical knowledge-graph semantics (this batch)\n\
         Each historical entry is a *canonical semantic cluster*; `merged_raw_examples` (when \
         present) shows specific raw semantics merged into the canonical and indicates the kinds \
         of behaviours the cluster covers.\n{historical}\n\
         \n## Task\nEmit ONE entry in the `matches` array for EVERY `(extract_id, historical_id)` \
         pair shown above — no silent omission. Each entry must carry an explicit `strength` from \
         {{`High`, `Medium`, `Low`}} plus an `evidence` rationale. Apply the strength rubric from \
         the system prompt; default to `Low` when uncertain. Evidence length floors: `High` and \
         `Medium` >= 40 chars (cite the specific shared mechanism), `Low` >= 15 chars (a brief \
         reason is enough). Respond with strict JSON in the exact shape:\n\
         ```json\n\
         {{\"matches\":[{{\"extract_id\":1,\"historical_id\":123,\"strength\":\"High\",\"evidence\":\"...\"}},{{\"extract_id\":1,\"historical_id\":124,\"strength\":\"Low\",\"evidence\":\"different mechanism\"}}]}}\n\
         ```\nNever invent ids that were not listed in the inputs. Coverage is mandatory: missing \
         pairs are treated as the agent shirking the contract and a warning is logged.",
        extracted = extracted_block,
        historical = render_historical_block(batch, max_rendered_raw_children),
    )
}

fn build_system_prompt(profile: &ProjectProfile, language_prompt_prefix: &str) -> String {
    let mut profile_block = String::new();
    if !language_prompt_prefix.trim().is_empty() {
        profile_block.push_str(language_prompt_prefix.trim_end());
        profile_block.push_str("\n\n");
    }
    profile_block.push_str("## Project profile\n");
    profile_block.push_str(&format!(
        "Domain summary: {}\n\n",
        profile.domain_summary.trim()
    ));
    profile_block.push_str("Subsystems (mapper compares historical descriptions against these `summary` lines, not against `name`):\n");
    for s in &profile.subsystems {
        profile_block.push_str(&format!("- {} — {}\n", s.name, s.summary.trim()));
    }
    profile_block.push('\n');
    profile_block.push_str(
        "Core components (project source the mapper may cite when justifying a High match):\n",
    );
    for c in &profile.core_components {
        profile_block.push_str(&format!("- {} ({}) — {}\n", c.name, c.path, c.role.trim()));
    }
    profile_block.push('\n');
    profile_block.push_str("Out-of-scope notes (use these to confidently emit Low when the historical's mechanism is excluded):\n");
    profile_block.push_str(profile.out_of_scope_notes.trim());
    profile_block.push_str("\n\n");

    format!(
        "{PREAMBLE}\n\n{profile_block}{RUBRIC}",
        PREAMBLE = MAPPER_PREAMBLE,
        profile_block = profile_block,
        RUBRIC = MAPPER_RUBRIC,
    )
}

const MAPPER_PREAMBLE: &str = "You are the Knowledge Mapper agent for an agentic smart-contract auditing pipeline. For each candidate (project_semantic ↔ historical_semantic) pair you emit, you MUST attach a `strength` label drawn from {High, Medium, Low} together with an `evidence` rationale (>= 40 chars). The label gates downstream cost: High pairs drive expensive spec generation; Low pairs may be abandoned. Wrong-tier labels waste budget — over-labeling as High burns tokens on dead leads; under-labeling as Low drops real bugs. Calibrate deliberately. Reply with strict JSON only, no commentary.";

const MAPPER_RUBRIC: &str = r#"## Strength rubric (single axis: MECHANISM OVERLAP)

Strength is determined by ONE check: does the historical semantic describe a mechanism that this project actually implements?

Both project and historical semantics are LLM-extracted records with the same shape: `category` (one of the DeFi enum values), `name` (free-form short label), `definition` (one-sentence formal statement), `description` (abstract description of user interaction / value flow). Names are NOT drawn from a shared vocabulary — different extractions of the same mechanism may have different names. Judge on definition + description content, not surface name match.

Map your judgment directly to strength. The bar between Medium and Low matters most — most cross-domain pairs should be Low, not Medium.

  High    Direct mechanism match. The historical's definition/description describes the same mechanism that appears in either:
            (a) one of this project's extracted semantics (match on described mechanism, not surface name), OR
            (b) a subsystem listed in the project profile above.
          The historical's described behaviour would apply to project code with at most cosmetic substitution (rename a variable, point at a different function in the same role).

  Medium  Partial mechanism overlap. The historical describes a process that SHARES CONCRETE SUB-STEPS with something the project does but differs in detail. To label Medium you must point to a specific shared sub-pattern that BOTH sides exhibit, such as:
            - same lifecycle stage with different accounting formula (e.g. both compute a per-user pending balance, but with different aggregation math)
            - same access-control style with different role granularity (e.g. both gate on roles, but project uses two roles and historical uses N)
            - same external-call pattern with different re-entrancy surface (e.g. both push tokens then update state, but in different orderings)
          The historical's behaviour cannot be applied without re-interpretation — but the re-interpretation has a concrete anchor on the project side.

          NOT acceptable as Medium evidence:
            - "same DeFiCategory" alone (Services / Others / Lending are far too broad; many unrelated mechanisms share them)
            - "both involve smart contracts" / "both are DeFi"
            - "both have an admin role" without naming what the role gates
          If you cannot name a concrete shared sub-pattern, the answer is Low, not Medium.

  Low     No meaningful mechanism overlap. Includes:
            - historical's mechanism has no analog in the project's extracted semantics or profile subsystems
            - project profile out_of_scope_notes explicitly excludes the historical's subdomain
            - same DeFiCategory but the mechanisms are entirely different (e.g. both Services, but one is timelock and the other is meta-transaction relayer)
          Most cross-domain pairs land here. This is the expected default — do not apologise for emitting Low. You MUST emit a `Low` entry for every candidate that isn't a clear High/Medium — there is no silent omission. The user prompt restates this as the per-batch contract; the response handler logs a warning when pairs are missing.

## Evidence requirements

The `evidence` field is REQUIRED. Length floors are tier-dependent because the new contract forces emitting per-candidate and most candidates land Low:
  - For High (>= 40 chars): describe the shared mechanism in your own words, citing the project-side (extract_id from the user prompt or a subsystem name from the profile) and the historical side (paraphrase its definition/description).
  - For Medium (>= 40 chars): name the concrete shared sub-pattern (NOT a category, NOT "both DeFi"); state what is shared AND what differs (the detail that prevents High).
  - For Low (>= 15 chars): brief reason is fine. State why the mechanism doesn't apply. If you can only write "same category" or "both X-protocols", that is not Medium — that is Low. Don't pad Low evidence; short reasons are expected.

## Anti-patterns

  • DEFAULT TO LOW when uncertain. No penalty for emitting Low. Real penalty for emitting Medium-or-above without concrete shared-sub-pattern evidence.
  • CATEGORY MATCH IS NOT MEDIUM. Two semantics sharing DeFiCategory is not, by itself, partial overlap.
  • DO NOT label every match Medium as a hedge. If you cannot name a concrete sub-pattern, the correct label is Low.
  • DO NOT promote to High solely because category fields agree, or because surface names overlap. High requires same mechanism, not same label.
  • DO NOT downgrade to Low just because the historical is from a different project. That is expected — every historical is from a different project. Downgrade only when the project's extracted semantics / profile subsystems / out_of_scope_notes show the mechanism truly isn't present."#;

#[derive(Debug, Deserialize)]
struct MatchResponse {
    matches: Vec<MatchEntry>,
}

#[derive(Debug, Deserialize)]
struct MatchEntry {
    extract_id: i32,
    historical_id: i32,
    strength: String,
    evidence: String,
}

#[cfg(test)]
mod same_category_tests {
    use super::*;

    #[test]
    fn a_chunk_pair_is_kept_only_when_a_category_is_shared() {
        let extracts = [DeFiCategory::Derivatives, DeFiCategory::Dexes];
        assert!(!chunks_share_category(extracts, [DeFiCategory::Lending]));
        assert!(chunks_share_category(extracts, [DeFiCategory::Lending, DeFiCategory::Dexes]));
        assert!(!chunks_share_category([], [DeFiCategory::Dexes]));
    }
}
