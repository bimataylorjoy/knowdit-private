//! Target-derived hypotheses share the mapper's tables under negative ids.
//! These tests pin the two properties that make that safe: a re-map leaves
//! them (and the specifications pointing at them) alone, and writing them
//! again never duplicates or deletes anything.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use knowdit_kg_model::audit_finding::FindingSeverity;
use knowdit_kg_model::category::DeFiCategory;
use knowdit_kg_model::db::{audit_finding, semantic_node};
use sea_orm::{ActiveValue::Set, EntityTrait};

use crate::db::{project_semantic as project_semantic_model, specification as specification_model};
use crate::repo::{RepoDatabase, SemanticMatchSet, TargetHypothesisRow};

struct TempDb {
    repo: RepoDatabase,
    path: PathBuf,
}

impl Drop for TempDb {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        let _ = std::fs::remove_file(self.path.with_extension("sqlite3-shm"));
        let _ = std::fs::remove_file(self.path.with_extension("sqlite3-wal"));
    }
}

async fn temp_db() -> TempDb {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "knowdit-target-hypotheses-test-{}-{unique}.sqlite3",
        std::process::id()
    ));
    let repo = RepoDatabase::open_sqlite(path.clone())
        .await
        .expect("test repo database should connect");
    repo.init_schema().await.expect("schema should initialize");
    TempDb { repo, path }
}

fn row(extract_id: i32, id: i32) -> TargetHypothesisRow {
    TargetHypothesisRow {
        extract_id,
        semantic: semantic_node::Model {
            id,
            name: format!("Target invariant {id}"),
            definition: "totalAssets == sum(balances)".to_string(),
            description: String::new(),
            category: DeFiCategory::Yield,
        },
        finding: audit_finding::Model {
            id,
            title: format!("[target accounting] identity {id}"),
            severity: FindingSeverity::Medium,
            root_cause: String::new(),
            description: String::new(),
            patterns: String::new(),
            exploits: String::new(),
        },
        evidence: "target-derived accounting invariant".to_string(),
    }
}

#[tokio::test]
async fn target_hypotheses_survive_remap_and_upsert_is_idempotent() {
    let db = temp_db().await;
    let repo = &db.repo;
    project_semantic_model::Entity::insert(project_semantic_model::ActiveModel {
        id: Set(1),
        name: Set("Vault deposit".to_string()),
        category: Set(DeFiCategory::Yield),
        definition: Set(String::new()),
        description: Set(String::new()),
        ..Default::default()
    })
    .exec(repo.connection())
    .await
    .expect("extract row");

    assert_eq!(repo.upsert_target_hypotheses(&[row(1, -7)]).await.expect("upsert"), 1);
    // Writing the same hypothesis again adds nothing.
    assert_eq!(repo.upsert_target_hypotheses(&[row(1, -7)]).await.expect("upsert"), 0);
    assert!(!repo.has_mapper_output().await.expect("count"));

    // A specification built on the hypothesis.
    specification_model::Entity::insert(specification_model::ActiveModel {
        semantic_id: Set(1),
        historical_id: Set(-7),
        finding_id: Set(-7),
        specification: Set("{}".to_string()),
        ..Default::default()
    })
    .exec(repo.connection())
    .await
    .expect("spec on target hypothesis");

    // A re-map with empty mapper output must not touch the hypothesis.
    repo.write_semantic_match_results(&SemanticMatchSet {
        historicals: Vec::new(),
        matches: Vec::new(),
    })
    .await
    .expect("re-map leaves negative ids alone");

    let loaded = repo.load_semantic_match_results().await.expect("load");
    assert_eq!(loaded.matches.len(), 1);
    assert_eq!(loaded.matches[0].historical_id, -7);
    let record = loaded
        .historicals
        .iter()
        .find(|record| record.semantic.id == -7)
        .expect("target historical loaded");
    assert_eq!(record.findings.len(), 1);
    assert_eq!(record.findings[0].finding.id, -7);
    // The pattern library never sees target rows.
    assert!(repo.load_historical_findings().await.expect("findings").is_empty());
}

#[tokio::test]
async fn positive_ids_are_refused() {
    let db = temp_db().await;
    assert!(db.repo.upsert_target_hypotheses(&[row(1, 5)]).await.is_err());
}
