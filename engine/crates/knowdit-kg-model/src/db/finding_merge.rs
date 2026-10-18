use sea_orm::entity::prelude::*;

#[sea_orm::model]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, DeriveEntityModel)]
#[sea_orm(table_name = "finding_merge")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub from_finding_id: i32,
    #[sea_orm(primary_key, auto_increment = false)]
    pub to_finding_id: i32,
    /// One-or-two-sentence notes, produced at merge time (all in the same emit
    /// call), describing how the raw child (`from_finding_id`) *extends* the
    /// canonical (`to_finding_id`) beyond the canonical's own
    /// concrete-representative fields — one per mutable field. Lets downstream
    /// rendering keep the canonical concise while still surfacing each merged
    /// variant's concrete delta.
    #[sea_orm(column_type = "Text")]
    pub appended_description: String,
    #[sea_orm(column_type = "Text")]
    pub appended_patterns: String,
    #[sea_orm(column_type = "Text")]
    pub appended_exploits: String,
}

impl ActiveModelBehavior for ActiveModel {}
