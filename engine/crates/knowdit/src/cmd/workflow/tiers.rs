//! Model tiering for the streamed pipeline.
//!
//! Most of a run's calls go to work that ends in nothing: semantics nobody
//! links, mapper matches the novelty judge defers, and gen-spec agents that
//! close a resemblance link without a specification. Those phases can run on a
//! cheaper model because what they produce is either thrown away or held to a
//! deterministic gate before anything is built on it (the commit gates refuse
//! a specification whose scope, value flow, authority or magnitude does not
//! check out, whichever model wrote it).
//!
//! Everything a surviving candidate touches afterwards — the harness that has
//! to compile and prove it, regen, reflection and the review that decides
//! whether it is submitted — stays on the primary model, and so do the
//! target-derived hypotheses, which are few and are where a finding no corpus
//! holds comes from.

use std::collections::BTreeSet;

use clap::{Args, ValueEnum};
use color_eyre::eyre::{Result, eyre};
use knowdit_audit::spec::target_hypotheses::is_target_link;
use llmy::clap::OpenAISetup;
use llmy::client::client::LLM;
use llmy::client::model::OpenAIModel;
use llmy::client::rust_decimal::Decimal;
use llmy::client::settings::Reasoning;
use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// A pipeline phase that can be assigned to the triage model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum TieredPhase {
    Profile,
    Extract,
    Mapper,
    Novelty,
    /// Specification generation on a mapper link (a historical resemblance).
    Spec,
    /// Specification generation on a target-derived hypothesis.
    TargetSpec,
    Codegen,
    Regen,
    Review,
}

#[derive(Args, Debug, Clone)]
pub struct ModelTierArgs {
    /// Cheaper model for the high-volume, low-yield phases named by
    /// `--triage-phases`. It is reached through the same endpoint and key as
    /// the primary model. Unset: every phase runs on the primary model.
    #[arg(long = "triage-model", env = "TRIAGE_OPENAI_API_MODEL")]
    pub triage_model: Option<OpenAIModel>,

    /// Phases the triage model runs, comma-separated. The default keeps the
    /// primary model on target hypotheses and on everything downstream of a
    /// committed specification (codegen, regen, reflection, review).
    #[arg(
        long = "triage-phases",
        value_enum,
        value_delimiter = ',',
        default_value = "extract,mapper,novelty,spec"
    )]
    pub triage_phases: Vec<TieredPhase>,

    /// Billing cap of the triage model's own ledger. Defaults to the primary
    /// model's cap.
    #[arg(long = "triage-billing-cap", env = "TRIAGE_OPENAI_BILLING_CAP")]
    pub triage_billing_cap: Option<Decimal>,

    /// OpenAI-compatible base URL of a second, independent LLM endpoint. With
    /// `--secondary-concurrency N > 0`, N link pipelines and N mapper workers
    /// run on it in addition to the primary endpoint's, so throughput adds up
    /// across providers. Every phase of a link runs on the endpoint it leased.
    #[arg(long = "secondary-openai-url", env = "SECONDARY_OPENAI_API_URL")]
    pub secondary_url: Option<String>,
    /// API key for the secondary endpoint (read from env; never pass on a shared
    /// command line).
    #[arg(long = "secondary-openai-key", env = "SECONDARY_OPENAI_API_KEY", hide_env_values = true)]
    pub secondary_key: Option<String>,
    /// Model on the secondary endpoint.
    #[arg(long = "secondary-model", env = "SECONDARY_OPENAI_API_MODEL")]
    pub secondary_model: Option<OpenAIModel>,
    /// Link pipelines (and mapper workers) on the secondary endpoint. 0 disables it.
    #[arg(long = "secondary-concurrency", env = "SECONDARY_CONCURRENCY", default_value_t = 0)]
    pub secondary_concurrency: usize,
    /// Reasoning effort for the secondary model. Defaults to the primary's.
    #[arg(long = "secondary-reasoning-effort", env = "SECONDARY_LLM_REASONING_EFFORT")]
    pub secondary_reasoning_effort: Option<Reasoning>,
    /// Reasoning effort for the mapper workers on the secondary endpoint.
    /// Defaults to `--secondary-reasoning-effort`. A mapper batch answers with
    /// a long match list; on a gateway that cuts requests at a fixed wall-clock
    /// limit, a lower effort keeps each batch under it.
    #[arg(
        long = "secondary-mapper-reasoning-effort",
        env = "SECONDARY_MAPPER_REASONING_EFFORT"
    )]
    pub secondary_mapper_reasoning_effort: Option<Reasoning>,
    /// Stream responses from the secondary endpoint. A gateway that cuts
    /// non-streaming requests at a fixed wall-clock limit (Atria: ~300 s, a
    /// 502 `upstream_unavailable`) lets a streamed answer run past it.
    #[arg(long = "secondary-stream", env = "SECONDARY_LLM_STREAM", default_value_t = false)]
    pub secondary_stream: bool,
}

impl ModelTierArgs {
    /// Build the tiers. `setup` is the primary model's configuration: the
    /// triage model is the same configuration with another model name.
    pub async fn build(
        &self,
        setup: &OpenAISetup,
        primary: &LLM,
        primary_link_cap: usize,
    ) -> Result<ModelTiers> {
        let mut tiers = self.build_tiers(setup, primary).await?;
        tiers.secondary = self.build_secondary(setup, primary_link_cap).await?;
        Ok(tiers)
    }

    /// The secondary endpoint, when `--secondary-concurrency` > 0: the primary
    /// setup with the URL, key, model and (optionally) reasoning effort swapped.
    async fn build_secondary(
        &self,
        setup: &OpenAISetup,
        primary_link_cap: usize,
    ) -> Result<Option<SecondaryEndpoint>> {
        if self.secondary_concurrency == 0 {
            return Ok(None);
        }
        let (Some(url), Some(key), Some(model)) = (
            self.secondary_url.clone(),
            self.secondary_key.clone(),
            self.secondary_model.clone(),
        ) else {
            return Err(eyre!(
                "--secondary-concurrency > 0 needs --secondary-openai-url, \
                 --secondary-openai-key (SECONDARY_OPENAI_API_KEY) and --secondary-model"
            ));
        };
        let label = format!("{} @ {}", model.model_name(), url);
        let mut secondary = setup.clone();
        secondary.openai_url = Some(url);
        secondary.openai_key = Some(key);
        secondary.azure_openai_endpoint = None;
        secondary.model = Some(model);
        if let Some(effort) = self.secondary_reasoning_effort.clone() {
            secondary.reasoning_effort = Some(effort);
        }
        if self.secondary_stream {
            secondary.llm_stream = true;
        }
        let llm = secondary
            .clone()
            .may_llm()
            .await
            .map_err(|err| eyre!("failed to build the secondary LLM `{label}`: {err}"))?
            .ok_or_else(|| eyre!("secondary endpoint `{label}` did not produce an LLM"))?;
        let mapper_llm = match self.secondary_mapper_reasoning_effort.clone() {
            None => llm.clone(),
            Some(effort) => {
                tracing::info!("[tiers] secondary mapper workers use reasoning effort {effort:?}");
                let mut mapper = secondary.clone();
                mapper.reasoning_effort = Some(effort);
                mapper
                    .may_llm()
                    .await
                    .map_err(|err| eyre!("failed to build the secondary mapper LLM `{label}`: {err}"))?
                    .ok_or_else(|| eyre!("secondary endpoint `{label}` did not produce a mapper LLM"))?
            }
        };
        tracing::info!(
            "[tiers] secondary endpoint `{label}`: {} link pipeline(s) and mapper worker(s), \
             alongside {primary_link_cap} on the primary",
            self.secondary_concurrency
        );
        Ok(Some(SecondaryEndpoint {
            llm,
            mapper_llm,
            label,
            concurrency: self.secondary_concurrency,
            primary_permits: Arc::new(Semaphore::new(primary_link_cap.max(1))),
            secondary_permits: Arc::new(Semaphore::new(self.secondary_concurrency)),
        }))
    }

    async fn build_tiers(&self, setup: &OpenAISetup, primary: &LLM) -> Result<ModelTiers> {
        let Some(model) = self.triage_model.clone() else {
            return Ok(ModelTiers::single(primary.clone()));
        };
        let name = model.model_name().to_string();
        let mut triage = setup.clone();
        triage.model = Some(model);
        if let Some(cap) = self.triage_billing_cap {
            triage.biling_cap = cap;
        }
        let llm = triage
            .may_llm()
            .await
            .map_err(|err| eyre!("failed to build the triage LLM `{name}`: {err}"))?
            .ok_or_else(|| eyre!("--triage-model `{name}` did not produce an LLM"))?;
        let tiers = ModelTiers {
            primary: primary.clone(),
            triage: Some(llm),
            phases: self.triage_phases.iter().copied().collect(),
            secondary: None,
        };
        tracing::info!(
            "[tiers] triage model `{name}` runs: {}; everything else runs on the primary model",
            tiers.triaged_phases()
        );
        Ok(tiers)
    }
}

/// A second, independent endpoint links and mapper workers can run on.
#[derive(Clone)]
pub struct SecondaryEndpoint {
    llm: LLM,
    /// Same endpoint, possibly at another reasoning effort, for mapper workers.
    mapper_llm: LLM,
    label: String,
    concurrency: usize,
    /// Link slots on each endpoint. A link leases one for its whole run.
    primary_permits: Arc<Semaphore>,
    secondary_permits: Arc<Semaphore>,
}

/// The endpoint one link pipeline runs on, held for the link's lifetime.
pub struct EndpointLease {
    /// `None` = the primary tiers (triage routing included).
    pub secondary: Option<LLM>,
    pub label: String,
    _permit: Option<OwnedSemaphorePermit>,
}

/// Which model each phase runs on.
#[derive(Clone)]
pub struct ModelTiers {
    primary: LLM,
    triage: Option<LLM>,
    phases: BTreeSet<TieredPhase>,
    secondary: Option<SecondaryEndpoint>,
}

impl ModelTiers {
    /// Every phase on one model.
    pub fn single(primary: LLM) -> Self {
        Self {
            primary,
            triage: None,
            phases: BTreeSet::new(),
            secondary: None,
        }
    }

    /// Link slots the secondary endpoint adds to the scheduler's active limit.
    pub fn secondary_concurrency(&self) -> usize {
        self.secondary.as_ref().map_or(0, |s| s.concurrency)
    }

    /// Mapper workers on the secondary endpoint, scoped for billing.
    pub fn mapper_extra_workers(&self) -> Vec<(LLM, usize)> {
        self.secondary
            .iter()
            .map(|s| (s.mapper_llm.scope(Some("mapper-secondary".to_string()), None), s.concurrency))
            .collect()
    }

    /// Lease an endpoint for one link pipeline: the primary when it has a free
    /// slot, else the secondary, else wait for whichever frees first. Without a
    /// secondary endpoint this is the primary with no permit (the scheduler's
    /// active limit already bounds it).
    pub async fn acquire_link_endpoint(&self) -> EndpointLease {
        let Some(sec) = &self.secondary else {
            return EndpointLease { secondary: None, label: "primary".to_string(), _permit: None };
        };
        loop {
            if let Ok(permit) = sec.primary_permits.clone().try_acquire_owned() {
                return EndpointLease {
                    secondary: None,
                    label: "primary".to_string(),
                    _permit: Some(permit),
                };
            }
            if let Ok(permit) = sec.secondary_permits.clone().try_acquire_owned() {
                return EndpointLease {
                    secondary: Some(sec.llm.clone()),
                    label: sec.label.clone(),
                    _permit: Some(permit),
                };
            }
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }
    }

    pub fn primary(&self) -> &LLM {
        &self.primary
    }

    pub fn triage(&self) -> Option<&LLM> {
        self.triage.as_ref()
    }

    /// Whether `phase` runs on the triage model.
    pub fn is_triaged(&self, phase: TieredPhase) -> bool {
        self.triage.is_some() && self.phases.contains(&phase)
    }

    pub fn for_phase(&self, phase: TieredPhase) -> &LLM {
        match &self.triage {
            Some(triage) if self.phases.contains(&phase) => triage,
            _ => &self.primary,
        }
    }

    /// The specification phase a link belongs to.
    pub fn spec_phase(historical_id: i32) -> TieredPhase {
        match is_target_link(historical_id) {
            true => TieredPhase::TargetSpec,
            false => TieredPhase::Spec,
        }
    }

    fn triaged_phases(&self) -> String {
        match self.phases.is_empty() {
            true => "no phase".to_string(),
            false => self
                .phases
                .iter()
                .filter_map(|phase| phase.to_possible_value())
                .map(|value| value.get_name().to_string())
                .collect::<Vec<_>>()
                .join(", "),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        tiers: ModelTierArgs,
    }

    #[test]
    fn default_phases_leave_target_hypotheses_and_survivors_on_the_primary_model() {
        let cli = Cli::parse_from(["test"]);
        let phases: BTreeSet<TieredPhase> = cli.tiers.triage_phases.into_iter().collect();
        assert_eq!(
            phases,
            BTreeSet::from([
                TieredPhase::Extract,
                TieredPhase::Mapper,
                TieredPhase::Novelty,
                TieredPhase::Spec,
            ])
        );
        for kept in [
            TieredPhase::TargetSpec,
            TieredPhase::Codegen,
            TieredPhase::Regen,
            TieredPhase::Review,
        ] {
            assert!(!phases.contains(&kept), "{kept:?} must stay on the primary model");
        }
    }

    #[test]
    fn phases_parse_from_a_comma_list() {
        let cli = Cli::parse_from(["test", "--triage-phases", "spec,target-spec,codegen"]);
        assert_eq!(
            cli.tiers.triage_phases,
            vec![TieredPhase::Spec, TieredPhase::TargetSpec, TieredPhase::Codegen]
        );
    }

    #[test]
    fn a_negative_historical_id_is_a_target_spec() {
        assert_eq!(ModelTiers::spec_phase(-7), TieredPhase::TargetSpec);
        assert_eq!(ModelTiers::spec_phase(7), TieredPhase::Spec);
    }
}
