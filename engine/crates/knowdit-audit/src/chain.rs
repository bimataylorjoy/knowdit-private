//! Read-only access to the deployed system the engagement insures.
//!
//! A bounty program pays for what happens to deployed state, and most of the
//! claims a specification makes about that state are decidable by reading it:
//! whether any live position is configured the way the setup needs, how much a
//! pool actually holds, which implementation sits behind a proxy, whether an
//! external dependency the sequence leans on behaves the way the stand-in
//! pretends. Without this, the specification agent can only read code, and the
//! precondition it writes is a guess — measured on one engagement, every
//! candidate that survived the harness and died in manual review died on one of
//! those reads (no live instance, a cent-sized balance, a dependency whose real
//! `claim` deletes what the mock left behind).
//!
//! Every read goes through `cast`, pinned to the block the run's harnesses fork
//! at, so what the agent observes here is the state its proof will stand on.
//! The endpoint URL never leaves this module: hosted RPC URLs carry their key,
//! and everything handed back to a model or a log is scrubbed of it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use alloy_primitives::{I256, U256};
use color_eyre::eyre::{Result, WrapErr, eyre};

use crate::harness::solidity::DeploymentBook;
use crate::scope::{ProgramScope, ScopeAsset, same_contract};
use crate::types::{Comparison, LiveCheck, LiveInstance};

/// EIP-1967 slots, read by [`ChainReader::account`] so an agent sees which
/// implementation actually runs behind an address before it reasons about it.
const EIP1967_SLOTS: [(&str, &str); 3] = [
    (
        "implementation",
        "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc",
    ),
    (
        "admin",
        "0xb53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103",
    ),
    (
        "beacon",
        "0xa3f0ad74e5423aebfd80d3ef4346578335a9a72aeaee59ff6cb3582b35133d50",
    ),
];

/// How a run reaches its chains. Built from the harness's own fork flags so
/// the agent and the harness read the same endpoint at the same block.
#[derive(Debug, Clone)]
pub struct ChainReaderOptions {
    /// The `cast` binary; `None` resolves `cast` next to `forge_bin`, then on
    /// `PATH`.
    pub cast_bin: Option<PathBuf>,
    /// The harness's `forge`, whose sibling `cast` is the one to prefer: both
    /// ship in one Foundry release.
    pub forge_bin: Option<PathBuf>,
    pub rpc_urls: Vec<String>,
    /// Block to pin every read to. `None` reads the endpoint's head once at
    /// connect time and pins to it.
    pub fork_block: Option<u64>,
    pub timeout: Duration,
    /// Cap on one answer handed back to a model.
    pub max_output_chars: usize,
    /// Widest block range one log query may span.
    pub max_log_span: u64,
    /// Most log entries one query returns.
    pub max_log_entries: usize,
}

impl Default for ChainReaderOptions {
    fn default() -> Self {
        Self {
            cast_bin: None,
            forge_bin: None,
            rpc_urls: Vec::new(),
            fork_block: None,
            timeout: Duration::from_secs(30),
            max_output_chars: 6_000,
            max_log_span: 50_000,
            max_log_entries: 40,
        }
    }
}

#[derive(Debug, Clone)]
struct ChainEndpoint {
    url: String,
    block: Option<u64>,
}

#[derive(Debug, Clone)]
enum ChainBackend {
    Cast { bin: PathBuf },
    /// Canned answers keyed by [`ChainQuery::key`]. Used by the regression
    /// suite, which has to replay what a chain said without reaching one.
    Fixture { responses: BTreeMap<String, String> },
}

/// One read against a chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainQuery {
    Call {
        address: String,
        signature: String,
        args: Vec<String>,
    },
    Storage {
        address: String,
        slot: String,
    },
    Logs {
        address: String,
        event: String,
        from_block: u64,
        to_block: u64,
    },
    CodeSize {
        address: String,
    },
    Balance {
        address: String,
    },
}

impl ChainQuery {
    /// Stable identity of a read, independent of the endpoint and block —
    /// what a fixture is keyed on.
    pub fn key(&self, chain_id: u64) -> String {
        match self {
            Self::Call {
                address,
                signature,
                args,
            } => format!(
                "{chain_id}:call:{}:{}:{}",
                address.to_lowercase(),
                signature.replace(' ', ""),
                args.join(",")
            ),
            Self::Storage { address, slot } => {
                format!("{chain_id}:storage:{}:{}", address.to_lowercase(), slot.to_lowercase())
            }
            Self::Logs {
                address,
                event,
                from_block,
                to_block,
            } => format!(
                "{chain_id}:logs:{}:{}:{from_block}:{to_block}",
                address.to_lowercase(),
                event.replace(' ', "")
            ),
            Self::CodeSize { address } => format!("{chain_id}:codesize:{}", address.to_lowercase()),
            Self::Balance { address } => format!("{chain_id}:balance:{}", address.to_lowercase()),
        }
    }

    /// `cast` argv for this read, without the endpoint.
    fn cast_args(&self, block: Option<u64>) -> Vec<String> {
        let mut argv: Vec<String> = match self {
            Self::Call {
                address,
                signature,
                args,
            } => {
                let mut argv = vec!["call".to_string(), address.clone(), signature.clone()];
                argv.extend(args.iter().cloned());
                argv
            }
            Self::Storage { address, slot } => {
                vec!["storage".to_string(), address.clone(), slot.clone()]
            }
            Self::Logs {
                address,
                event,
                from_block,
                to_block,
            } => {
                // Logs carry their own range; `--block` does not apply.
                return vec![
                    "logs".to_string(),
                    "--from-block".to_string(),
                    from_block.to_string(),
                    "--to-block".to_string(),
                    to_block.to_string(),
                    "--address".to_string(),
                    address.clone(),
                    event.clone(),
                    "--json".to_string(),
                ];
            }
            Self::CodeSize { address } => vec!["codesize".to_string(), address.clone()],
            Self::Balance { address } => vec!["balance".to_string(), address.clone()],
        };
        if let Some(block) = block {
            argv.push("--block".to_string());
            argv.push(block.to_string());
        }
        argv
    }

    /// Reject malformed input before it reaches a process. `cast` is spawned
    /// without a shell, so this is about useful errors, not injection.
    fn validate(&self) -> std::result::Result<(), String> {
        let address = match self {
            Self::Call { address, .. }
            | Self::Storage { address, .. }
            | Self::Logs { address, .. }
            | Self::CodeSize { address }
            | Self::Balance { address } => address,
        };
        ChainReader::check_address(address)?;
        match self {
            Self::Call { signature, args, .. } => {
                let open = signature.find('(');
                if open.is_none_or(|i| i == 0) || !signature.ends_with(')') {
                    return Err(format!(
                        "signature `{signature}` must look like `name(argTypes)(returnTypes)`, \
                         e.g. `balanceOf(address)(uint256)`"
                    ));
                }
                if args.len() > 12 {
                    return Err("at most 12 call arguments".to_string());
                }
                Ok(())
            }
            Self::Storage { slot, .. } => {
                let ok = slot.parse::<U256>().is_ok();
                ok.then_some(())
                    .ok_or_else(|| format!("slot `{slot}` is not a number or 0x-hex word"))
            }
            Self::Logs {
                event,
                from_block,
                to_block,
                ..
            } => {
                if !event.contains('(') {
                    return Err(format!(
                        "event `{event}` must be a signature like `Transfer(address,address,uint256)`"
                    ));
                }
                (from_block <= to_block)
                    .then_some(())
                    .ok_or_else(|| "from_block is after to_block".to_string())
            }
            Self::CodeSize { .. } | Self::Balance { .. } => Ok(()),
        }
    }
}

/// Read-only chain access for one run.
#[derive(Debug, Clone)]
pub struct ChainReader {
    backend: ChainBackend,
    endpoints: BTreeMap<u64, ChainEndpoint>,
    book: Arc<DeploymentBook>,
    assets: Vec<ScopeAsset>,
    /// The engagement's chain when the scope names one, else the lowest
    /// configured chain — the same deterministic choice the harness makes.
    default_chain: Option<u64>,
    options: ChainReaderOptions,
}

impl ChainReader {
    /// Probe every configured endpoint and pin each to a block. `None` when the
    /// run configured no endpoint: the tools are then simply not offered, and
    /// the gates that need a chain fall back to requiring an argument instead
    /// of an observation.
    pub async fn connect(
        options: ChainReaderOptions,
        repo_root: &Path,
        scope: Option<&ProgramScope>,
    ) -> Result<Option<Self>> {
        if options.rpc_urls.is_empty() {
            return Ok(None);
        }
        let bin = Self::resolve_cast(&options)?;
        let mut reader = Self {
            backend: ChainBackend::Cast { bin },
            endpoints: BTreeMap::new(),
            book: Arc::new(DeploymentBook::load(repo_root).await),
            assets: scope.map(|s| s.assets.clone()).unwrap_or_default(),
            default_chain: None,
            options: options.clone(),
        };
        for url in &options.rpc_urls {
            let chain_id = reader
                .raw(url, &["chain-id".to_string()])
                .await
                .map_err(|error| eyre!("chain endpoint probe failed: {error}"))?
                .trim()
                .parse::<u64>()
                .wrap_err("chain endpoint returned a non-numeric chain id")?;
            let block = match options.fork_block {
                Some(block) => Some(block),
                None => reader
                    .raw(url, &["block-number".to_string()])
                    .await
                    .ok()
                    .and_then(|text| text.trim().parse::<u64>().ok()),
            };
            if reader.endpoints.contains_key(&chain_id) {
                tracing::warn!(chain_id, "two chain endpoints serve one chain; keeping the first");
                continue;
            }
            tracing::info!(
                chain_id,
                block = ?block,
                "[chain] read-only endpoint connected"
            );
            reader.endpoints.insert(chain_id, ChainEndpoint { url: url.clone(), block });
        }
        reader.default_chain = scope
            .and_then(|s| s.chain_id)
            .filter(|id| reader.endpoints.contains_key(id))
            .or_else(|| reader.endpoints.keys().next().copied());
        Ok(Some(reader))
    }

    /// A reader that answers from canned responses instead of a chain. The
    /// regression suite uses it to replay what a chain said when a finding was
    /// falsified, so the gate is tested on the observation, not on a network.
    pub fn fixture(
        chain_id: u64,
        block: Option<u64>,
        responses: BTreeMap<String, String>,
        assets: Vec<ScopeAsset>,
    ) -> Self {
        let mut endpoints = BTreeMap::new();
        endpoints.insert(
            chain_id,
            ChainEndpoint {
                url: String::new(),
                block,
            },
        );
        Self {
            backend: ChainBackend::Fixture { responses },
            endpoints,
            book: Arc::new(DeploymentBook::default()),
            assets,
            default_chain: Some(chain_id),
            options: ChainReaderOptions::default(),
        }
    }

    fn resolve_cast(options: &ChainReaderOptions) -> Result<PathBuf> {
        if let Some(bin) = &options.cast_bin {
            return Ok(bin.clone());
        }
        let sibling = options
            .forge_bin
            .as_ref()
            .and_then(|forge| forge.parent().map(|dir| dir.join("cast")))
            .filter(|cast| cast.is_file());
        let on_path = || {
            std::env::var("PATH")
                .unwrap_or_default()
                .split(':')
                .map(|dir| Path::new(dir).join("cast"))
                .find(|cast| cast.is_file())
        };
        sibling.or_else(on_path).ok_or_else(|| {
            eyre!(
                "`cast` not found next to the forge binary or on PATH; the read-only chain tools \
                 need it (install Foundry, or pass --chain-cast-bin)"
            )
        })
    }

    /// One line per chain, for a prompt: which chains the agent can read and
    /// at which block.
    pub fn describe(&self) -> String {
        let mut out = String::new();
        for (chain_id, endpoint) in &self.endpoints {
            let block = endpoint
                .block
                .map(|b| format!("pinned at block {b}"))
                .unwrap_or_else(|| "at head (unpinned)".to_string());
            let default = if Some(*chain_id) == self.default_chain {
                " (default)"
            } else {
                ""
            };
            out.push_str(&format!("- chain {chain_id}{default}: {block}\n"));
        }
        out
    }

    pub fn block_for(&self, chain_id: u64) -> Option<u64> {
        self.endpoints.get(&chain_id).and_then(|e| e.block)
    }

    fn check_address(address: &str) -> std::result::Result<(), String> {
        let hex = address.strip_prefix("0x").unwrap_or("");
        (hex.len() == 40 && hex.chars().all(|c| c.is_ascii_hexdigit()))
            .then_some(())
            .ok_or_else(|| format!("`{address}` is not a 20-byte 0x address"))
    }

    pub fn chain_or_default(&self, chain_id: Option<u64>) -> std::result::Result<u64, String> {
        match chain_id.or(self.default_chain) {
            Some(id) if self.endpoints.contains_key(&id) => Ok(id),
            Some(id) => Err(format!(
                "chain {id} is not configured for this run; readable chains:\n{}",
                self.describe()
            )),
            None => Err("no chain is configured for this run".to_string()),
        }
    }

    /// Every address a name resolves to: the engagement's own asset list
    /// first (it is what the program insures), then the repository's
    /// deployment artifacts.
    pub fn candidates(&self, chain_id: u64, name: &str) -> Vec<(String, String, &'static str)> {
        let mut out: Vec<(String, String, &'static str)> = Vec::new();
        for asset in &self.assets {
            let chain_matches = asset.chain_id.is_none_or(|id| id == chain_id);
            if let Some(address) = &asset.address
                && chain_matches
                && (same_contract(&asset.name, name) || asset.name.eq_ignore_ascii_case(name))
            {
                out.push((asset.name.clone(), address.clone(), "scope"));
            }
        }
        for (artifact, address) in self.book.addresses_named(chain_id, name) {
            if !out.iter().any(|(_, a, _)| a.eq_ignore_ascii_case(&address)) {
                out.push((artifact, address, "deployments"));
            }
        }
        out
    }

    /// Resolve an address or a contract name to `(chain, address)`. A name
    /// that resolves to several addresses is refused rather than guessed: the
    /// caller has to pick one, because reading the wrong instance produces a
    /// confident wrong answer.
    pub fn resolve(
        &self,
        chain_id: Option<u64>,
        target: &str,
    ) -> std::result::Result<(u64, String), String> {
        let chain_id = self.chain_or_default(chain_id)?;
        let target = target.trim();
        if target.starts_with("0x") {
            Self::check_address(target)?;
            return Ok((chain_id, target.to_string()));
        }
        let found = self.candidates(chain_id, target);
        match found.as_slice() {
            [] => Err(format!(
                "`{target}` does not resolve to an address on chain {chain_id} (neither the scope's \
                 asset list nor the repository's deployment artifacts name it). Pass the address \
                 itself."
            )),
            [(_, address, _)] => Ok((chain_id, address.clone())),
            many => Err(format!(
                "`{target}` resolves to several addresses on chain {chain_id}; pass one explicitly:\n{}",
                many.iter()
                    .map(|(name, address, source)| format!("- {name} → {address} ({source})"))
                    .collect::<Vec<_>>()
                    .join("\n")
            )),
        }
    }

    /// Run one read, pinned to the chain's block. Errors come back as text
    /// scrubbed of the endpoint, ready to hand to a model.
    pub async fn run(&self, chain_id: u64, query: &ChainQuery) -> std::result::Result<String, String> {
        query.validate()?;
        let endpoint = self
            .endpoints
            .get(&chain_id)
            .ok_or_else(|| format!("chain {chain_id} is not configured for this run"))?;
        match &self.backend {
            ChainBackend::Fixture { responses } => responses
                .get(&query.key(chain_id))
                .cloned()
                .ok_or_else(|| format!("no recorded answer for `{}`", query.key(chain_id))),
            ChainBackend::Cast { .. } => {
                let out = self.raw(&endpoint.url, &query.cast_args(endpoint.block)).await?;
                Ok(self.truncate(out.trim()))
            }
        }
    }

    /// Spawn `cast` with `argv` against `url`.
    async fn raw(&self, url: &str, argv: &[String]) -> std::result::Result<String, String> {
        let ChainBackend::Cast { bin } = &self.backend else {
            return Err("this reader has no chain backend".to_string());
        };
        let mut command = tokio::process::Command::new(bin);
        command
            .args(argv)
            .arg("--rpc-url")
            .arg(url)
            .kill_on_drop(true)
            .stdin(std::process::Stdio::null());
        let output = tokio::time::timeout(self.options.timeout, command.output())
            .await
            .map_err(|_| format!("chain read timed out after {:?}", self.options.timeout))?
            .map_err(|error| format!("failed to spawn cast: {error}"))?;
        if output.status.success() {
            return Ok(String::from_utf8_lossy(&output.stdout).to_string());
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Keep the lines that say what went wrong; transport diagnostics
        // (headers, ray ids) are noise to a model and may echo the host.
        let reason: Vec<&str> = stderr
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .filter(|line| !line.contains(':') || line.starts_with("Error"))
            .take(3)
            .collect();
        let reason = if reason.is_empty() {
            stderr.lines().next().unwrap_or("cast failed").to_string()
        } else {
            reason.join(" ")
        };
        Err(self.scrub(&reason))
    }

    /// Remove every configured endpoint from `text`.
    fn scrub(&self, text: &str) -> String {
        let mut out = text.to_string();
        for endpoint in self.endpoints.values() {
            if endpoint.url.is_empty() {
                continue;
            }
            out = out.replace(&endpoint.url, "<rpc>");
            if let Some(host) = endpoint
                .url
                .split("://")
                .nth(1)
                .and_then(|rest| rest.split('/').next())
            {
                out = out.replace(host, "<rpc-host>");
            }
        }
        out
    }

    fn truncate(&self, text: &str) -> String {
        if text.chars().count() <= self.options.max_output_chars {
            return text.to_string();
        }
        let kept: String = text.chars().take(self.options.max_output_chars).collect();
        format!("{kept}\n… (truncated at {} chars)", self.options.max_output_chars)
    }

    /// Logs for one event on one address, bounded in range and count. The
    /// range defaults to the `max_log_span` blocks ending at the pinned block.
    pub async fn logs(
        &self,
        chain_id: u64,
        address: &str,
        event: &str,
        from_block: Option<u64>,
        to_block: Option<u64>,
    ) -> std::result::Result<String, String> {
        let head = self.block_for(chain_id);
        let to = match (to_block, head) {
            (Some(to), Some(head)) => to.min(head),
            (Some(to), None) => to,
            (None, Some(head)) => head,
            (None, None) => {
                return Err(
                    "this chain is not pinned to a block; pass an explicit to_block".to_string()
                );
            }
        };
        let span = self.options.max_log_span;
        let from = from_block
            .unwrap_or(to.saturating_sub(span))
            .max(to.saturating_sub(span));
        let query = ChainQuery::Logs {
            address: address.to_string(),
            event: event.to_string(),
            from_block: from,
            to_block: to,
        };
        let raw = self.run(chain_id, &query).await?;
        let entries: Vec<serde_json::Value> = serde_json::from_str(&raw).unwrap_or_default();
        let total = entries.len();
        let mut out = format!("{total} log(s) for `{event}` on {address} in blocks {from}..={to}");
        if from_block.is_some_and(|f| f < from) {
            out.push_str(&format!(" (range clamped to the last {span} blocks)"));
        }
        out.push('\n');
        for entry in entries.iter().rev().take(self.options.max_log_entries) {
            let block = entry
                .get("blockNumber")
                .and_then(|b| b.as_str())
                .and_then(|b| u64::from_str_radix(b.trim_start_matches("0x"), 16).ok())
                .map(|b| b.to_string())
                .unwrap_or_default();
            let tx = entry.get("transactionHash").and_then(|t| t.as_str()).unwrap_or("");
            let topics: Vec<&str> = entry
                .get("topics")
                .and_then(|t| t.as_array())
                .map(|t| t.iter().skip(1).filter_map(|x| x.as_str()).collect())
                .unwrap_or_default();
            let data = entry.get("data").and_then(|d| d.as_str()).unwrap_or("");
            out.push_str(&format!(
                "- block {block} tx {tx} topics [{}] data {data}\n",
                topics.join(", ")
            ));
        }
        if total > self.options.max_log_entries {
            out.push_str(&format!(
                "(showing the latest {} of {total})\n",
                self.options.max_log_entries
            ));
        }
        Ok(self.truncate(&out))
    }

    /// What lives at an address: whether it has code, its native balance, and
    /// what its EIP-1967 slots point at.
    pub async fn account(&self, chain_id: u64, address: &str) -> std::result::Result<String, String> {
        let codesize = self
            .run(chain_id, &ChainQuery::CodeSize { address: address.to_string() })
            .await?;
        let balance = self
            .run(chain_id, &ChainQuery::Balance { address: address.to_string() })
            .await
            .unwrap_or_else(|error| format!("unavailable ({error})"));
        let mut out = format!(
            "{address} on chain {chain_id}{}\n- code size: {codesize} bytes{}\n- native balance (wei): {balance}\n",
            self.block_for(chain_id)
                .map(|b| format!(" at block {b}"))
                .unwrap_or_default(),
            if codesize.trim() == "0" { " (an EOA or an undeployed address)" } else { "" },
        );
        for (label, slot) in EIP1967_SLOTS {
            let word = self
                .run(
                    chain_id,
                    &ChainQuery::Storage {
                        address: address.to_string(),
                        slot: slot.to_string(),
                    },
                )
                .await
                .unwrap_or_default();
            let value = word.trim().parse::<U256>().unwrap_or(U256::ZERO);
            if value.is_zero() {
                continue;
            }
            let hex = format!("{value:x}");
            out.push_str(&format!("- EIP-1967 {label}: 0x{hex:0>40}\n"));
        }
        Ok(out)
    }

    /// Execute a live-instance read and compare it with what the
    /// specification expects.
    pub async fn check(&self, instance: &LiveInstance) -> std::result::Result<LiveCheck, String> {
        let (chain_id, address) = self.resolve(instance.chain_id, &instance.target)?;
        let observed = self
            .run(
                chain_id,
                &ChainQuery::Call {
                    address: address.clone(),
                    signature: instance.signature.trim().to_string(),
                    args: instance.args.clone(),
                },
            )
            .await?;
        let holds = instance.comparison.holds(&observed, &instance.expected);
        Ok(LiveCheck {
            chain_id,
            address,
            block: self.block_for(chain_id),
            observed,
            holds,
        })
    }

    /// Whether `address` has code on `chain_id`. `None` when the read failed.
    pub async fn has_code(&self, chain_id: u64, address: &str) -> Option<bool> {
        self.run(chain_id, &ChainQuery::CodeSize { address: address.to_string() })
            .await
            .ok()
            .and_then(|size| size.trim().parse::<u64>().ok())
            .map(|size| size > 0)
    }
}

/// A value `cast` printed, read the way a comparison needs it. `cast`
/// annotates integers with a scientific rendering (`5000 [5e3]`), quotes
/// strings, and prints addresses and words as hex.
#[derive(Debug, Clone, PartialEq)]
enum Observed {
    Int(I256),
    Uint(U256),
    Bool(bool),
    Text(String),
}

impl Observed {
    fn parse(raw: &str) -> Self {
        let first = raw.trim().lines().next().unwrap_or("").trim();
        let token = first.split(" [").next().unwrap_or(first).trim();
        let token = token.trim_matches('"');
        match token {
            "true" => return Self::Bool(true),
            "false" => return Self::Bool(false),
            _ => {}
        }
        let plain = token.replace('_', "");
        if let Some(negative) = plain.strip_prefix('-')
            && negative.chars().all(|c| c.is_ascii_digit())
            && let Ok(value) = I256::from_dec_str(&plain)
        {
            return Self::Int(value);
        }
        let numeric = plain.chars().all(|c| c.is_ascii_digit())
            || (plain.starts_with("0x") && plain.len() <= 66);
        if numeric && let Ok(value) = plain.parse::<U256>() {
            return Self::Uint(value);
        }
        Self::Text(token.to_string())
    }

    fn is_zero(&self) -> bool {
        match self {
            Self::Int(v) => v.is_zero(),
            Self::Uint(v) => v.is_zero(),
            Self::Bool(b) => !b,
            Self::Text(t) => t.is_empty(),
        }
    }

    /// Ordering between two observations when both are numbers.
    fn order(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Self::Uint(a), Self::Uint(b)) => Some(a.cmp(b)),
            (Self::Int(a), Self::Int(b)) => Some(a.cmp(b)),
            (Self::Int(a), Self::Uint(b)) => I256::try_from(*b).ok().map(|b| a.cmp(&b)),
            (Self::Uint(a), Self::Int(b)) => I256::try_from(*a).ok().map(|a| a.cmp(b)),
            _ => None,
        }
    }
}

impl Comparison {
    /// Whether `observed` (as `cast` printed it) satisfies this comparison
    /// against `expected`. A comparison that cannot be evaluated — ordering
    /// two strings, say — does not hold: an undecidable precondition has not
    /// been shown to exist.
    pub fn holds(self, observed: &str, expected: &str) -> bool {
        let left = Observed::parse(observed);
        let right = Observed::parse(expected);
        use std::cmp::Ordering::*;
        match self {
            Self::Nonzero => !left.is_zero(),
            Self::Contains => observed
                .to_lowercase()
                .contains(&expected.trim().to_lowercase()),
            Self::Eq | Self::Ne => {
                let equal = match left.order(&right) {
                    Some(order) => order == Equal,
                    None => match (&left, &right) {
                        (Observed::Text(a), Observed::Text(b)) => a.eq_ignore_ascii_case(b),
                        _ => left == right,
                    },
                };
                (self == Self::Eq) == equal
            }
            Self::Gt => left.order(&right) == Some(Greater),
            Self::Gte => matches!(left.order(&right), Some(Greater | Equal)),
            Self::Lt => left.order(&right) == Some(Less),
            Self::Lte => matches!(left.order(&right), Some(Less | Equal)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_reads_cast_output() {
        assert!(Comparison::Gt.holds("50005723824792250 [5e16]", "1000"));
        assert!(Comparison::Eq.holds("0", "0x0"));
        assert!(Comparison::Nonzero.holds("\"USD Coin\"", ""));
        assert!(!Comparison::Nonzero.holds("0x0000000000000000000000000000000000000000", ""));
        assert!(Comparison::Lt.holds("-5", "3"));
        assert!(Comparison::Eq.holds("true", "true"));
        assert!(Comparison::Contains.holds("\"USD Coin\"", "usd"));
        // Ordering text is undecidable, so it does not hold.
        assert!(!Comparison::Gt.holds("abc", "abd"));
    }

    #[tokio::test]
    async fn fixture_answers_and_resolves_scope_names() {
        let address = "0x1111111111111111111111111111111111111111";
        let mut responses = BTreeMap::new();
        let query = ChainQuery::Call {
            address: address.to_string(),
            signature: "totalSupply()(uint256)".to_string(),
            args: Vec::new(),
        };
        responses.insert(query.key(1), "42 [4.2e1]".to_string());
        let reader = ChainReader::fixture(
            1,
            Some(100),
            responses,
            vec![ScopeAsset {
                name: "Vault".to_string(),
                chain_id: Some(1),
                address: Some(address.to_string()),
                value_usd: None,
            }],
        );
        let check = reader
            .check(&LiveInstance {
                chain_id: None,
                target: "VaultProxy".to_string(),
                signature: "totalSupply()(uint256)".to_string(),
                args: Vec::new(),
                comparison: Comparison::Gt,
                expected: "0".to_string(),
                note: String::new(),
                check: None,
            })
            .await
            .expect("fixture read");
        assert!(check.holds);
        assert_eq!(check.block, Some(100));
        assert!(reader.resolve(None, "Unknown").is_err());
    }

    #[test]
    fn scrub_hides_endpoint() {
        let mut reader = ChainReader::fixture(1, None, BTreeMap::new(), Vec::new());
        reader.endpoints.insert(
            5,
            ChainEndpoint {
                url: "https://rpc.example/v2/SECRET".to_string(),
                block: None,
            },
        );
        let text = reader.scrub("failed https://rpc.example/v2/SECRET and rpc.example");
        assert!(!text.contains("SECRET"));
        assert!(!text.contains("rpc.example"));
    }
}
