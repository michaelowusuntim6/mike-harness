//! Exact DeepSeek token counting for context accounting.
//!
//! The rest of the workspace measures text with `ceil(bytes / 4)`
//! (`codex-utils-string`), which is fine for deriving a byte budget but is a
//! poor proxy for what DeepSeek charges: measured on this repository's own
//! source the heuristic over-counts prose by up to 10% and under-counts
//! identifier-dense code by 30%, and a 30% under-count means auto-compaction
//! fires a third too late.
//!
//! This crate embeds DeepSeek's published HuggingFace tokenizer, so anything
//! that compares a number against a window or compaction threshold can use the
//! same vocabulary the bridge sends with. The Python side
//! (`backend/server/token_counter.py`) reads the identical asset, and both
//! suites assert `assets/tokenizer_parity.json`, which is what keeps the two
//! implementations from drifting apart.
//!
//! Truncation deliberately keeps using the cheap heuristic: those sites turn a
//! token budget *into* a byte cap, a BPE count cannot be inverted, and they run
//! on multi-hundred-kilobyte tool output where accuracy buys nothing.

use std::num::NonZeroUsize;
use std::sync::LazyLock;
use std::sync::Mutex;

use lru::LruCache;
use sha1::Digest;
use sha1::Sha1;
use tokenizers::Tokenizer;

/// Embedded so a checkout is self-contained and cannot fall back silently.
const TOKENIZER_JSON: &str = include_str!("../../../../assets/tokenizer.json");

/// Name of the loaded vocabulary, matching `token_counter.VOCAB` in Python.
pub const VOCAB: &str = "deepseek-bpe-128k";

/// Per-text token cache size. A 400k-token window holds far fewer than this many
/// distinct items, and tool output is capped at 20 KB, so memory stays bounded.
const CACHE_ITEMS: usize = 2048;

static TOKENIZER: LazyLock<Option<Tokenizer>> =
    LazyLock::new(|| Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes()).ok());

/// Counts are memoized by content digest: history items are re-counted every turn,
/// and re-running BPE over a whole 400k-token context would cost seconds.
static CACHE: LazyLock<Mutex<LruCache<[u8; 20], i64>>> = LazyLock::new(|| {
    Mutex::new(LruCache::new(
        NonZeroUsize::new(CACHE_ITEMS).unwrap_or(NonZeroUsize::MIN),
    ))
});

fn digest(text: &str) -> [u8; 20] {
    let mut hasher = Sha1::new();
    hasher.update(text.as_bytes());
    let finished = hasher.finalize();
    let mut key = [0u8; 20];
    key.copy_from_slice(finished.as_slice());
    key
}

/// True when the embedded tokenizer loaded; otherwise every count falls back to
/// the byte heuristic and the numbers stay as good as they were before.
pub fn is_available() -> bool {
    TOKENIZER.is_some()
}

/// Which counter is live, for status lines and logs.
pub fn tokenizer_name() -> &'static str {
    if is_available() { VOCAB } else { "unavailable" }
}

/// Exact count, or `None` when the tokenizer is unavailable.
pub fn exact_token_count(text: &str) -> Option<i64> {
    let tokenizer = TOKENIZER.as_ref()?;
    let encoding = tokenizer.encode(text, /*add_special_tokens*/ false).ok()?;
    Some(encoding.len() as i64)
}

/// Exact count when possible, else `ceil(bytes / 4)`. Use this wherever a number
/// is compared against a window or compaction threshold.
pub fn token_count_or_approx(text: &str) -> i64 {
    exact_token_count(text).unwrap_or_else(|| codex_utils_string::approx_token_count(text) as i64)
}

/// [`token_count_or_approx`] memoized by content digest. Repeat strings are free,
/// so a turn pays only for the items it actually added.
pub fn count_cached(text: &str) -> i64 {
    if text.is_empty() {
        return 0;
    }
    let key = digest(text);
    let mut cache = CACHE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(count) = cache.get(&key) {
        return *count;
    }
    let count = token_count_or_approx(text);
    cache.put(key, count);
    count
}

#[cfg(test)]
mod parity_tests;
