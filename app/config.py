"""Pipeline configuration presets.

Usage in any module:
    from config import CFG

Select preset via env var FRAUD_PRESET (default: "full"):
    FRAUD_PRESET=fast  docker exec fraud-app python /app/run_all.py

Presets
-------
fast   Short GNN run, looser fraud thresholds, smaller model — demo / CI.
       Total runtime: ~1.5 min (vs ~5 min for full).
       Looser velocity threshold catches more accounts — better for live demo
       when dataset slice is small.

full   Benchmarked production-like settings. All thresholds calibrated on
       full PaySim dataset. PageRank converges in 2 iterations; betweenness
       sampled at 100 nodes (142x faster than exact, negligible quality loss).

Threshold rationale
-------------------
velocity_tx_threshold / velocity_window:
    Business decision: how aggressive to flag card-testing.
    Tight (>5 txn) = fewer false positives. Loose (>2 txn) = higher recall.
    Tune per client risk appetite.

drain_pct:
    What fraction of balance drained counts as account-takeover.
    0.95 = must drain ≥95% (conservative, fewer FP).
    0.80 = drain ≥80% (aggressive, more recall).

gnn_fraud_threshold:
    Score cutoff for classifying an account as fraud.
    Lower = higher recall, more false positives.
    Higher = more precise, misses borderline cases.
    Tune using PR curve on validation set.

llm_model:
    None → falls back to OLLAMA_MODEL env var.
    Set explicitly to pin a model per preset independent of env.
"""
import os

_PRESETS = {
    "fast": {
        # ── GDS ──────────────────────────────────────────────────────────────
        "louvain_max_levels":      1,
        "pagerank_max_iterations": 5,
        "pagerank_damping":        0.85,
        "betweenness_sampling":    50,

        # ── Fraud rules ───────────────────────────────────────────────────────
        # Looser thresholds → more matches → better for demo with small dataset slice
        "velocity_tx_threshold":   2,    # flag accounts with >2 txn (vs >3 in full)
        "velocity_window":         15,   # within 15 steps (vs 10 in full)
        "drain_pct":               0.80, # drain ≥80% (vs ≥95% in full)

        # ── GNN ───────────────────────────────────────────────────────────────
        "gnn_epochs":              30,
        "gnn_hidden_dim":          32,
        "gnn_layers":              2,    # shallower = faster
        "gnn_lr":                  0.01,
        "gnn_dropout":             0.3,
        "gnn_fraud_threshold":     0.4,  # lower threshold → higher recall

        # ── Chat agent ────────────────────────────────────────────────────────
        "agent_max_retries":       2,
        "llm_model":               None, # fall back to OLLAMA_MODEL env var
    },
    "full": {
        # ── GDS — values validated by benchmark (benchmark_report.md) ─────────
        "louvain_max_levels":      1,     # identical modularity to maxLevels=10, 5x faster
        "pagerank_max_iterations": 5,     # converges in 2 iterations on this graph
        "pagerank_damping":        0.85,
        "betweenness_sampling":    100,   # 142x speedup vs exact

        # ── Fraud rules — calibrated on full PaySim (6.3M rows) ──────────────
        "velocity_tx_threshold":   3,     # >3 txn within window
        "velocity_window":         10,    # 10 time steps
        "drain_pct":               0.95,  # ≥95% balance drained

        # ── GNN — 150 epochs, 3-layer GraphSAGE, val AUC ~0.92 ───────────────
        "gnn_epochs":              150,
        "gnn_hidden_dim":          64,
        "gnn_layers":              3,
        "gnn_lr":                  0.005,
        "gnn_dropout":             0.3,
        "gnn_fraud_threshold":     0.5,   # standard decision boundary

        # ── Chat agent ────────────────────────────────────────────────────────
        "agent_max_retries":       2,
        "llm_model":               None, # fall back to OLLAMA_MODEL env var
    },
}

_preset_name = os.environ.get("FRAUD_PRESET", "full").lower()
if _preset_name not in _PRESETS:
    raise ValueError(f"Unknown FRAUD_PRESET={_preset_name!r}. Choose: {list(_PRESETS)}")

CFG = _PRESETS[_preset_name]
PRESET_NAME = _preset_name
