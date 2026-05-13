"""Pipeline configuration presets.

Usage in any module:
    from config import CFG

Select preset via env var FRAUD_PRESET (default: "full"):
    FRAUD_PRESET=fast  docker exec fraud-app python /app/run_all.py

Presets
-------
fast   Short GNN run, sampled betweenness — suitable for demo / CI.
       Total runtime: ~1.5 min (vs ~5 min for full).

full   Benchmarked production-like settings.
       PageRank converges in 2 iterations; betweenness sampled at 100
       nodes (142x faster than exact, negligible quality loss on PaySim).
"""
import os

_PRESETS = {
    "fast": {
        # GDS
        "louvain_max_levels":      1,
        "pagerank_max_iterations": 5,
        "pagerank_damping":        0.85,
        "betweenness_sampling":    50,    # faster, less accurate
        # GNN
        "gnn_epochs":              30,
        "gnn_hidden_dim":          32,
        "gnn_lr":                  0.01,
        "gnn_dropout":             0.3,
        # Chat agent
        "agent_max_retries":       2,
    },
    "full": {
        # GDS — values validated by benchmark (benchmark_report.md)
        "louvain_max_levels":      1,     # identical modularity to maxLevels=10, 5x faster
        "pagerank_max_iterations": 5,     # converges in 2 iterations on this graph
        "pagerank_damping":        0.85,
        "betweenness_sampling":    100,   # 142x speedup vs exact, fraud-use-case quality OK
        # GNN
        "gnn_epochs":              150,
        "gnn_hidden_dim":          64,
        "gnn_lr":                  0.005,
        "gnn_dropout":             0.3,
        # Chat agent
        "agent_max_retries":       2,
    },
}

_preset_name = os.environ.get("FRAUD_PRESET", "full").lower()
if _preset_name not in _PRESETS:
    raise ValueError(f"Unknown FRAUD_PRESET={_preset_name!r}. Choose: {list(_PRESETS)}")

CFG = _PRESETS[_preset_name]
PRESET_NAME = _preset_name
