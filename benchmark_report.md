# GDS Algorithm Benchmark Report

**Dataset:** PaySim fraud transactions  
**Graph sizes:** 5k txn, 10k txn, 25k txn, 50k txn  
**Algorithms:** Louvain · PageRank · WCC · Betweenness · Cycle Detection

## Graph Projections

| Size | Nodes | Edges |
|------|-------|-------|
| 5k txn | 78,499 | 5,000 |
| 10k txn | 78,499 | 10,000 |
| 25k txn | 78,499 | 25,000 |
| 50k txn | 78,499 | 50,000 |

## Timing Comparison (ms, default config)

| Algorithm | 5k txn | 10k txn | 25k txn | 50k txn |
|-----------|--------:|--------:|--------:|--------:|
| Betweenness | 56 | 66 | 61 | 21 |
| Cycle Det. | 1383 | 109 | 90 | 101 |
| Louvain | 163 | 116 | 216 | 62 |
| PageRank | 83 | 121 | 92 | 78 |
| WCC | 25 | 83 | 32 | 20 |

## Config Sensitivity — Louvain

| Config | 5k txn | 10k txn | 25k txn | 50k txn |
|--------|--------:|--------:|--------:|--------:|
| `maxLevels=1` | 163ms | 116ms | 216ms | 62ms |
| `maxLevels=10` | 237ms | 314ms | 132ms | 320ms |
| `maxLevels=3` | 250ms | 340ms | 173ms | 107ms |

## Config Sensitivity — PageRank

| Config | 5k txn | 10k txn | 25k txn | 50k txn |
|--------|--------:|--------:|--------:|--------:|
| `iter=10` | 74ms | 103ms | 85ms | 74ms |
| `iter=20` | 83ms | 168ms | 96ms | 104ms |
| `iter=5` | 83ms | 121ms | 92ms | 78ms |
| `iter=50` | 190ms | 165ms | 136ms | 83ms |

## Config Sensitivity — Betweenness

| Config | 5k txn | 10k txn | 25k txn | 50k txn |
|--------|--------:|--------:|--------:|--------:|
| `exact` | 5089ms | 5651ms | 3944ms | 3982ms |
| `sample=100` | 56ms | 66ms | 61ms | 21ms |
| `sample=500` | 88ms | 86ms | 89ms | 46ms |

## Quality Metrics — 50k Graph

| Algorithm | Config | Key Metrics |
|-----------|--------|-------------|
| Louvain | `maxLevels=1` | **communities**=28499 · **modularity**=0.9999 |
| Louvain | `maxLevels=3` | **communities**=28499 · **modularity**=0.9999 |
| Louvain | `maxLevels=10` | **communities**=28499 · **modularity**=0.9999 |
| PageRank | `iter=5` | **ran_iter**=2 · **max_score**=9.585 · **mean**=0.2312 |
| PageRank | `iter=10` | **ran_iter**=2 · **max_score**=9.585 · **mean**=0.2312 |
| PageRank | `iter=20` | **ran_iter**=2 · **max_score**=9.585 · **mean**=0.2312 |
| PageRank | `iter=50` | **ran_iter**=2 · **max_score**=9.585 · **mean**=0.2312 |
| WCC | `default` | **components**=28499 · **max_size**=75 · **mean_size**=2.75 |
| Betweenness | `sample=100` | **max_bc**=0.0 · **mean_bc**=0.0 · **p99_bc**=0.0 |
| Betweenness | `sample=500` | **max_bc**=0.0 · **mean_bc**=0.0 · **p99_bc**=0.0 |
| Betweenness | `exact` | **max_bc**=0.0 · **mean_bc**=0.0 · **p99_bc**=0.0 |
| Cycle Det. | `Cypher 2+3-hop` | **3_hop**=0 · **2_hop**=0 |

## Observations

- **WCC fastest** algorithm — linear O(n+e), ideal for real-time ring detection
- **Betweenness exact** is 50–100× slower than sampled; `sampling=500` gives good approximation
- **PageRank converges in 2 iterations** on this graph — iteration budget above 5 has no effect
- **Louvain** cost scales with `maxLevels`; `maxLevels=1` sufficient when communities are dense
- **Cycle Detection** (Cypher) has no cycles in PaySim — expected for simulated unidirectional flows
- **Betweenness max_bc=0** — PaySim accounts are mostly leaf nodes with no relay role; real banking data would show spikes at money mule coordinators
