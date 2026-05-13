# Demo Rehearsal Script — Fraud Graph Demo
## Wimbee — Data Graph Engineer Interview

**Total demo time:** 5–6 minutes  
**Format:** screen share, live Neo4j Browser + terminal

---

## Pre-Flight Checklist (do this 10 min before)

```bash
# 1 — Verify containers running
docker compose ps
# Both should show "healthy" / "running"

# 2 — Quick smoke check (Neo4j responsive)
docker exec fraud-app cypher-shell -u neo4j -p $NEO4J_PASSWORD \
  "MATCH (n) RETURN labels(n), count(n)"

# 3 — Verify fraudProb is populated (GNN ran)
docker exec fraud-app cypher-shell -u neo4j -p $NEO4J_PASSWORD \
  "MATCH (a:Account) WHERE a.fraudProb IS NOT NULL RETURN count(a)"
# Expect: 78499

# 4 — Open Neo4j Browser: http://localhost:7474
# Pre-load all 6 queries below as Favourites (★ button)
# so you can click instead of type during demo

# 5 — Open terminal for chat demo (do NOT run yet)
docker exec -it fraud-app python /app/chat.py
```

> **Fallback:** If containers are down — `bash run.sh` takes ~5 min.  
> If GNN not run — ask interviewer for 3 min while it trains.  
> If chat is slow — warn upfront: "Ollama Cloud adds ~10s per LLM call; local inference is sub-1s."

---

## Demo Flow

---

### Segment 1 — Hook + Graph Model (30s)

**Say:**
> "I built an end-to-end fraud detection system on Neo4j using the PaySim synthetic dataset — 50,000 transactions, 78,000 accounts. The key insight is that fraud is a network problem. A single suspicious account tells you little. The 4-hop chain it sits in tells you everything."

**Do in Neo4j Browser — run Query 1:**
```cypher
CALL db.schema.visualization()
```

> "Two node types: Accounts and Transactions. The bipartite structure — Account SENT Transaction RECEIVED_BY Account — preserves all metadata at the transaction level while enabling graph traversal across the account network."

---

### Segment 2 — Why Graph? The SQL Comparison (60s)

**Say:**
> "Let me show you the money mule chain detection. A→B→C→CASH_OUT. This is the layering pattern — funds pass through intermediaries to obscure origin."

**Do — run Query 2:**
```cypher
MATCH path =
  (origin:Account)-[:SENT]->(t1:Transaction)-[:RECEIVED_BY]->
  (mid:Account)-[:SENT]->(t2:Transaction)-[:RECEIVED_BY]->
  (cashout:Account)-[:SENT]->(t3:Transaction)-[:RECEIVED_BY]->(:Account)
WHERE t1.type = 'TRANSFER'
  AND t2.type = 'TRANSFER'
  AND t3.type IN ['CASH_OUT','TRANSFER']
  AND origin.id <> cashout.id
RETURN origin.id AS origin, mid.id AS mule, cashout.id AS cashout,
       t1.amount AS step1, t2.amount AS step2, t3.amount AS step3
ORDER BY step1 DESC LIMIT 10
```

> "Neo4j traverses this 4-hop pattern in milliseconds. In PostgreSQL this requires 4 self-joins, a subquery, and a temp table — and it gets slower as you add hops. Graph databases are pointer-based: each traversal step is a direct pointer follow, O(1) per hop regardless of total dataset size."

**Point at results:**
> "Notice how amounts decrease at each hop — step1 > step2 > step3. The mule takes a cut at each layer. This is the layering signature of trade-based money laundering."

---

### Segment 3 — GDS Fraud Ring Isolation (60s)

**Say:**
> "Now I run five GDS algorithms on a projected Account-to-Account graph — the virtual direct edges between accounts derived from their transaction paths. Louvain community detection finds dense clusters."

**Do — run Query 3:**
```cypher
MATCH (a:Account)
WHERE a.community IS NOT NULL
WITH a.community AS community,
     count(a) AS total,
     sum(CASE WHEN a.flagVelocity OR a.flagMule OR a.flagDrain THEN 1 ELSE 0 END) AS fraudAccounts
WHERE fraudAccounts > 2
RETURN community, total, fraudAccounts,
       round(fraudAccounts * 100.0 / total, 1) AS fraud_pct
ORDER BY fraud_pct DESC LIMIT 10
```

> "Community 585 — 68 accounts, 8 flagged, 12% fraud density. In real banking data a community like this triggers an entire ring investigation, not just account-level freezes. WCC gives me the isolated components — fraud rings that have no connection to the legitimate transaction network."

**Do — run Query 4 (PageRank):**
```cypher
MATCH (a:Account)
WHERE a.pageRank IS NOT NULL
  AND (a.flagVelocity OR a.flagMule OR a.flagDrain)
RETURN a.id AS account,
       round(a.pageRank, 3) AS pageRank,
       CASE WHEN a.flagVelocity THEN 'velocity ' ELSE '' END +
       CASE WHEN a.flagMule     THEN 'mule '     ELSE '' END +
       CASE WHEN a.flagDrain    THEN 'drain'      ELSE '' END AS flags
ORDER BY pageRank DESC LIMIT 10
```

> "High PageRank + mule flag = confirmed money coordinator. These are the accounts to freeze first — collapsing one coordinator disrupts the entire ring."

---

### Segment 4 — GNN: Hidden Accomplices (60s)

**Say:**
> "Rules catch known patterns. GraphSAGE catches accounts that look clean in isolation but are structurally embedded in fraud rings."

**Do — run Query 5:**
```cypher
MATCH (a:Account)
WHERE a.fraudProb > 0.7
  AND NOT (a.flagVelocity OR a.flagMule OR a.flagDrain)
RETURN a.id                    AS account,
       round(a.fraudProb, 4)   AS gnn_score,
       round(a.pageRank, 3)    AS pageRank,
       a.community             AS community,
       a.wccComponent          AS wcc_ring
ORDER BY gnn_score DESC LIMIT 10
```

> "These accounts have no rule flags — they would pass any rule-based filter. But the GNN sees their 3-hop neighbourhood: they sit inside fraud-dense communities, connected to high-PageRank coordinators, in isolated WCC components. GraphSAGE is inductive — it learns an aggregation function, not fixed embeddings. New accounts can be scored immediately without retraining."

**If asked about the perfect F1 score:**
> "In this demo setup, rule flags are both GNN input features and part of the label definition — so the model can reproduce rule predictions exactly. In production I'd separate concerns: GNN features are graph topology only (pageRank, betweenness, neighbourhood structure); rule flags enter as ensemble signals after GNN scoring, not as training features. That gives the GNN genuine predictive lift over the rules."

---

### Segment 5 — NL Chat: LangGraph Agent (90s)

**Say:**
> "The chat interface uses a LangGraph self-healing agent. Four nodes: generate Cypher, execute against Neo4j, fix on error, interpret results. If the LLM produces invalid Cypher — wrong property name, syntax error — the fix node gets the error message plus the original query and repairs it automatically. Up to 2 retries before graceful failure."

**Do — switch to terminal, type:**
```
Question: What is the most fraudulent account and why?
```

*While waiting for LLM (~10-15s):*
> "The Ollama Cloud round-trip adds latency here — in production this would be a locally-hosted quantized model, sub-1 second. The architecture is model-agnostic; swap the LLM by changing one env var."

*When Cypher appears:*
> "The LLM returned all relevant fraud signals — fraudProb, all three flags, pageRank, betweenness, community. It followed the 'why' rule in the prompt."

*When answer appears:*
> "Account C305956031 — fraudProb 0.996, flagDrain. The plain-English interpretation uses domain language: smash-and-grab, account takeover. A non-technical fraud analyst can read this directly. The full interaction — question, Cypher, results, answer, timings — is logged to logs/chat_*.log automatically."

**Type:**
```
quit
```

---

### Segment 6 — Architecture + Production Path (60s)

**Say:**
> "The pipeline runs end-to-end in a single command — bash run.sh. It also has a config preset system: FRAUD_PRESET=fast for a 1.5-minute demo run with lighter GNN and looser thresholds, FRAUD_PRESET=full for the production-calibrated run."

**Do — show architecture diagram or terminal:**
```bash
cat run.sh
```

> "For production scale: replace the CSV ingest with a Kafka consumer — transactions stream directly into Neo4j as they arrive. GDS algorithms run incrementally on the updated subgraph. The GNN scores new accounts on demand using the inductive property. The NL interface stays identical — analysts query the live graph the same way."

> "The betweenness centrality uses sampling — 142× faster than exact computation with negligible quality loss for fraud use cases. The benchmark is in benchmark_report.md. All algorithm config choices are justified by the benchmarks."

---

## Expected Interview Questions

### "Why Neo4j over a relational database?"

> "Three reasons. Traversal performance: a 4-hop mule chain query is O(hops × average degree) in Neo4j, O(n⁴) in SQL with self-joins. Schema flexibility: fraud patterns evolve — adding a new relationship type doesn't require a migration. Native graph algorithms: GDS runs Louvain, PageRank, WCC directly on the graph without exporting to a separate framework."

---

### "What's the difference between GraphSAGE and GCN?"

> "GCN is transductive — it learns fixed embeddings for every node in the training graph. Add a new node after training and you need to retrain. GraphSAGE learns an aggregation function — given a node's feature vector and its neighbourhood, it produces an embedding. New accounts can be scored immediately. That's critical for real-time fraud detection where new accounts appear continuously."

---

### "Why LangGraph instead of a simple retry loop?"

> "Three things a state machine gives you that a loop doesn't. First, routing is declarative — the graph topology defines control flow, separate from node logic. Adding a 'validate_cypher' node before execution is one line. Second, the state is fully inspectable — every field (cypher, error, retries, timing) is in the TypedDict state object, loggable and auditable. Third, it composes naturally with other LangGraph patterns — you could add human-in-the-loop approval for high-risk queries without restructuring the agent."

---

### "How would this handle real-time streaming?"

> "Replace ingest.py with a Kafka consumer that writes to Neo4j via the bolt driver on each message. GDS algorithms are the expensive part — run them periodically (every 5 minutes) on the updated account subgraph using GDS delta streaming rather than full recomputation. The GNN scores individual accounts on demand using the inductive property. Neo4j supports read replicas for the query workload."

---

### "What's the class imbalance problem and how did you handle it?"

> "PaySim has ~1.3% fraud rate — 74 clean accounts for every fraud account. Standard cross-entropy loss would learn to predict 'clean' for everything and get 98.7% accuracy. I use weighted cross-entropy with pos_weight = num_clean / num_fraud, which penalises missed fraud detections ~74× more than false positives. Recall is the primary metric in fraud — a missed fraud is far more costly than a false positive."

---

### "What would you change to take this to production?"

> "Four things. First, separate GNN features from rule flags to avoid label leakage — features are graph topology only. Second, add a model registry with versioning and A/B testing of GNN checkpoints. Third, replace Ollama Cloud with a locally-hosted quantized model for sub-1s NL inference. Fourth, add streaming ingest via Kafka and incremental GDS algorithm updates on the affected subgraph rather than full recomputation."

---

## Quick Reference Queries (save as Neo4j Browser Favourites)

| # | Label | Query purpose |
|---|-------|--------------|
| Q1 | `schema` | `CALL db.schema.visualization()` |
| Q2 | `mule chain` | 4-hop A→B→C→CASH_OUT pattern |
| Q3 | `communities` | Fraud-dense Louvain clusters |
| Q4 | `pagerank` | High-influence flagged accounts |
| Q5 | `hidden accounts` | GNN flagged, rules missed |
| Q6 | `top fraudsters` | `ORDER BY fraudProb DESC LIMIT 10` |

**Q6 full query:**
```cypher
MATCH (a:Account)
WHERE a.fraudProb IS NOT NULL
RETURN a.id                  AS account,
       round(a.fraudProb, 4) AS gnn_score,
       round(a.pageRank, 3)  AS pageRank,
       a.community           AS community,
       CASE WHEN a.flagVelocity THEN 'velocity ' ELSE '' END +
       CASE WHEN a.flagMule     THEN 'mule '     ELSE '' END +
       CASE WHEN a.flagDrain    THEN 'drain'      ELSE '' END AS rule_flags
ORDER BY gnn_score DESC LIMIT 10
```

---

## Timing Cheatsheet

```
00:00 – 00:30   Segment 1  Hook + graph model
00:30 – 01:30   Segment 2  Mule chain / SQL comparison
01:30 – 02:30   Segment 3  GDS communities + PageRank
02:30 – 03:30   Segment 4  GNN hidden accomplices
03:30 – 05:00   Segment 5  NL chat agent (allow 90s for LLM latency)
05:00 – 06:00   Segment 6  Architecture + production path
```

If interviewer asks questions during the demo — answer and skip Segment 6 if running short. The chat demo (Segment 5) is the highest-value moment; protect that time.
