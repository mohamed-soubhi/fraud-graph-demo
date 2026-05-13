# Fraud Graph Demo

Fraud detection knowledge graph using **Neo4j**, **Python**, **GDS**, and **Ollama LLM**.  
Demonstrates graph engineering + AI integration for the Data Graph Engineer role.

---

## Architecture

```
PaySim CSV
    │
    ▼
ingest.py ──► Neo4j Graph (Accounts + Transactions)
                    │
          ┌─────────┼──────────┐
          ▼         ▼          ▼
   fraud_rules  gds_analysis  chat.py
   (Cypher)     (Louvain/     (LangChain
                 PageRank)     + Ollama)
```

---

## Prerequisites

- Docker + Docker Compose
- PaySim dataset (Kaggle — see T-02 in TASKS.md)
- Ollama paid API key

---

## Step-by-Step Setup

### Step 1 — Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set:
```
OLLAMA_API_KEY=your_key_here
NEO4J_PASSWORD=choose_a_password
```

---

### Step 2 — Download PaySim dataset

1. Go to: https://www.kaggle.com/datasets/ealaxi/paysim1
2. Download `PS_20174392719_1491204439457_log.csv`
3. Place file in `fraud-graph-demo/data/`

Verify:
```bash
wc -l data/*.csv
# expected: ~6,362,621 lines
```

---

### Step 3 — Build and start containers

```bash
docker compose up -d --build
```

Wait ~30 seconds for Neo4j to be ready. Verify:
```bash
docker compose ps
# both containers should show "healthy" / "running"
```

### Open Neo4j Browser

**URL:** http://localhost:7474

**First-time login:**
1. Open http://localhost:7474 in browser
2. Connection URL field: `bolt://localhost:7687`
3. Username: `neo4j`
4. Password: value of `NEO4J_PASSWORD` from your `.env`
5. Click **Connect**

> If connection refused: Neo4j needs ~30s to initialize. Run `docker compose logs -f neo4j` and wait for `"Started"` in the logs.

**Useful Browser shortcuts:**
| Action | How |
|--------|-----|
| Run query | Ctrl+Enter (or click ▶) |
| Clear editor | Ctrl+Shift+I |
| Style nodes by property | Click result → paintbrush icon → node label → color |
| Toggle graph / table view | Graph / Table / Text tabs on result panel |
| Expand a node's relationships | Double-click any node in graph view |
| Save a query as a favourite | Click ★ in the query bar |

**First queries to run after login:**
```cypher
// 1 — verify nodes loaded
MATCH (n) RETURN labels(n), count(n) ORDER BY count(n) DESC

// 2 — check schema
CALL db.schema.visualization()

// 3 — sample 25 transactions
MATCH (src:Account)-[:SENT]->(tx:Transaction)-[:RECEIVED_BY]->(dst:Account)
RETURN src.id, tx.type, tx.amount, dst.id, tx.isFraud LIMIT 25
```

---

### Step 4 — Load data into Neo4j

```bash
docker compose exec app python app/ingest.py
```

Expected output:
```
Loading PaySim data...
Creating indexes...
Loading 50000 transactions...
100%|████████████| 500/500 [02:15<00:00]
Done. Loaded 50000 transactions, 41832 accounts.
```

Verify in Neo4j Browser:
```cypher
MATCH (n) RETURN labels(n), count(n)
```

---

### Step 5 — Run fraud detection rules

```bash
docker compose exec app python app/fraud_rules.py
```

Expected output:
```
[RULE 1] Velocity fraud — accounts sending >3 txns in <10 steps
Found: 127 suspicious accounts

[RULE 2] Money mule chains (4-hop TRANSFER→CASH_OUT)
Found: 43 chains

[RULE 3] Balance drain (account emptied in single transfer)
Found: 891 transactions
```

---

### Step 6 — Run GDS community detection

```bash
docker compose exec app python app/gds_analysis.py
```

Expected output:
```
Running Louvain community detection...
Communities found: 284
Largest fraud community size: 47 accounts
PageRank scores written to Account nodes.
```

Visualize in Neo4j Browser — style nodes by `community` property. Fraud clusters appear in red.

---

### Step 7 — Start natural language chat

```bash
docker compose exec app python app/chat.py
```

Example session:
```
You: Which accounts sent over $100k to flagged accounts?
Graph: Found 12 accounts. Top: C1234567 sent $450,000 to flagged C9876543.

You: Show the top 5 most suspicious accounts by PageRank
Graph: 1. C8823901 (score: 4.21, 3 fraud transactions)
       2. C4412309 (score: 3.87, flagged destination)
       ...

You: exit
```

---

### Step 8 — Full pipeline (one command)

```bash
docker compose exec app python app/run_all.py
```

Runs all steps in sequence with progress output.

---

## Manual Test Cases

Run all queries in Neo4j Browser (http://localhost:7474) after completing Steps 4–6.

---

### TC-01 — Data integrity: node counts

```cypher
MATCH (n) RETURN labels(n), count(n) ORDER BY count(n) DESC
```

**Expect:** Two rows — `Account` (many) and `Transaction` (equal to `LOAD_LIMIT`, default 50 000)

---

### TC-02 — Relationship integrity: every transaction has exactly 2 edges

```cypher
MATCH (tx:Transaction)
OPTIONAL MATCH (tx)<-[:SENT]-(src)
OPTIONAL MATCH (tx)-[:RECEIVED_BY]->(dst)
WITH tx, count(src) AS senders, count(dst) AS receivers
WHERE senders <> 1 OR receivers <> 1
RETURN count(tx) AS orphaned_transactions
```

**Expect:** `0` — every transaction has exactly one sender and one receiver

---

### TC-03 — Fraud label sanity: labeled vs flagged

```cypher
MATCH (tx:Transaction)
RETURN
  count(tx)                                    AS total,
  sum(CASE WHEN tx.isFraud = true  THEN 1 ELSE 0 END) AS labeled_fraud,
  sum(CASE WHEN tx.isFlagged = true THEN 1 ELSE 0 END) AS system_flagged,
  sum(CASE WHEN tx.isFraud = true AND tx.isFlagged = false THEN 1 ELSE 0 END) AS missed_by_system
```

**Expect:** `labeled_fraud` > 0; `missed_by_system` reveals real detection gap (demo talking point)

---

### TC-04 — Velocity pattern: accounts with rapid-fire transactions

```cypher
MATCH (src:Account)-[:SENT]->(tx:Transaction)
WITH src, count(tx) AS txCount, collect(tx.step) AS steps
WITH src, txCount, reduce(mx=0, s IN steps | CASE WHEN s>mx THEN s ELSE mx END) -
         reduce(mn=9999, s IN steps | CASE WHEN s<mn THEN s ELSE mn END) AS window
WHERE txCount > 3 AND window <= 10
RETURN src.id AS account, txCount, window
ORDER BY txCount DESC LIMIT 10
```

**Expect:** Several accounts with `txCount` > 3 in a short window — these are card-testing candidates

---

### TC-05 — Mule chain: A→B→C→cashout (3-hop)

```cypher
MATCH path =
  (origin:Account)-[:SENT]->(t1:Transaction)-[:RECEIVED_BY]->
  (mid:Account)-[:SENT]->(t2:Transaction)-[:RECEIVED_BY]->
  (exit:Account)-[:SENT]->(t3:Transaction)
WHERE t1.type = 'TRANSFER'
  AND t2.type = 'TRANSFER'
  AND t3.type IN ['CASH_OUT','TRANSFER']
  AND origin.id <> exit.id
RETURN origin.id AS origin, mid.id AS mule, exit.id AS exit,
       t1.amount AS step1, t2.amount AS step2, t3.amount AS step3
ORDER BY step1 DESC LIMIT 10
```

**Expect:** Chains where amounts decrease slightly at each hop (layering signature)

---

### TC-06 — Balance drain: account emptied in single transfer

```cypher
MATCH (src:Account)-[:SENT]->(tx:Transaction)
WHERE tx.type = 'TRANSFER'
  AND src.balance > 0
  AND tx.amount >= src.balance * 0.95
RETURN src.id AS account,
       src.balance AS initial_balance,
       tx.amount AS drained,
       round(tx.amount / src.balance * 100, 1) AS pct_drained,
       tx.isFraud AS ground_truth
ORDER BY drained DESC LIMIT 15
```

**Expect:** Accounts with `pct_drained` ≥ 95%. `ground_truth=true` rows confirm the rule catches real fraud

---

### TC-07 — GDS: fraud community isolation

```cypher
// Communities that contain at least one flagged account
MATCH (a:Account)
WHERE a.community IS NOT NULL
WITH a.community AS community,
     count(a) AS total,
     sum(CASE WHEN a.flagVelocity OR a.flagMule OR a.flagDrain THEN 1 ELSE 0 END) AS fraudAccounts
WHERE fraudAccounts > 0
RETURN community, total, fraudAccounts,
       round(fraudAccounts * 100.0 / total, 1) AS fraud_pct
ORDER BY fraud_pct DESC LIMIT 10
```

**Expect:** Some communities with high `fraud_pct` — graph topology isolates fraud rings

---

### TC-08 — GDS: top accounts by PageRank cross-referenced with fraud flags

```cypher
MATCH (a:Account)
WHERE a.pageRank IS NOT NULL
RETURN a.id AS account,
       round(a.pageRank, 3) AS pageRank,
       a.community AS community,
       CASE WHEN a.flagVelocity THEN '⚡' ELSE '' END +
       CASE WHEN a.flagMule     THEN '🔗' ELSE '' END +
       CASE WHEN a.flagDrain    THEN '🚨' ELSE '' END AS flags
ORDER BY pageRank DESC LIMIT 15
```

**Expect:** High-PageRank accounts with flag symbols — central nodes in fraud rings score high

---

### TC-09 — NL Chat: 5 demo questions

Run `docker compose exec app python chat.py` then type each:

| # | Question | Expected Cypher pattern |
|---|----------|------------------------|
| 1 | `Which accounts sent over $100k to flagged accounts?` | `WHERE tx.amount > 100000` + fraud flag join |
| 2 | `Show me the top 5 most suspicious accounts by PageRank` | `ORDER BY a.pageRank DESC LIMIT 5` |
| 3 | `Find accounts that emptied their balance in a single transfer` | `tx.amount >= src.balance * 0.95` |
| 4 | `How many transactions are labeled as fraud?` | `count(tx) WHERE tx.isFraud = true` |
| 5 | `Which community has the most fraud accounts?` | GROUP BY community, count flagged |

**Expect:** Valid Cypher generated for each; Neo4j returns non-empty results for Q2 and Q4 regardless of sample size

---

### TC-10 — Shortest path between two accounts (graph vs SQL demo)

```cypher
// Find if two accounts are connected — replace IDs with real ones from TC-04
MATCH (a:Account {id: 'C1231006815'}), (b:Account {id: 'C1666544250'})
CALL gds.shortestPath.dijkstra.stream('fraud-graph', {
  sourceNode: a,
  targetNode: b
})
YIELD path
RETURN [n IN nodes(path) | n.id] AS hops, length(path) AS distance
```

**Talking point:** "SQL would need 4+ self-joins for this 4-hop query. Neo4j traverses it natively in milliseconds."

> Replace account IDs with real ones from your dataset — pick two from TC-04 output.

---

## Stopping and Cleanup

```bash
# Stop containers (keep data)
docker compose stop

# Stop and delete all data
docker compose down -v
```

---

## Project Structure

```
fraud-graph-demo/
├── .env.example          ← copy to .env, fill in keys
├── .gitignore
├── docker-compose.yml
├── TASKS.md              ← progress tracker
├── README.md             ← this file
├── app/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── ingest.py         ← T-03: load PaySim → Neo4j
│   ├── fraud_rules.py    ← T-04: Cypher fraud patterns
│   ├── gds_analysis.py   ← T-05: GDS Louvain + PageRank
│   ├── chat.py           ← T-06: LangChain + Ollama NL chat
│   └── run_all.py        ← T-07: full pipeline runner
└── data/
    └── *.csv             ← PaySim dataset (gitignored)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Neo4j not ready | `docker compose logs neo4j` — wait for "Started" |
| GDS plugin missing | Check `NEO4J_PLUGINS` in docker-compose.yml |
| Ollama auth error | Verify `OLLAMA_API_KEY` in `.env` |
| CSV not found | Check file is in `data/` with exact filename from Kaggle |
| Out of memory | Reduce `LOAD_LIMIT` in `.env` (default 50000) |
