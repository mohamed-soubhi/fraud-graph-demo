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

Open Neo4j Browser: http://localhost:7474  
Login: `neo4j` / your password from `.env`

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

## Useful Neo4j Cypher Queries

```cypher
// All fraud transactions
MATCH (t:Transaction {isFraud: true})
RETURN t.id, t.amount, t.type LIMIT 25

// Money mule chain
MATCH path = (a:Account)-[:SENT]->(t1:Transaction)-[:RECEIVED_BY]->
             (b:Account)-[:SENT]->(t2:Transaction)-[:RECEIVED_BY]->
             (c:Account)
WHERE t1.type = 'TRANSFER' AND t2.type = 'CASH_OUT'
RETURN path LIMIT 5

// Suspicious accounts by community + PageRank
MATCH (a:Account)
WHERE a.pageRank > 2.0
RETURN a.id, a.pageRank, a.community
ORDER BY a.pageRank DESC LIMIT 10
```

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
