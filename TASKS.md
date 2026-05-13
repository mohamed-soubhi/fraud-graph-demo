# Fraud Graph Demo — Task Tracker

> Interview demo project: PaySim fraud detection using Neo4j + GraphRAG + Ollama
> Target: working demo by 2026-05-14

---

## Progress

| Ticket | Description | Status |
|--------|-------------|--------|
| T-01 | Project setup + Docker | ✅ Done |
| T-01b | Kaggle MCP server config | ✅ Done |
| T-02 | Download PaySim dataset | ⬜ Blocked: restart Claude Code |
| T-03 | Neo4j schema + ingestion | ✅ Done |
| T-04 | Fraud detection Cypher rules | ✅ Done |
| T-05 | GDS community detection | ✅ Done |
| T-06 | LangChain + Ollama NL chat | ✅ Done |
| T-07 | End-to-end smoke test | ✅ Done |
| T-08 | Demo rehearsal script | ⬜ Todo |

---

## T-01b — Kaggle MCP Server Config
**Status:** ✅ Done  
**Commit:** `chore: register kaggle mcp server in global claude settings`

### What was configured
- Added `kaggle` entry to `~/.claude.json` → `projects["/home/msoubhi"].mcpServers`
- Added catalog entry to `~/.claude/mcp-configs/mcp-servers.json`
- MCP type: `http`, URL: `https://www.kaggle.com/mcp`
- Authentication: OAuth (handled by Claude Code on first connect)

### To activate
1. **Restart Claude Code** — MCP servers load at startup only
2. On restart, Claude Code will prompt Kaggle OAuth login
3. After auth, Kaggle MCP tools become available in session
4. Run: download `ealaxi/paysim1` → `fraud-graph-demo/data/`

### Files changed (global, not in this repo)
```
~/.claude.json                          ← kaggle entry added to mcpServers
~/.claude/mcp-configs/mcp-servers.json ← catalog entry added
```

---

## T-01 — Project Setup + Docker
**Status:** ✅ Done  
**Commit:** `feat: project scaffold, docker-compose, env, readme`

- [x] Create folder structure
- [x] `.env.example` with all variables
- [x] `.gitignore` (exclude `.env`, CSV data)
- [x] `docker-compose.yml` with Neo4j + app
- [x] `app/requirements.txt`
- [x] `README.md` user manual

---

## T-02 — Download PaySim Dataset
**Status:** ⬜ Todo  
**Commit:** `chore: add paysim dataset to data/`

### Steps
1. Go to: https://www.kaggle.com/datasets/ealaxi/paysim1
2. Download `PS_20174392719_1491204439457_log.csv`
3. Place in `fraud-graph-demo/data/`
4. Verify: `wc -l data/*.csv` → should be ~6.4M rows

### Verify columns
```
step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
```

---

## T-03 — Neo4j Schema + Data Ingestion
**Status:** ⬜ Todo  
**Commit:** `feat: paysim ingestion to neo4j graph`

### Steps
1. Start containers: `docker compose up -d`
2. Verify Neo4j at http://localhost:7474
3. Run: `docker compose exec app python app/ingest.py`
4. Verify in Neo4j Browser:
   ```cypher
   MATCH (n) RETURN labels(n), count(n)
   ```

### Graph model to build
```
(:Account {id, balance})
(:Transaction {id, amount, type, isFraud, step})
(:Account)-[:SENT]->(t:Transaction)-[:RECEIVED_BY]->(:Account)
```

---

## T-04 — Fraud Detection Cypher Rules
**Status:** ⬜ Todo  
**Commit:** `feat: cypher fraud pattern queries`

### Steps
1. Run: `docker compose exec app python app/fraud_rules.py`
2. Verify 3 patterns return results:
   - **Velocity**: accounts sending >3 transactions in <10 steps
   - **Mule chain**: money path A→B→C→cashout
   - **High-value drain**: account emptied in single TRANSFER

---

## T-05 — GDS Community Detection
**Status:** ⬜ Todo  
**Commit:** `feat: gds louvain fraud community detection`

### Steps
1. Verify GDS plugin loaded:
   ```cypher
   RETURN gds.version()
   ```
2. Run: `docker compose exec app python app/gds_analysis.py`
3. Open Neo4j Browser → style nodes by `community` property
4. Fraud nodes should cluster visually

---

## T-06 — LangChain + Ollama NL Chat
**Status:** ⬜ Todo  
**Commit:** `feat: langchain ollama natural language graph queries`

### Steps
1. Add `OLLAMA_API_KEY` to `.env`
2. Run: `docker compose exec app python app/chat.py`
3. Test questions:
   - `"Which accounts sent over $100k to flagged accounts?"`
   - `"Show me the top 5 most suspicious accounts by PageRank"`
   - `"Find accounts that emptied their balance in a single transfer"`

---

## T-07 — End-to-End Smoke Test
**Status:** ⬜ Todo  
**Commit:** `test: end-to-end smoke test all components`

### Steps
1. Fresh start: `docker compose down -v && docker compose up -d`
2. Run full pipeline: `docker compose exec app python app/run_all.py`
3. Confirm:
   - [ ] Nodes loaded (Account + Transaction counts)
   - [ ] Fraud rules return results
   - [ ] GDS communities assigned
   - [ ] Chat responds to 3 test queries

---

## T-08 — Demo Rehearsal Script
**Status:** ⬜ Todo  

### Demo flow (5 min)
1. **(30s)** Open Neo4j Browser → show full graph → "this is 50K PaySim transactions"
2. **(60s)** Run mule chain Cypher → explain graph traversal advantage over SQL
3. **(60s)** Show GDS Louvain result → fraud community highlighted in red
4. **(90s)** Live chat demo → type NL question → show auto-generated Cypher → show answer
5. **(60s)** Architecture diagram → mention Kafka/Spark/Airflow as production scale-up path

### Key talking points during demo
- "Neo4j finds this 4-hop money mule chain in milliseconds — SQL would require 4 self-joins"
- "GDS runs PageRank and community detection natively — no data export needed"
- "LangChain translates analyst questions to Cypher automatically — democratizes graph access"
- "Production version would use Kafka for real-time transaction streaming into the graph"
