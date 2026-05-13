# Architecture — Fraud Graph Demo

End-to-end fraud detection system using Neo4j knowledge graph, GDS algorithms, GraphSAGE GNN, and LLM-powered natural language querying.

---

## System Overview

```mermaid
graph TB
    subgraph Host["Host Machine"]
        CSV["PaySim CSV\n50k transactions"]
        OL["Ollama Cloud API\ndeepseek-v4-flash"]
    end

    subgraph Docker["Docker Compose"]
        subgraph App["fraud-app container\nPython 3.11 · PyTorch · PyG"]
            ING["ingest.py\nCSV → Graph"]
            FR["fraud_rules.py\nCypher Rules"]
            GDS["gds_analysis.py\nGDS Algorithms"]
            GNN["gnn_train.py\nGraphSAGE GNN"]
            CH["chat.py\nNL → Cypher"]
            BM["benchmark.py\nPerformance Tests"]
        end

        subgraph Neo4j["fraud-neo4j container\nNeo4j 5 + GDS 2.13"]
            DB[("Graph DB\nAccounts + Transactions")]
        end
    end

    CSV -->|volume mount| ING
    ING -->|Bolt 7687| DB
    FR -->|Bolt 7687| DB
    GDS -->|Bolt 7687| DB
    GNN -->|read features| DB
    GNN -->|WRITE fraudProb| DB
    CH -->|Bolt 7687| DB
    BM -->|Bolt 7687| DB
    CH -->|HTTPS| OL
    DB -->|HTTP 7474| Browser["Neo4j Browser\nlocalhost:7474"]
```

---

## Data Pipeline

```mermaid
flowchart LR
    A["PaySim CSV\n6.3M rows"] -->|LOAD_LIMIT=50k| B["ingest.py\nMERGE batches\n500 rows each"]
    B --> C[("Neo4j Graph\n~78k Accounts\n50k Transactions")]
    C --> D["fraud_rules.py\n3 Cypher rules"]
    D -->|SET flags| C
    C --> E["gds_analysis.py\n5 GDS algorithms"]
    E -->|WRITE properties| C
    C --> G["gnn_train.py\nGraphSAGE · 150 epochs"]
    G -->|READ pageRank,betweenness\nflagVelocity,flagMule,flagDrain| C
    G -->|WRITE fraudProb| C
    C --> F["chat.py\nNL query interface"]
    F -->|Cypher| C
    C -->|results| F
```

---

## Graph Data Model

```mermaid
graph LR
    A["Account\n─────────\nid\nbalance\npageRank\ncommunity\nwccComponent\nbetweenness\ntriangleCount\nflagVelocity\nflagMule\nflagDrain"]
    T["Transaction\n─────────\nid\namount\ntype\nstep\nisFraud\nisFlagged\nflagDrain"]

    A -->|SENT| T
    T -->|RECEIVED_BY| A2["Account"]
```

**Transaction types:** `PAYMENT` · `TRANSFER` · `CASH_OUT` · `DEBIT` · `CASH_IN`

**Fraud flags written by `fraud_rules.py`:**

| Flag | Rule | Pattern |
|------|------|---------|
| `flagVelocity` | >3 txns within 10 steps | Card-testing / account takeover |
| `flagMule` | On A→B→C→cashout chain | Money mule layering |
| `flagDrain` | Emptied ≥95% balance in one transfer | Smash-and-grab fraud |

**GDS properties written by `gds_analysis.py`:**

| Property | Algorithm | Fraud signal |
|----------|-----------|--------------|
| `community` | Louvain | High-fraud-density clusters |
| `pageRank` | PageRank | Central money-hub accounts |
| `wccComponent` | WCC | Isolated fraud rings |
| `betweenness` | Betweenness Centrality | Bridge / relay accounts |
| `triangleCount` | Cycle Detection (Cypher) | Circular layering flows |

---

## GDS Algorithm Pipeline

```mermaid
flowchart TD
    RAW[("Neo4j Graph\nAccount + Transaction nodes")]

    RAW -->|"gds.graph.project.cypher\nAccount→Account virtual edges"| PROJ["In-Memory\nGDS Projection\n78k nodes · 50k edges"]

    PROJ --> LV["Louvain\ncommunity detection\nmaxLevels=1\n~62ms"]
    PROJ --> PR["PageRank\ninfluence ranking\niter=5 converges\n~78ms"]
    PROJ --> WCC["WCC\nring isolation\nO(n+e)\n~20ms"]
    PROJ --> BC["Betweenness\nrelay accounts\nsample=100\n~21ms"]
    RAW  --> CD["Cycle Detection\nCypher 2+3-hop\n~100ms"]

    LV  -->|"WRITE community"| OUT[("Properties on\nAccount nodes")]
    PR  -->|"WRITE pageRank"| OUT
    WCC -->|"WRITE wccComponent"| OUT
    BC  -->|"WRITE betweenness"| OUT
    CD  -->|"WRITE triangleCount"| OUT
```

---

## GNN Pipeline — GraphSAGE

```mermaid
flowchart TD
    subgraph Input["Input — from Neo4j after GDS"]
        F1["pageRank\n(normalised)"]
        F2["betweenness\n(normalised)"]
        F3["wccComponent\n(normalised)"]
        F4["flagVelocity\n0 or 1"]
        F5["flagMule\n0 or 1"]
        F6["flagDrain\n0 or 1"]
    end

    subgraph Model["GraphSAGE Model (3-layer)"]
        L1["SAGEConv(6→64)\n+ ReLU + Dropout(0.3)"]
        L2["SAGEConv(64→64)\n+ ReLU + Dropout(0.3)"]
        L3["SAGEConv(64→2)\nlogits"]
        SIG["sigmoid\n→ fraudProb ∈ 0,1"]
        L1 --> L2 --> L3 --> SIG
    end

    subgraph Training["Training"]
        LOSS["Weighted CrossEntropy\npos_weight ≈ 74×"]
        ADAM["Adam · lr=0.005\nweight_decay=5e-4"]
        SPLIT["Train 60% · Val 20% · Test 20%\nBest model by val AUC"]
    end

    subgraph Output["Output"]
        EVAL["Metrics: Precision · Recall · F1 · AUC-ROC"]
        COMP["Comparison: Rules vs GNN vs Ensemble"]
        WRITE["fraudProb → SET a.fraudProb\non all Account nodes"]
    end

    Input --> Model
    Model --> Training
    Training --> Output
```

**Layer behaviour:** Each SAGEConv aggregates features from 1-hop neighbours (mean aggregation), concatenates with the node's own features, and applies a linear transform. After 3 layers, each node's representation encodes its 3-hop neighbourhood — enough to capture the full mule chain structure (A→B→C→cashout).

---

## NL Chat Pipeline

Chat launches automatically after `run_all.py` completes — always, regardless of smoke test result.  
Each session is logged to `logs/chat_YYYY-MM-DD_HH-MM-SS.log` (question · Cypher · results · timing).  
To quit: type `quit` / `exit` / `q`.

Two LangChain chains run per question — Cypher generator + result interpreter:

```mermaid
sequenceDiagram
    participant U as User
    participant CH as chat.py
    participant SP as spaCy NER
    participant LM1 as LLM Call 1\nCypher Generator
    participant DB as Neo4j
    participant LM2 as LLM Call 2\nResult Interpreter
    participant LOG as logs/chat_*.log

    U->>CH: "What is the most fraudulent account and why?"
    CH->>SP: extract entities
    SP-->>CH: None detected
    CH->>LM1: PROMPT(schema + entities + question)
    Note over LM1: ~1-2s · retries on 503
    LM1-->>CH: MATCH (a:Account) ORDER BY a.fraudProb DESC LIMIT 1
    CH->>DB: execute Cypher
    DB-->>CH: {id: C123, fraudProb: 0.55, flagDrain: true, pageRank: 7.2}
    CH->>LM2: INTERPRET_PROMPT(question + raw rows)
    Note over LM2: ~1-2s · explains fraud signals
    LM2-->>CH: "Account C123 is highest risk — drained balance in single transfer..."
    CH->>U: raw rows + plain-English answer
    CH->>LOG: question · Cypher · results · interpretation · all timings
```

**Two LangChain chains:**
```python
chain           = PROMPT | llm           # generates Cypher
interpret_chain = INTERPRET_PROMPT | llm # explains results in plain English
```

**Retry logic:** On 503 overloaded — waits 5/10/15s, retries up to 3×. Interpreter failure is non-fatal — raw results still shown.

**Schema exposed to Cypher LLM:**
```
(:Account {id, balance, pageRank, community, wccComponent, betweenness,
           triangleCount, flagVelocity, flagMule, flagDrain, fraudProb})
```

**Flag initialization:** `fraud_rules.py` sets `flagVelocity=false`, `flagMule=false`, `flagDrain=false` on all accounts before running rules — eliminates null values that break boolean filters.

---

## Container Architecture

```mermaid
graph TB
    subgraph Compose["docker-compose.yml"]
        direction TB

        subgraph NEO["fraud-neo4j"]
            NI["neo4j:5-community"]
            GI["GDS plugin 2.13"]
            NV[("neo4j_data volume\nneo4j_logs volume")]
            NI --- GI
            NI --- NV
        end

        subgraph APP["fraud-app"]
            DK["python:3.11-slim"]
            EP["entrypoint.sh\nauto-ingest on start"]
            PY["Python scripts"]
            DK --- EP
            EP --- PY
        end

        HC["healthcheck\ncypher-shell RETURN 1\nevery 15s"]
        NEO -->|healthy| APP
        NEO --- HC
    end

    ENV[".env\nNEO4J_URI\nOLLAMA_BASE_URL\nOLLAMA_API_KEY\nLOAD_LIMIT"]
    ENV -->|env_file| APP

    DNS["DNS: 8.8.8.8 / 1.1.1.1\nresolves api.ollama.com"]
    DNS -->|injected| APP
```

---

## File Structure

```
fraud-graph-demo/
├── docker-compose.yml        ← two services: neo4j + app
├── .env                      ← secrets (gitignored)
├── .env.example              ← template
├── README.md                 ← setup + test cases
├── ARCHITECTURE.md           ← this file
├── benchmark_report.md       ← generated benchmark results
├── architecture.drawio       ← draw.io diagram
│
├── app/
│   ├── Dockerfile            ← python:3.11-slim + spaCy model
│   ├── entrypoint.sh         ← auto-ingest if DB empty on startup
│   ├── requirements.txt
│   ├── ingest.py             ← PaySim CSV → Neo4j (MERGE batches)
│   ├── fraud_rules.py        ← 3 Cypher fraud detection rules
│   ├── gds_analysis.py       ← 5 GDS algorithms + Cypher cycle detection
│   ├── gnn_train.py          ← GraphSAGE 3-layer · fraudProb → Neo4j
│   ├── chat.py               ← spaCy NER + LangChain + Ollama NL→Cypher + session logging
│   ├── run_all.py            ← full pipeline smoke test (9 checks) + auto-launch chat
│   └── benchmark.py          ← timing benchmark across 4 sizes × 3 configs
│
├── logs/                     ← run evidence (volume-mounted from container)
│   ├── run_YYYY-MM-DD_HH-MM-SS.log   ← full pipeline output
│   └── chat_YYYY-MM-DD_HH-MM-SS.log  ← chat session log
│
└── data/
    └── *.csv                 ← PaySim dataset (gitignored)
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Graph DB | Neo4j 5 community | Native graph traversal, GDS plugin, Bolt protocol |
| Graph projection | Cypher (Account→Account) | Native projection missed Account↔Account edges through Transaction nodes |
| LLM | Ollama Cloud deepseek-v4-flash | Fast inference, no local GPU needed, Ollama API compatible |
| LLM client | Custom `OllamaCloudLLM(LLM)` | `langchain_ollama.OllamaLLM` ignores `base_url`/`headers` params |
| Betweenness | `samplingSize=100` | 142× faster than exact, sufficient for fraud use case |
| Louvain | `maxLevels=1` | Same modularity (0.9999), 5× cheaper |
| PageRank | `maxIterations=5` | Converges in 2 iterations on this graph; higher wastes compute |
| Ingest | MERGE not CREATE | Idempotent — safe to re-run; entrypoint skips if DB already populated |
| Cycle detection | Cypher 2+3-hop instead of GDS triangleCount | GDS triangleCount requires UNDIRECTED projection; money flows are directed |
| GNN model | GraphSAGE over GCN | GraphSAGE is inductive — new accounts scored without retraining; GCN is transductive |
| GNN features | GDS properties + rule flags | Combines structural signals (pageRank, betweenness) with rule-based signals in one feature vector |
| GNN loss | Weighted cross-entropy (pos_weight ≈ 74×) | PaySim fraud rate ~1.3% — without weighting the model ignores the minority class |
| GNN epochs | 150 with best-val-AUC checkpoint | Prevents overfitting; fraud graphs have noisy labels (simulation artefacts) |

---

## Algorithm Design Rationale

### Why Louvain for Community Detection?

Louvain maximises **modularity** — the fraction of edges inside communities minus the expected fraction if edges were placed randomly. Fraud rings have dense internal edges (accounts transact repeatedly within the ring) and few external edges (they avoid legitimate accounts). This structure scores high modularity, making Louvain naturally suited to isolate fraud rings.

Alternatives considered:
- **Label Propagation** — faster but non-deterministic; different runs produce different communities
- **K-Means on embeddings** — requires feature engineering; Louvain works directly on graph structure

### Why PageRank for Coordinator Detection?

In a money-flow graph, PageRank propagates "importance" along transaction edges. Accounts that receive funds from many other accounts accumulate high PageRank — they are the **aggregation points** in the layering chain. Unlike simple degree centrality (count of connections), PageRank weights connections by the importance of the sender, making it robust to accounts that simply have many low-value connections.

### Why WCC for Real-Time Ring Isolation?

WCC runs in **O(n+e)** via union-find. It's the only algorithm in the pipeline that can scale to streaming fraud detection. The insight: legitimate accounts connect to the broad transaction network (thousands of reachable accounts); fraud rings are self-contained subgraphs with limited external connections. WCC assigns all nodes in a connected component the same ID — downstream filtering then flags components where fraud density exceeds a threshold.

### Why Betweenness Sampling?

Exact betweenness requires computing shortest paths between **all pairs of nodes** — O(n·e) for unweighted graphs, O(n·e + n²·log n) for weighted. At 78k nodes this is 4.5 seconds. Approximate betweenness samples a random subset of source nodes (100 here) and extrapolates — **142× faster** with near-identical rankings for the highest-betweenness nodes (which are the fraud signals we care about).

### Why Cypher Cycle Detection instead of GDS triangleCount?

GDS triangle counting requires an UNDIRECTED projection because triangles are symmetric (A-B-C-A is the same as A-C-B-A in an undirected graph). Our Account→Account projection is **directed** — money flows from sender to receiver, not both ways. Projecting as UNDIRECTED would lose the directional information needed to distinguish genuine circular flows from coincidental return payments. Cypher pattern matching on the directed graph is slower (~115ms) but semantically correct.

### Why GraphSAGE for Fraud Node Classification?

**Rules vs GDS vs GNN — three complementary layers:**

| Layer | Mechanism | Catches | Misses |
|-------|-----------|---------|--------|
| Fraud rules (Cypher) | Pattern matching | Known fraud patterns (velocity, mule, drain) | Novel patterns, structural guilt-by-association |
| GDS algorithms | Graph topology metrics | Structural roles (hubs, bridges, rings) | Can't label fraud directly — unsupervised |
| GraphSAGE (GNN) | Learned neighbourhood aggregation | Complex patterns + structural context + known signals together | Patterns outside training distribution |

**Why neighbourhood aggregation matters:**  
A "clean" account (no rule flags, moderate PageRank) that consistently routes money from high-velocity senders to high-drain receivers is structurally suspicious. Rules won't fire. GDS metrics will be moderate. But GraphSAGE, aggregating across 3 hops, sees that its neighbourhood is saturated with fraud signals — and scores it high.

**Why 3 layers / 3 hops?**  
A typical mule chain is 3–4 hops: origin → mule → aggregator → cashout. 3 SAGEConv layers means each node's embedding incorporates information from all accounts within 3 transaction steps — exactly the range needed to see the full layering structure.

**Inductive vs transductive:**  
GCN (Graph Convolutional Network) learns fixed embeddings for each node — it cannot score new nodes without retraining. GraphSAGE learns an *aggregation function* (mean of neighbour features + linear transform). When a new account joins the graph, its embedding is computed on-the-fly from its neighbours' existing features. This is essential for production fraud detection where new accounts appear constantly.

**Class imbalance strategy:**  
PaySim fraud rate ≈ 1.3% (1 fraud in 74 legitimate accounts). Without correction, the model achieves 98.7% accuracy by predicting "clean" for everything. Weighted cross-entropy assigns `pos_weight = 74` to fraud samples — each fraud example counts as much as 74 clean examples in the loss gradient. This forces the model to learn fraud-discriminative features rather than the majority class distribution.

### Graph Projection Design

The raw PaySim graph is **bipartite**: Accounts and Transactions are separate node types connected by `SENT` and `RECEIVED_BY` relationships. GDS community/centrality algorithms are designed for **unipartite** graphs (one node type). The Cypher projection solves this by collapsing each `Account→Transaction→Account` path into a virtual direct `Account→Account` edge:

```
Raw graph:    (src:Account) -[:SENT]-> (tx:Transaction) -[:RECEIVED_BY]-> (dst:Account)
Projected:    (src:Account) ──────────────────────────────────────────────> (dst:Account)
```

Native projection alternative (rejected): mapping `SENT` relationships would give `Account→Transaction` edges — GDS would see Transaction nodes as the neighbours of Account nodes, producing meaningless community assignments.
