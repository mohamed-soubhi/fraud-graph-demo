"""T-06: Natural language → Cypher chat via LangChain + Ollama.

Pipeline: Question → spaCy NER (extract entities/amounts) → enriched prompt → LLM → Cypher → Neo4j
Each session is logged to /app/logs/chat_YYYY-MM-DD_HH-MM-SS.log
"""
import os
import re
import time
import spacy
import ollama
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional
from neo4j import GraphDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from dotenv import load_dotenv

load_dotenv()

URI          = os.environ["NEO4J_URI"]
USER         = os.environ["NEO4J_USER"]
PASSWORD     = os.environ["NEO4J_PASSWORD"]
OLLAMA_URL   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_KEY   = os.environ.get("OLLAMA_API_KEY", "")


class OllamaCloudLLM(LLM):
    model: str
    base_url: str
    api_key: str = ""

    @property
    def _llm_type(self) -> str:
        return "ollama-cloud"

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        client = ollama.Client(host=self.base_url, headers=headers)
        response = client.generate(model=self.model, prompt=prompt)
        return response["response"]

TX_TYPES = {"PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"}

SCHEMA = """
Graph schema:
  (:Account {id, balance, pageRank, community, wccComponent, betweenness, triangleCount,
             flagVelocity, flagMule, flagDrain, fraudProb})
  (:Transaction {id, amount, type, step, isFraud, isFlagged, flagDrain})
  (:Account)-[:SENT]->(:Transaction)-[:RECEIVED_BY]->(:Account)

Transaction types: PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN

Account fraud signals (all properties are always set — never null):
  flagVelocity  boolean — sent >3 transactions within 10 steps (card-testing)
  flagMule      boolean — appears on A→B→C→cashout chain (money mule)
  flagDrain     boolean — emptied ≥95% balance in one transfer (account takeover)
  fraudProb     float [0,1] — GraphSAGE GNN score (neighbourhood-based, 3-hop)
  pageRank      float — money-flow influence (high = central aggregator)
  betweenness   float — relay/bridge position between accounts
  community     int   — Louvain fraud ring cluster ID
  wccComponent  int   — isolated connected component ID
"""

PROMPT = PromptTemplate.from_template("""
You are a Neo4j Cypher expert. Convert the question into a single valid Cypher query.
Return ONLY the Cypher query — no markdown, no explanation, no comments.

Rules:
- For "why" or "explain" questions: return id plus ALL relevant fraud signals
  (fraudProb, flagVelocity, flagMule, flagDrain, pageRank, betweenness, community)
- For ranking questions: ORDER BY the most relevant property DESC, LIMIT 10
- Never use properties that don't exist in the schema
- All flag properties are boolean (never null) — safe to use in WHERE clauses

{schema}

Extracted entities from the question:
{entities}

Question: {question}

Cypher:
""")

INTERPRET_PROMPT = PromptTemplate.from_template("""
You are a fraud analyst assistant. A user asked a question about a fraud detection graph database.
Based on the Cypher query results below, write a concise 2-4 sentence plain-English answer.

Explain WHAT the data shows and WHY it is significant for fraud detection.
Reference specific account IDs and numeric values from the results.
Use fraud domain language: money mule, smash-and-grab, velocity fraud, layering, GNN score, etc.

Fraud signal meanings:
  flagVelocity=true  → sent >3 transactions within 10 steps (card-testing)
  flagMule=true      → on a A→B→C→cashout money mule chain (layering)
  flagDrain=true     → emptied ≥95% balance in one transfer (account takeover)
  fraudProb          → GraphSAGE GNN score: neighbourhood-based fraud probability [0,1]
  pageRank           → money-flow influence (high = central aggregator hub)
  betweenness        → relay/bridge node between many accounts (laundering coordinator)
  community          → Louvain fraud ring cluster

Question: {question}

Query results (up to 10 rows):
{results}

Answer (plain English, 2-4 sentences):
""")

_ACCOUNT_RE = re.compile(r'\b[CM]\d+\b')
_MONEY_RE   = re.compile(r'\$?([\d,]+\.?\d*)\s*([kKmM]?)\b')


# ── chat logger ──────────────────────────────────────────────────────────────

class ChatLogger:
    def __init__(self):
        log_dir  = Path("/app/logs")
        log_dir.mkdir(exist_ok=True)
        stamp    = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = log_dir / f"chat_{stamp}.log"
        self._f  = self.path.open("w", buffering=1)
        self._write_header()

    def _write_header(self):
        self._f.write(
            f"{'='*60}\n"
            f"FRAUD GRAPH CHAT SESSION\n"
            f"Started : {datetime.now().isoformat()}\n"
            f"Model   : {OLLAMA_MODEL}\n"
            f"Neo4j   : {URI}\n"
            f"Log     : {self.path}\n"
            f"{'='*60}\n\n"
        )

    def log_interaction(self, *, question: str, entities: str, cypher: str,
                        cypher_ms: float, rows: list | None, query_ms: float,
                        error: str | None, interpretation: str | None = None,
                        interpret_ms: float = 0.0):
        total_ms = cypher_ms + (query_ms if query_ms else 0) + interpret_ms
        ts = datetime.now().strftime("%H:%M:%S")
        block = (
            f"[{ts}] QUESTION\n"
            f"  {question}\n\n"
            f"  Entities  : {entities}\n\n"
            f"  Cypher ({cypher_ms:.0f}ms):\n"
            f"    {cypher}\n\n"
        )
        if error:
            block += f"  ERROR: {error}\n"
        else:
            block += f"  Results ({len(rows)} rows) [{query_ms:.0f}ms]:\n"
            for r in (rows or [])[:10]:
                block += f"    {r}\n"
            if rows and len(rows) > 10:
                block += f"    … {len(rows)-10} more rows\n"
        if interpretation:
            block += f"\n  Answer ({interpret_ms:.0f}ms):\n  {interpretation}\n"
        block += f"\n  Total round-trip: {total_ms:.0f}ms\n"
        block += f"{'-'*60}\n\n"
        self._f.write(block)
        self._f.flush()

    def close(self):
        self._f.write(f"Session ended: {datetime.now().isoformat()}\n")
        self._f.close()


# ── NLP helpers ──────────────────────────────────────────────────────────────

def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("spaCy model not found — run: python -m spacy download en_core_web_sm")
        raise


def normalize_amount(num_str: str, suffix: str) -> float:
    value = float(num_str.replace(",", ""))
    if suffix.lower() == "k":
        value *= 1_000
    elif suffix.lower() == "m":
        value *= 1_000_000
    return value


def extract_entities(nlp, question: str) -> dict:
    doc = nlp(question)
    entities = {
        "account_ids": _ACCOUNT_RE.findall(question),
        "tx_types":    [w.upper() for w in question.split() if w.upper() in TX_TYPES],
        "amounts":     [],
        "spacy_ents":  [],
    }
    for ent in doc.ents:
        entities["spacy_ents"].append(f"{ent.text} ({ent.label_})")
    for match in _MONEY_RE.finditer(question):
        try:
            entities["amounts"].append(normalize_amount(match.group(1), match.group(2)))
        except ValueError:
            pass
    return entities


def format_entities(entities: dict) -> str:
    lines = []
    if entities["account_ids"]:
        lines.append(f"Account IDs: {', '.join(entities['account_ids'])}")
    if entities["tx_types"]:
        lines.append(f"Transaction types: {', '.join(entities['tx_types'])}")
    if entities["amounts"]:
        lines.append(f"Amounts (normalized): {', '.join(str(a) for a in entities['amounts'])}")
    if entities["spacy_ents"]:
        lines.append(f"Named entities: {', '.join(entities['spacy_ents'])}")
    return "\n".join(lines) if lines else "None detected"


# ── main loop ────────────────────────────────────────────────────────────────

def _llm_invoke_with_retry(chain, inputs: dict) -> str:
    """Call chain.invoke with up to 3 retries on 503 overload."""
    for attempt in range(1, 4):
        try:
            return chain.invoke(inputs).strip()
        except Exception as e:
            msg = str(e)
            if "503" in msg or "overloaded" in msg.lower():
                wait = attempt * 5
                print(f"  LLM overloaded — retrying in {wait}s (attempt {attempt}/3)...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("LLM unavailable after 3 retries")


def chat_loop():
    from agent import build_agent, initial_state

    nlp             = load_nlp()
    driver          = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    llm             = OllamaCloudLLM(model=OLLAMA_MODEL, base_url=OLLAMA_URL, api_key=OLLAMA_KEY)
    chain           = PROMPT | llm
    interpret_chain = INTERPRET_PROMPT | llm
    agent           = build_agent(llm, driver, chain, interpret_chain, SCHEMA)
    logger          = ChatLogger()

    print(f"Fraud Graph Chat (LangGraph self-healing agent) — type a question, 'quit' to exit")
    print(f"Session log → {logger.path}\n")

    demo_questions = [
        "Which accounts sent over $100k to flagged accounts?",
        "Show me the top 5 most suspicious accounts by PageRank",
        "Which accounts have fraudProb > 0.8 but no rule flags set?",
        "What is the most fraudulent account and why?",
    ]
    print("Demo questions to try:")
    for q in demo_questions:
        print(f"  > {q}")
    print()

    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        entities   = extract_entities(nlp, question)
        entity_str = format_entities(entities)
        print(f"\nExtracted: {entity_str}")

        print("Running agent (generate → execute → [fix →] interpret)...")
        try:
            state = agent.invoke(initial_state(question, entity_str))
        except Exception as e:
            print(f"  Agent error: {e}\n")
            logger.log_interaction(
                question=question, entities=entity_str,
                cypher="(none)", cypher_ms=0,
                rows=None, query_ms=0, error=str(e),
            )
            print("  (logged)\n")
            continue

        cypher         = state.get("cypher", "(none)")
        cypher_ms      = state.get("cypher_ms", 0.0)
        fix_ms         = state.get("fix_ms", 0.0)
        rows           = state.get("rows")
        query_ms       = state.get("query_ms", 0.0)
        cypher_error   = state.get("cypher_error")
        interpretation = state.get("interpretation")
        interpret_ms   = state.get("interpret_ms", 0.0)
        retries        = state.get("retries", 0)

        fix_note = f" (fixed after {retries} attempt(s))" if fix_ms > 0 else ""
        print(f"\nCypher [{cypher_ms:.0f}ms{fix_note}]:\n{cypher}\n")

        if cypher_error:
            print(f"Query failed after {retries} fix attempt(s): {cypher_error}")
        elif rows:
            print(f"Results ({len(rows)} rows) [{query_ms:.0f}ms]:")
            for r in rows[:10]:
                print(f"  {r}")
            if len(rows) > 10:
                print(f"  … {len(rows)-10} more rows")
        else:
            print(f"No results. [{query_ms:.0f}ms]")

        if interpretation:
            print(f"\n{'─'*50}")
            print(f"Answer [{interpret_ms:.0f}ms]:")
            print(f"  {interpretation}")
            print(f"{'─'*50}")

        logger.log_interaction(
            question=question,
            entities=entity_str,
            cypher=cypher,
            cypher_ms=cypher_ms + fix_ms,
            rows=rows,
            query_ms=query_ms,
            error=cypher_error,
            interpretation=interpretation,
            interpret_ms=interpret_ms,
        )
        print(f"  (logged)\n")

    driver.close()
    logger.close()
    print(f"\nChat session log saved → {logger.path}")


if __name__ == "__main__":
    chat_loop()
