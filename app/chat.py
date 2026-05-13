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
Fraud flags on Account: flagVelocity, flagMule, flagDrain (boolean)
fraudProb: float [0,1] — GraphSAGE GNN fraud probability (written by gnn_train.py)
"""

PROMPT = PromptTemplate.from_template("""
You are a Neo4j expert. Convert the user's question into a valid Cypher query.
Return ONLY the Cypher query, no explanation.

{schema}

Extracted entities from the question:
{entities}

Question: {question}

Cypher:
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
                        error: str | None):
        total_ms = cypher_ms + (query_ms if query_ms else 0)
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
        block += f"  Total round-trip: {total_ms:.0f}ms\n"
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

def chat_loop():
    nlp    = load_nlp()
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    llm    = OllamaCloudLLM(model=OLLAMA_MODEL, base_url=OLLAMA_URL, api_key=OLLAMA_KEY)
    chain  = PROMPT | llm
    logger = ChatLogger()

    print(f"Fraud Graph Chat — type a question, 'quit' to exit")
    print(f"Session log → {logger.path}\n")

    demo_questions = [
        "Which accounts sent over $100k to flagged accounts?",
        "Show me the top 5 most suspicious accounts by PageRank",
        "Which accounts have fraudProb > 0.8 but no rule flags set?",
        "Find accounts that emptied their balance in a single transfer",
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

        print("Generating Cypher...")
        t0 = time.perf_counter()
        cypher = chain.invoke({
            "schema":   SCHEMA,
            "entities": entity_str,
            "question": question,
        }).strip()
        cypher_ms = (time.perf_counter() - t0) * 1000
        print(f"\nCypher [{cypher_ms:.0f}ms]:\n{cypher}\n")

        rows      = None
        query_ms  = 0.0
        error     = None
        try:
            t1 = time.perf_counter()
            with driver.session() as session:
                rows = session.run(cypher).data()
            query_ms = (time.perf_counter() - t1) * 1000
            if rows:
                print(f"Results ({len(rows)} rows) [{query_ms:.0f}ms]:")
                for r in rows[:10]:
                    print(f"  {r}")
                if len(rows) > 10:
                    print(f"  … {len(rows)-10} more rows")
            else:
                print(f"No results. [{query_ms:.0f}ms]")
        except Exception as e:
            error = str(e)
            print(f"Query error: {error}")

        logger.log_interaction(
            question=question,
            entities=entity_str,
            cypher=cypher,
            cypher_ms=cypher_ms,
            rows=rows,
            query_ms=query_ms,
            error=error,
        )
        print(f"  (logged)\n")

    driver.close()
    logger.close()
    print(f"\nChat session log saved → {logger.path}")


if __name__ == "__main__":
    chat_loop()
