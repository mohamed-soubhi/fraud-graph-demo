"""T-06: Natural language → Cypher chat via LangChain + Ollama.

Pipeline: Question → spaCy NER (extract entities/amounts) → enriched prompt → LLM → Cypher → Neo4j
spaCy pre-processing reduces LLM hallucinations by grounding recognized entities before generation.
"""
import os
import re
import time
import spacy
import ollama
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
  (:Account {id, balance, pageRank, community, flagVelocity, flagMule, flagDrain})
  (:Transaction {id, amount, type, step, isFraud, isFlagged, flagDrain})
  (:Account)-[:SENT]->(:Transaction)-[:RECEIVED_BY]->(:Account)

Transaction types: PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN
Fraud flags on Account: flagVelocity, flagMule, flagDrain (boolean)
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

# PaySim account IDs start with C (customer) or M (merchant)
_ACCOUNT_RE = re.compile(r'\b[CM]\d+\b')
# monetary amounts: $100k, $1,000, 50000
_MONEY_RE = re.compile(r'\$?([\d,]+\.?\d*)\s*([kKmM]?)\b')


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
        "tx_types": [w.upper() for w in question.split() if w.upper() in TX_TYPES],
        "amounts": [],
        "spacy_ents": [],
    }

    # spaCy named entities (MONEY, ORG, CARDINAL)
    for ent in doc.ents:
        entities["spacy_ents"].append(f"{ent.text} ({ent.label_})")

    # normalize money amounts
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


def build_llm():
    return OllamaCloudLLM(model=OLLAMA_MODEL, base_url=OLLAMA_URL, api_key=OLLAMA_KEY)


def run_cypher(driver, cypher: str):
    with driver.session() as session:
        result = session.run(cypher)
        return result.data()


def chat_loop():
    nlp    = load_nlp()
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    llm    = build_llm()
    chain  = PROMPT | llm

    print("Fraud Graph Chat — type a question, 'quit' to exit\n")
    demo_questions = [
        "Which accounts sent over $100k to flagged accounts?",
        "Show me the top 5 most suspicious accounts by PageRank",
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

        entities     = extract_entities(nlp, question)
        entity_str   = format_entities(entities)
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

        try:
            t1 = time.perf_counter()
            rows = run_cypher(driver, cypher)
            query_ms = (time.perf_counter() - t1) * 1000
            if rows:
                print(f"Results ({len(rows)} rows) [{query_ms:.0f}ms]:")
                for r in rows[:10]:
                    print(f"  {r}")
            else:
                print(f"No results. [{query_ms:.0f}ms]")
        except Exception as e:
            print(f"Query error: {e}")
        print()

    driver.close()


if __name__ == "__main__":
    chat_loop()
