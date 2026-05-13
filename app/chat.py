"""T-06: Natural language → Cypher chat via LangChain + Ollama."""
import os
from neo4j import GraphDatabase
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

URI          = os.environ["NEO4J_URI"]
USER         = os.environ["NEO4J_USER"]
PASSWORD     = os.environ["NEO4J_PASSWORD"]
OLLAMA_URL   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_KEY   = os.environ.get("OLLAMA_API_KEY", "")

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

Question: {question}

Cypher:
""")


def build_llm():
    kwargs = {"base_url": OLLAMA_URL, "model": OLLAMA_MODEL}
    if OLLAMA_KEY:
        kwargs["headers"] = {"Authorization": f"Bearer {OLLAMA_KEY}"}
    return OllamaLLM(**kwargs)


def run_cypher(driver, cypher: str):
    with driver.session() as session:
        result = session.run(cypher)
        return result.data()


def chat_loop():
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

        print("\nGenerating Cypher...")
        cypher = chain.invoke({"schema": SCHEMA, "question": question}).strip()
        print(f"\nCypher:\n{cypher}\n")

        try:
            rows = run_cypher(driver, cypher)
            if rows:
                print(f"Results ({len(rows)} rows):")
                for r in rows[:10]:
                    print(f"  {r}")
            else:
                print("No results.")
        except Exception as e:
            print(f"Query error: {e}")
        print()

    driver.close()


if __name__ == "__main__":
    chat_loop()
