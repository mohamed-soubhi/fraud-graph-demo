"""T-03: Load PaySim CSV into Neo4j graph."""
import os, glob
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

URI      = os.environ["NEO4J_URI"]
USER     = os.environ["NEO4J_USER"]
PASSWORD = os.environ["NEO4J_PASSWORD"]
LIMIT    = int(os.environ.get("LOAD_LIMIT", 50_000))

CSV_PATH = os.environ.get("PAYSIM_CSV_PATH", "/app/data/PS_20174392719_1491204439457_log.csv")
# fallback: pick any csv in /app/data/
if not os.path.exists(CSV_PATH):
    matches = glob.glob("/app/data/*.csv")
    if matches:
        CSV_PATH = matches[0]

BATCH = 500


def create_constraints(session):
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Transaction) REQUIRE t.id IS UNIQUE")


def ingest_batch(session, rows):
    session.run("""
        UNWIND $rows AS r
        MERGE (src:Account {id: r.nameOrig})
          ON CREATE SET src.balance = r.oldbalanceOrg
        MERGE (dst:Account {id: r.nameDest})
          ON CREATE SET dst.balance = r.oldbalanceDest
        MERGE (tx:Transaction {id: r.txId})
          SET tx.amount    = r.amount,
              tx.type      = r.type,
              tx.step      = r.step,
              tx.isFraud   = r.isFraud,
              tx.isFlagged = r.isFlaggedFraud
        MERGE (src)-[:SENT]->(tx)
        MERGE (tx)-[:RECEIVED_BY]->(dst)
    """, rows=rows)


def main():
    print(f"Loading {LIMIT} rows from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, nrows=LIMIT)
    df["txId"] = df.index.astype(str)

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with driver.session() as session:
        create_constraints(session)

        batches = [df.iloc[i:i+BATCH] for i in range(0, len(df), BATCH)]
        for batch in tqdm(batches, desc="Ingesting"):
            records = batch.to_dict("records")
            ingest_batch(session, records)

    driver.close()

    print("\nDone. Verify:")
    print("  MATCH (n) RETURN labels(n), count(n)")


if __name__ == "__main__":
    main()
