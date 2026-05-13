"""T-04: Cypher-based fraud pattern detection."""
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI      = os.environ["NEO4J_URI"]
USER     = os.environ["NEO4J_USER"]
PASSWORD = os.environ["NEO4J_PASSWORD"]


RULES = {
    "velocity": {
        "desc": "Accounts sending >3 transactions within 10 steps (card-testing pattern)",
        "cypher": """
            MATCH (src:Account)-[:SENT]->(tx:Transaction)
            WITH src, count(tx) AS txCount, max(tx.step) - min(tx.step) AS window
            WHERE txCount > 3 AND window <= 10
            SET src.flagVelocity = true
            RETURN src.id AS account, txCount, window
            ORDER BY txCount DESC LIMIT 20
        """,
    },
    "mule_chain": {
        "desc": "Money path A→B→C→cashout (layering pattern)",
        "cypher": """
            MATCH path = (origin:Account)-[:SENT]->(:Transaction)-[:RECEIVED_BY]->
                         (mid:Account)-[:SENT]->(:Transaction)-[:RECEIVED_BY]->
                         (cashout:Account)-[:SENT]->(t:Transaction)-[:RECEIVED_BY]->(:Account)
            WHERE t.type IN ['CASH_OUT', 'TRANSFER']
              AND origin.id <> cashout.id
            SET mid.flagMule = true
            RETURN origin.id AS origin, mid.id AS mule, cashout.id AS cashout,
                   t.amount AS finalAmount
            ORDER BY finalAmount DESC LIMIT 20
        """,
    },
    "high_value_drain": {
        "desc": "Account fully emptied in a single TRANSFER (smurfing/account takeover)",
        "cypher": """
            MATCH (src:Account)-[:SENT]->(tx:Transaction)
            WHERE tx.type = 'TRANSFER'
              AND tx.amount >= src.balance * 0.95
              AND src.balance > 0
            SET src.flagDrain = true, tx.flagDrain = true
            RETURN src.id AS account, src.balance AS balance,
                   tx.amount AS drained, tx.isFraud AS labeled
            ORDER BY drained DESC LIMIT 20
        """,
    },
}


def run_rules(driver):
    with driver.session() as session:
        for name, rule in RULES.items():
            print(f"\n{'='*60}")
            print(f"RULE: {name.upper()}")
            print(f"DESC: {rule['desc']}")
            print("-" * 60)
            result = session.run(rule["cypher"])
            records = result.data()
            if records:
                for r in records[:10]:
                    print(r)
                print(f"  ... {len(records)} total matches")
            else:
                print("  No matches (dataset may be subset — expected with sample)")


def main():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    run_rules(driver)
    driver.close()
    print("\nFraud flags written to graph as node/rel properties.")


if __name__ == "__main__":
    main()
