"""T-04: Cypher-based fraud pattern detection."""
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from config import CFG, PRESET_NAME

load_dotenv()

URI      = os.environ["NEO4J_URI"]
USER     = os.environ["NEO4J_USER"]
PASSWORD = os.environ["NEO4J_PASSWORD"]


def _build_rules():
    """Build rule definitions from config thresholds.

    Thresholds are business decisions — tune per client risk appetite:
      velocity_tx_threshold / velocity_window : card-testing aggressiveness
      drain_pct                               : account-takeover sensitivity
    """
    v_thresh = CFG["velocity_tx_threshold"]
    v_window = CFG["velocity_window"]
    drain    = CFG["drain_pct"]

    return {
        "velocity": {
            "desc": (
                f"Accounts sending >{v_thresh} transactions within {v_window} steps "
                "(card-testing pattern)"
            ),
            "cypher": """
                MATCH (src:Account)-[:SENT]->(tx:Transaction)
                WITH src, count(tx) AS txCount, max(tx.step) - min(tx.step) AS window
                WHERE txCount > $v_thresh AND window <= $v_window
                SET src.flagVelocity = true
                RETURN src.id AS account, txCount, window
                ORDER BY txCount DESC LIMIT 20
            """,
            "params": {"v_thresh": v_thresh, "v_window": v_window},
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
            "params": {},
        },
        "high_value_drain": {
            "desc": (
                f"Account drained ≥{int(drain*100)}% balance in single TRANSFER "
                "(smurfing/account takeover)"
            ),
            "cypher": """
                MATCH (src:Account)-[:SENT]->(tx:Transaction)
                WHERE tx.type = 'TRANSFER'
                  AND tx.amount >= src.balance * $drain_pct
                  AND src.balance > 0
                SET src.flagDrain = true, tx.flagDrain = true
                RETURN src.id AS account, src.balance AS balance,
                       tx.amount AS drained, tx.isFraud AS labeled
                ORDER BY drained DESC LIMIT 20
            """,
            "params": {"drain_pct": drain},
        },
    }


def _init_flags(session):
    """Set all flag properties to False on every Account before running rules.
    Prevents null values on non-matching accounts — null breaks boolean filters."""
    session.run(
        "MATCH (a:Account) "
        "SET a.flagVelocity = coalesce(a.flagVelocity, false), "
        "    a.flagMule     = coalesce(a.flagMule, false), "
        "    a.flagDrain    = coalesce(a.flagDrain, false)"
    )


def run_rules(driver):
    print(
        f"Fraud rules preset: {PRESET_NAME.upper()}  "
        f"(velocity >{CFG['velocity_tx_threshold']} txn / {CFG['velocity_window']} steps, "
        f"drain ≥{int(CFG['drain_pct']*100)}%)"
    )
    rules = _build_rules()
    with driver.session() as session:
        _init_flags(session)
        for name, rule in rules.items():
            print(f"\n{'='*60}")
            print(f"RULE: {name.upper()}")
            print(f"DESC: {rule['desc']}")
            print("-" * 60)
            result = session.run(rule["cypher"], **rule["params"])
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
