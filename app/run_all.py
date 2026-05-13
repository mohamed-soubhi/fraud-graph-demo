"""T-07: End-to-end smoke test — runs full pipeline and reports pass/fail."""
import os, sys
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI      = os.environ["NEO4J_URI"]
USER     = os.environ["NEO4J_USER"]
PASSWORD = os.environ["NEO4J_PASSWORD"]

CHECKS = []


def check(label, fn):
    try:
        result = fn()
        status = "PASS" if result else "FAIL"
        CHECKS.append((label, status, str(result)))
    except Exception as e:
        CHECKS.append((label, "FAIL", str(e)))


def main():
    print("Running full pipeline smoke test...\n")

    # 1 — ingest
    print("[1/4] Ingesting data...")
    import ingest
    ingest.main()

    # 2 — fraud rules
    print("\n[2/4] Running fraud rules...")
    import fraud_rules
    fraud_rules.main()

    # 3 — GDS
    print("\n[3/4] Running GDS analysis...")
    import gds_analysis
    gds_analysis.main()

    # 4 — verify graph state
    print("\n[4/4] Verifying graph state...")
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with driver.session() as session:
        check(
            "Account nodes loaded",
            lambda: session.run("MATCH (a:Account) RETURN count(a) AS c").single()["c"] > 0,
        )
        check(
            "Transaction nodes loaded",
            lambda: session.run("MATCH (t:Transaction) RETURN count(t) AS c").single()["c"] > 0,
        )
        check(
            "Fraud flags exist",
            lambda: session.run(
                "MATCH (a:Account) WHERE a.flagVelocity OR a.flagMule OR a.flagDrain "
                "RETURN count(a) AS c"
            ).single()["c"] > 0,
        )
        check(
            "Community property set",
            lambda: session.run(
                "MATCH (a:Account) WHERE a.community IS NOT NULL RETURN count(a) AS c"
            ).single()["c"] > 0,
        )
        check(
            "PageRank property set",
            lambda: session.run(
                "MATCH (a:Account) WHERE a.pageRank IS NOT NULL RETURN count(a) AS c"
            ).single()["c"] > 0,
        )
        check(
            "WCC component property set",
            lambda: session.run(
                "MATCH (a:Account) WHERE a.wccComponent IS NOT NULL RETURN count(a) AS c"
            ).single()["c"] > 0,
        )
        check(
            "Betweenness property set",
            lambda: session.run(
                "MATCH (a:Account) WHERE a.betweenness IS NOT NULL RETURN count(a) AS c"
            ).single()["c"] > 0,
        )
        check(
            "WCC fraud rings detected",
            lambda: session.run("""
                MATCH (a:Account)
                WHERE a.wccComponent IS NOT NULL
                WITH a.wccComponent AS c, count(a) AS size,
                     sum(CASE WHEN a.flagVelocity OR a.flagMule OR a.flagDrain THEN 1 ELSE 0 END) AS flags
                WHERE flags > 0
                RETURN count(c) AS ringsWithFraud
            """).single()["ringsWithFraud"] > 0,
        )
    driver.close()

    print("\n" + "="*50)
    print("SMOKE TEST RESULTS")
    print("="*50)
    all_pass = True
    for label, status, detail in CHECKS:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} [{status}] {label}  ({detail})")
        if status == "FAIL":
            all_pass = False

    print("="*50)
    print("OVERALL:", "ALL PASS" if all_pass else "FAILURES DETECTED")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
