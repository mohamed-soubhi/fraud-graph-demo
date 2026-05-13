"""End-to-end smoke test — runs full pipeline and reports pass/fail.

Output is mirrored to logs/run_YYYY-MM-DD_HH-MM-SS.log for evidence.
"""
import os, sys
from datetime import datetime
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI      = os.environ["NEO4J_URI"]
USER     = os.environ["NEO4J_USER"]
PASSWORD = os.environ["NEO4J_PASSWORD"]

CHECKS = []

# ── logging setup ────────────────────────────────────────────────────────────

class _Tee:
    """Write to both the original stream and a log file simultaneously."""
    def __init__(self, stream, log_path: Path):
        self._stream = stream
        self._file   = log_path.open("a", buffering=1)

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    # forward everything else (isatty, fileno, etc.)
    def __getattr__(self, name):
        return getattr(self._stream, name)


def setup_logging() -> Path:
    log_dir = Path("/app/logs")
    log_dir.mkdir(exist_ok=True)
    stamp    = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"run_{stamp}.log"

    tee = _Tee(sys.stdout, log_path)
    sys.stdout = tee
    sys.stderr = tee          # capture sub-module prints & warnings too
    return log_path


# ── checks ───────────────────────────────────────────────────────────────────

def check(label, fn):
    try:
        result = fn()
        status = "PASS" if result else "FAIL"
        CHECKS.append((label, status, str(result)))
    except Exception as e:
        CHECKS.append((label, "FAIL", str(e)))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    log_path = setup_logging()

    header = f"{'='*60}\nFRAUD GRAPH DEMO — PIPELINE RUN\n{datetime.now().isoformat()}\nLog: {log_path}\n{'='*60}\n"
    print(header)

    print("[1/5] Ingesting data...")
    import ingest
    ingest.main()

    print("\n[2/5] Running fraud rules...")
    import fraud_rules
    fraud_rules.main()

    print("\n[3/5] Running GDS analysis...")
    import gds_analysis
    gds_analysis.main()

    print("\n[4/5] Running GNN (GraphSAGE)...")
    import gnn_train
    gnn_train.main()

    print("\n[5/5] Verifying graph state...")
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
        check(
            "GNN fraudProb written",
            lambda: session.run(
                "MATCH (a:Account) WHERE a.fraudProb IS NOT NULL RETURN count(a) AS c"
            ).single()["c"] > 0,
        )
    driver.close()

    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    all_pass = True
    for label, status, detail in CHECKS:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} [{status}] {label}  ({detail})")
        if status == "FAIL":
            all_pass = False

    print("=" * 60)
    print("OVERALL:", "ALL PASS" if all_pass else "FAILURES DETECTED")
    print(f"\nLog saved → {log_path}")

    # restore stdout/stderr before launching interactive chat
    tee = sys.stdout
    sys.stdout = tee._stream
    sys.stderr = tee._stream
    tee.close()

    if all_pass:
        print("\nPipeline complete — launching chat interface...\n")
        import chat
        chat.chat_loop()
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
