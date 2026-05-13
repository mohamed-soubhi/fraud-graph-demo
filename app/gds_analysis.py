"""GDS algorithms: Louvain, PageRank, WCC, Betweenness Centrality, Cycle Detection."""
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI      = os.environ["NEO4J_URI"]
USER     = os.environ["NEO4J_USER"]
PASSWORD = os.environ["NEO4J_PASSWORD"]

ACCT_GRAPH = "fraud-account-graph"   # Account→Account virtual edges (used by all GDS algos)


def check_gds(session):
    version = session.run("RETURN gds.version() AS v").single()["v"]
    print(f"GDS version: {version}")


def drop_graph_if_exists(session, name):
    exists = session.run(
        "CALL gds.graph.exists($name) YIELD exists", name=name
    ).single()["exists"]
    if exists:
        session.run("CALL gds.graph.drop($name)", name=name)
        print(f"Dropped existing projection: {name}")


def project_account_to_account(session):
    """Virtual Account→Account edges derived from Account→Tx→Account path."""
    session.run("""
        CALL gds.graph.project.cypher(
            $name,
            'MATCH (a:Account) RETURN id(a) AS id',
            'MATCH (src:Account)-[:SENT]->(:Transaction)-[:RECEIVED_BY]->(dst:Account)
             RETURN id(src) AS source, id(dst) AS target'
        )
    """, name=ACCT_GRAPH)
    print(f"Projected account-to-account graph: {ACCT_GRAPH}")


# ── Louvain ──────────────────────────────────────────────────────────────────

def run_louvain(session):
    print("\nRunning Louvain community detection...")
    session.run("""
        CALL gds.louvain.write($name, { writeProperty: 'community' })
        YIELD communityCount, modularity
    """, name=ACCT_GRAPH)

    result = session.run("""
        MATCH (a:Account)
        WHERE a.community IS NOT NULL
        WITH a.community AS community, count(a) AS size,
             sum(CASE WHEN a.flagVelocity OR a.flagMule OR a.flagDrain THEN 1 ELSE 0 END) AS fraudFlags
        WHERE size > 1
        RETURN community, size, fraudFlags
        ORDER BY fraudFlags DESC, size DESC
        LIMIT 15
    """)
    print("Top communities (by fraud flags):")
    for r in result:
        print(f"  community={r['community']}  size={r['size']}  fraud_flags={r['fraudFlags']}")


# ── PageRank ─────────────────────────────────────────────────────────────────

def run_pagerank(session):
    print("\nRunning PageRank (high-influence accounts)...")
    session.run("""
        CALL gds.pageRank.write($name, {
            writeProperty: 'pageRank',
            maxIterations: 20,
            dampingFactor: 0.85
        })
        YIELD nodePropertiesWritten
    """, name=ACCT_GRAPH)

    result = session.run("""
        MATCH (a:Account)
        WHERE a.pageRank IS NOT NULL
        RETURN a.id AS account, round(a.pageRank, 4) AS pageRank,
               CASE WHEN a.flagVelocity OR a.flagMule OR a.flagDrain THEN true ELSE false END AS flagged
        ORDER BY pageRank DESC LIMIT 15
    """)
    print("Top accounts by PageRank:")
    for r in result:
        flag = " *** FLAGGED" if r["flagged"] else ""
        print(f"  {r['account']}  PR={r['pageRank']}{flag}")


# ── WCC ───────────────────────────────────────────────────────────────────────

def run_wcc(session):
    print("\nRunning Weakly Connected Components (fraud ring isolation)...")
    session.run("""
        CALL gds.wcc.write($name, { writeProperty: 'wccComponent' })
        YIELD componentCount, componentDistribution
    """, name=ACCT_GRAPH)

    result = session.run("""
        MATCH (a:Account)
        WHERE a.wccComponent IS NOT NULL
        WITH a.wccComponent AS component, count(a) AS size,
             sum(CASE WHEN a.flagVelocity OR a.flagMule OR a.flagDrain THEN 1 ELSE 0 END) AS fraudFlags,
             collect(CASE WHEN a.flagVelocity OR a.flagMule OR a.flagDrain THEN a.id END)[0..3] AS sampleFlagged
        WHERE size > 1
        RETURN component, size, fraudFlags, sampleFlagged
        ORDER BY fraudFlags DESC, size DESC
        LIMIT 15
    """)
    print("Top WCC components (isolated fraud rings):")
    for r in result:
        print(f"  component={r['component']}  size={r['size']}  "
              f"fraud_flags={r['fraudFlags']}  sample={r['sampleFlagged']}")


# ── Betweenness Centrality ────────────────────────────────────────────────────

def run_betweenness(session):
    print("\nRunning Betweenness Centrality (bridge/relay accounts)...")
    session.run("""
        CALL gds.betweenness.write($name, { writeProperty: 'betweenness' })
        YIELD centralityDistribution
    """, name=ACCT_GRAPH)

    result = session.run("""
        MATCH (a:Account)
        WHERE a.betweenness IS NOT NULL
        RETURN a.id AS account,
               round(a.betweenness, 2) AS betweenness,
               CASE WHEN a.flagVelocity OR a.flagMule OR a.flagDrain THEN true ELSE false END AS flagged
        ORDER BY betweenness DESC LIMIT 15
    """)
    print("Top accounts by Betweenness (money relay hubs):")
    for r in result:
        flag = " *** FLAGGED" if r["flagged"] else ""
        print(f"  {r['account']}  BC={r['betweenness']}{flag}")


# ── Cycle Detection (Cypher) ──────────────────────────────────────────────────
# GDS triangleCount requires UNDIRECTED projection; for directed fraud graphs
# Cypher cycle detection is more precise and fraud-specific.

def run_cycle_detection(session):
    print("\nRunning Cycle Detection (A→B→C→A circular layering patterns)...")

    # 3-hop cycles: A→B→C→A
    result = session.run("""
        MATCH (a:Account)-[:SENT]->(:Transaction)-[:RECEIVED_BY]->(b:Account)
              -[:SENT]->(:Transaction)-[:RECEIVED_BY]->(c:Account)
              -[:SENT]->(:Transaction)-[:RECEIVED_BY]->(a)
        WHERE a <> b AND b <> c AND a <> c
        WITH a, b, c
        SET a.triangleCount = coalesce(a.triangleCount, 0) + 1,
            b.triangleCount = coalesce(b.triangleCount, 0) + 1,
            c.triangleCount = coalesce(c.triangleCount, 0) + 1
        RETURN a.id AS a, b.id AS b, c.id AS c
        LIMIT 20
    """)
    rows = list(result)
    if rows:
        print(f"Found {len(rows)} 3-hop cycles (showing up to 20):")
        for r in rows:
            print(f"  {r['a']} → {r['b']} → {r['c']} → {r['a']}")
    else:
        print("No 3-hop cycles found — checking 2-hop cycles (A→B→A)...")
        result2 = session.run("""
            MATCH (a:Account)-[:SENT]->(:Transaction)-[:RECEIVED_BY]->(b:Account)
                  -[:SENT]->(:Transaction)-[:RECEIVED_BY]->(a)
            WHERE a <> b
            WITH a, b, count(*) AS txPairs
            SET a.triangleCount = coalesce(a.triangleCount, 0) + txPairs,
                b.triangleCount = coalesce(b.triangleCount, 0) + txPairs
            RETURN a.id AS a, b.id AS b, txPairs
            ORDER BY txPairs DESC LIMIT 15
        """)
        rows2 = list(result2)
        if rows2:
            print("2-hop cycles (bilateral round-trip flows):")
            for r in rows2:
                print(f"  {r['a']} ↔ {r['b']}  ({r['txPairs']} tx pairs)")
        else:
            print("No circular flows detected in this dataset slice.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with driver.session() as session:
        check_gds(session)

        drop_graph_if_exists(session, ACCT_GRAPH)
        project_account_to_account(session)

        run_louvain(session)
        run_pagerank(session)
        run_wcc(session)
        run_betweenness(session)
        run_cycle_detection(session)

    driver.close()
    print("\nGDS analysis complete.")
    print("Node properties written: community, pageRank, wccComponent, betweenness, triangleCount")


if __name__ == "__main__":
    main()
