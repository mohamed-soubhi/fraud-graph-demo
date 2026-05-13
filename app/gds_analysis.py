"""T-05: GDS Louvain community detection + PageRank on transaction graph."""
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI      = os.environ["NEO4J_URI"]
USER     = os.environ["NEO4J_USER"]
PASSWORD = os.environ["NEO4J_PASSWORD"]

GRAPH_NAME = "fraud-graph"


def check_gds(session):
    result = session.run("RETURN gds.version() AS v")
    version = result.single()["v"]
    print(f"GDS version: {version}")


def drop_graph_if_exists(session):
    exists = session.run(
        "CALL gds.graph.exists($name) YIELD exists", name=GRAPH_NAME
    ).single()["exists"]
    if exists:
        session.run("CALL gds.graph.drop($name)", name=GRAPH_NAME)
        print(f"Dropped existing projection: {GRAPH_NAME}")


def project_graph(session):
    session.run("""
        CALL gds.graph.project(
            $name,
            ['Account'],
            {
                SENT: {
                    type: 'SENT',
                    orientation: 'NATURAL'
                },
                RECEIVED_BY: {
                    type: 'RECEIVED_BY',
                    orientation: 'REVERSE'
                }
            }
        )
    """, name=GRAPH_NAME)
    print(f"Projected graph: {GRAPH_NAME}")


def run_louvain(session):
    print("\nRunning Louvain community detection...")
    session.run("""
        CALL gds.louvain.write($name, {
            writeProperty: 'community'
        })
        YIELD communityCount, modularity
    """, name=GRAPH_NAME)

    result = session.run("""
        MATCH (a:Account)
        WHERE a.community IS NOT NULL
        WITH a.community AS community, count(a) AS size,
             sum(CASE WHEN a.flagVelocity OR a.flagMule OR a.flagDrain THEN 1 ELSE 0 END) AS fraudFlags
        WHERE size > 1
        RETURN community, size, fraudFlags
        ORDER BY fraudFlags DESC, size DESC
        LIMIT 20
    """)
    print("\nTop communities (by fraud flags):")
    for r in result:
        print(f"  community={r['community']}  size={r['size']}  fraud_flags={r['fraudFlags']}")


def run_pagerank(session):
    print("\nRunning PageRank (identifies high-influence accounts)...")
    session.run("""
        CALL gds.pageRank.write($name, {
            writeProperty: 'pageRank',
            maxIterations: 20,
            dampingFactor: 0.85
        })
        YIELD nodePropertiesWritten
    """, name=GRAPH_NAME)

    result = session.run("""
        MATCH (a:Account)
        WHERE a.pageRank IS NOT NULL
        RETURN a.id AS account, round(a.pageRank, 4) AS pageRank,
               CASE WHEN a.flagVelocity OR a.flagMule OR a.flagDrain THEN true ELSE false END AS flagged
        ORDER BY pageRank DESC LIMIT 15
    """)
    print("\nTop accounts by PageRank:")
    for r in result:
        flag = " *** FLAGGED" if r["flagged"] else ""
        print(f"  {r['account']}  PR={r['pageRank']}{flag}")


def main():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with driver.session() as session:
        check_gds(session)
        drop_graph_if_exists(session)
        project_graph(session)
        run_louvain(session)
        run_pagerank(session)

    driver.close()
    print("\nGDS analysis complete. Open Neo4j Browser and style nodes by 'community' property.")


if __name__ == "__main__":
    main()
