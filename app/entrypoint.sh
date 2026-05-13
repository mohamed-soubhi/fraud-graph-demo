#!/bin/sh
set -e

echo "Checking if Neo4j already has data..."
COUNT=$(python - <<'EOF'
import os
from neo4j import GraphDatabase
driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
with driver.session() as s:
    result = s.run("MATCH (n) RETURN count(n) AS c").single()
    print(result["c"])
driver.close()
EOF
)

if [ "$COUNT" -gt "0" ]; then
    echo "Data exists ($COUNT nodes). Skipping ingest."
else
    echo "Empty DB. Running ingest..."
    python /app/ingest.py
fi

exec tail -f /dev/null
