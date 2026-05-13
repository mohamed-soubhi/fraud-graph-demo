#!/usr/bin/env bash
set -e

echo "=== Stopping and removing existing containers ==="
docker rm -f fraud-neo4j fraud-app 2>/dev/null || true

echo "=== Building and starting containers ==="
docker compose up -d --build

echo "=== Waiting for Neo4j to be healthy ==="
until docker inspect --format='{{.State.Health.Status}}' fraud-neo4j 2>/dev/null | grep -q "healthy"; do
    printf "."
    sleep 3
done
echo " ready"

echo "=== Running full pipeline + chat ==="
docker exec -it fraud-app python /app/run_all.py
