#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_id="$(date +%s)-$$"
ch="ragv2-ch-$run_id"
runner="ragv2-bun-$run_id"
app="ragv2-app-$run_id"
image="ragv2-image-$run_id"
manifest_mode=$(stat -c '%a' package.json)
cleanup() {
  chmod "$manifest_mode" package.json
  docker rm -fv "$app" "$runner" "$ch" >/dev/null 2>&1 || true
  docker image rm "$image" >/dev/null 2>&1 || true
}
trap cleanup EXIT
# No published database port or application credentials.
docker run -d --name "$ch" --network none --cpus 2 --memory 2g \
  -e CLICKHOUSE_SKIP_USER_SETUP=1 clickhouse/clickhouse-server:26.8 >/dev/null
for attempt in $(seq 1 60); do
  if docker exec "$ch" clickhouse-client --query 'SELECT 1' >/dev/null 2>&1; then break; fi
  if [ "$attempt" = 60 ]; then printf 'ClickHouse did not become ready\n' >&2; exit 1; fi
  sleep 1
done
docker run -d --name "$runner" --network "container:$ch" --entrypoint sh \
  -e RAG_V2_TEST_CLICKHOUSE_URL=http://127.0.0.1:8123 oven/bun:1.4.2-debian -c 'sleep 300' >/dev/null
docker exec "$runner" mkdir /app
docker cp . "$runner:/app/v2"
docker exec -w /app/v2 "$runner" bun test test/integration.test.ts
# Qualify the shipped non-root image even when checkout files are owner-only.
chmod 600 package.json
docker build --quiet -t "$image" .
chmod "$manifest_mode" package.json
docker exec "$ch" clickhouse-client --query 'CREATE DATABASE rag_v2' >/dev/null
jwks=$(docker exec -w /app/v2 "$runner" bun scripts/runtime.ts prepare)
docker run -d --name "$app" --network "container:$ch" --read-only --cap-drop ALL \
  --security-opt no-new-privileges -e RAG_V2_JWKS_JSON="$jwks" \
  -e RAG_V2_CLICKHOUSE_URL=http://127.0.0.1:8123 \
  -e RAG_V2_EMBEDDING_PROVIDER=test -e RAG_V2_ALLOW_TEST_PROVIDER=true \
  -e RAG_V2_EMBEDDING_DIMENSIONS=64 -e RAG_V2_MODE=coordinator \
  -e RAG_V2_SINGLE_WRITER=true "$image" sh -c 'bun dist/migrate.js && exec bun dist/main.js' >/dev/null
test "$(docker inspect "$app" --format '{{.Config.User}}')" = bun
for attempt in $(seq 1 30); do
  if docker exec "$runner" bun --eval 'fetch("http://127.0.0.1:8001/health").then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))' >/dev/null 2>&1; then break; fi
  if [ "$attempt" = 30 ]; then docker logs "$app"; exit 1; fi
  sleep 1
done
docker exec -w /app/v2 "$runner" bun scripts/runtime.ts check
if [ "${RAG_V2_RUN_BENCH:-false}" = true ]; then
  docker exec -w /app/v2 "$runner" bun scripts/benchmark.ts
fi
