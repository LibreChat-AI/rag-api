#!/usr/bin/env bash
set -euo pipefail
image=${1:?Pass the locally built image name}
root=$(git rev-parse --show-toplevel)
docker run --rm --entrypoint sh "$image" -c '! command -v python && ! command -v python3'
docker run --rm -i --entrypoint bun "$image" -e '
import { createApp } from "./src/app.ts";
import { SignJWT } from "jose";
const secret = "image-smoke-test-key-not-a-real-service-key";
const data = await Bun.stdin.bytes();
const application = createApp({enabled: true, secret});
const server = Bun.serve({hostname: "127.0.0.1", port: 0, fetch: application.fetch});
try {
  const health = await fetch(`http://127.0.0.1:${server.port}/health`);
  if (health.status !== 200) throw new Error("Health failed");
  const token = await new SignJWT({scopes: ["rag:documents"]}).setProtectedHeader({alg: "HS256"})
    .setIssuer("librechat").setAudience("rag-api").setSubject("owner").setExpirationTime("1m")
    .sign(new TextEncoder().encode(secret));
  const body = new FormData(); body.append("profile", "document-v1");
  body.append("file", new File([data], "report.docx", {type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}));
  const response = await fetch(`http://127.0.0.1:${server.port}/v1/extract`, {method: "POST", body, headers: {Authorization: `Bearer ${token}`}});
  const result = await response.json();
  if (response.status !== 200 || !result.text?.includes("Regional Totals") || result.parser?.name !== "anydoc") {
    throw new Error(`Native DOCX failed: ${response.status}`);
  }
  console.log("PASS: Bun listener, health and native DOCX extraction in a Python-free image");
} finally { await server.stop(true); }
' < "$root/tests/fixtures/structured.docx"
