FROM oven/bun:1.4.2-slim AS dependencies
WORKDIR /app
COPY service/package.json service/bun.lock ./
RUN bun install --frozen-lockfile --production

FROM oven/bun:1.4.2-slim
WORKDIR /app
COPY --from=dependencies --chown=bun:bun /app/node_modules ./node_modules
COPY --chown=bun:bun service/package.json ./package.json
COPY --chown=bun:bun service/src ./src
USER bun
EXPOSE 8001
CMD ["bun", "run", "src/server.ts"]
