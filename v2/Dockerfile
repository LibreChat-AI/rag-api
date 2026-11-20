FROM oven/bun:1.4.2-debian AS build
WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile --ignore-scripts
COPY tsconfig.json ./
COPY src ./src
RUN bun run typecheck && bun run build
FROM oven/bun:1.4.2-debian
WORKDIR /app
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY --chmod=0444 package.json ./
USER bun
ENV RAG_V2_HOST=0.0.0.0
EXPOSE 8001
CMD ["bun", "dist/main.js"]
