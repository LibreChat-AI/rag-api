import { fileURLToPath } from "node:url";
import type { WorkerRequest } from "./worker";
import {
  ExtractionError,
  workerResponseSchema,
  type ExtractionResult,
} from "./contract";

const workerPath = fileURLToPath(new URL("./worker.ts", import.meta.url));
export type Runner = (
  request: WorkerRequest,
  signal: AbortSignal,
) => Promise<ExtractionResult>;

export async function runWorker(
  request: WorkerRequest,
  signal: AbortSignal,
  command: readonly string[] = [process.execPath, workerPath],
): Promise<ExtractionResult> {
  signal.throwIfAborted();
  const executable = command[0];
  if (!executable) throw new ExtractionError("PARSER_UNAVAILABLE");
  const child = Bun.spawn([executable, "--no-env-file", ...command.slice(1)], {
    stdin: "pipe",
    stdout: "pipe",
    stderr: "ignore",
    // Native parsing gets no signing key, provider credential or service config.
    env: { PATH: process.env.PATH ?? "" },
  });
  const abort = () => {
    child.kill("SIGKILL");
  };
  signal.addEventListener("abort", abort, { once: true });
  const reader = child.stdout.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    signal.throwIfAborted();
    child.stdin.write(JSON.stringify(request));
    child.stdin.end();
    while (true) {
      const part = await reader.read();
      if (part.done) break;
      size += part.value.byteLength;
      if (size > request.maxOutputBytes)
        throw new ExtractionError("PARSER_OUTPUT_LIMIT");
      chunks.push(part.value);
    }
    const exit = await child.exited;
    signal.throwIfAborted();
    if (exit !== 0) throw new ExtractionError("PARSER_CRASH");
    const result = workerResponseSchema.safeParse(
      JSON.parse(Buffer.concat(chunks, size).toString("utf8")),
    );
    if (!result.success) throw new ExtractionError("PARSER_CRASH");
    if (!result.data.ok) throw new ExtractionError(result.data.code);
    return result.data.result;
  } catch (error) {
    child.kill("SIGKILL");
    await child.exited;
    await reader.cancel().catch(() => {});
    if (signal.aborted) throw new ExtractionError("REQUEST_CANCELLED");
    if (error instanceof ExtractionError) throw error;
    throw new ExtractionError("PARSER_CRASH");
  } finally {
    signal.removeEventListener("abort", abort);
    reader.releaseLock();
  }
}
