export {};

const path = process.argv[2];
if (!path) throw new Error("Missing fixture path");
await Bun.write(`${path}.pid`, String(process.pid));
await Bun.sleep(60_000);
