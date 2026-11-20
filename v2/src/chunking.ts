import {
  MAX_DOCUMENT_CHUNKS,
  RagError,
  type Chunk,
  type Segment,
} from "./contracts";

function boundary(text: string, position: number): number {
  const code = text.charCodeAt(position - 1);
  return code >= 0xd800 && code <= 0xdbff ? position - 1 : position;
}

export function* chunks(
  segments: readonly Segment[],
  size = 1500,
  overlap = 150,
): Generator<Chunk> {
  if (size < 4 || overlap < 0 || overlap >= size - 2)
    throw new Error("INVALID_CHUNK_SIZE");
  let index = 0;
  for (const segment of segments) {
    const sections: Array<{ start: number; end: number; path: string[] }> = [];
    const stack: Array<{ depth: number; title: string }> = [];
    let start = 0;
    let path: string[] = [];
    let offset = 0;
    let fenced = false;
    for (const line of segment.text.split("\n")) {
      if (/^\s*(```|~~~)/.test(line)) fenced = !fenced;
      const heading = fenced ? null : /^(#{1,6})[ \t]+(.+?)\s*$/.exec(line);
      if (heading) {
        sections.push({ start, end: offset, path });
        const depth = heading[1]!.length;
        while (stack.length && stack.at(-1)!.depth >= depth) stack.pop();
        const title = heading[2]!;
        stack.push({
          depth,
          title: title.slice(0, boundary(title, Math.min(512, title.length))),
        });
        path = stack.map((entry) => entry.title);
        start = offset;
      }
      offset += line.length + 1;
    }
    sections.push({ start, end: segment.text.length, path });
    for (const section of sections) {
      let position = section.start;
      while (position < section.end) {
        const end = boundary(
          segment.text,
          Math.min(position + size, section.end),
        );
        const text = segment.text.slice(position, end);
        if (text.trim()) {
          if (index >= MAX_DOCUMENT_CHUNKS)
            throw new RagError("DOCUMENT_CHUNK_LIMIT", 413);
          yield {
            index: index++,
            text,
            page: segment.kind === "page" ? segment.index : null,
            segment: segment.index,
            start: position,
            end,
            section: section.path,
          };
        }
        if (end >= section.end) break;
        position = Math.max(
          position + 1,
          boundary(segment.text, end - overlap),
        );
      }
    }
  }
}

export function documentEmbeddingInput(
  chunk: Chunk,
  title: string,
  maxInputBytes: number,
): string {
  const textBytes = Buffer.byteLength(chunk.text);
  if (!Number.isSafeInteger(maxInputBytes) || textBytes > maxInputBytes) {
    throw new RagError("EMBEDDING_INPUT_LIMIT", 422);
  }
  const header = [title, chunk.section.join(" > ")]
    .filter(Boolean)
    .join("\n\n");
  const bytes = Buffer.from(header);
  let end = Math.min(bytes.length, Math.max(0, maxInputBytes - textBytes - 2));
  while (end > 0 && end < bytes.length && (bytes[end]! & 0xc0) === 0x80) end--;
  const prefix = bytes.subarray(0, end).toString("utf8").trimEnd();
  return prefix ? `${prefix}\n\n${chunk.text}` : chunk.text;
}
