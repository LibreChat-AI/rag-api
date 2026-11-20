import { ExtractionError } from "./contract";

type Release = () => void;
type Waiter = {
  signal: AbortSignal;
  resolve: (release: Release) => void;
  reject: (error: Error) => void;
  abort: () => void;
};

export class Admission {
  private active = 0;
  private readonly waiting: Waiter[] = [];
  constructor(
    private readonly concurrent: number,
    private readonly queued: number,
  ) {}

  acquire(signal: AbortSignal): Promise<Release> {
    signal.throwIfAborted();
    if (this.active < this.concurrent) {
      this.active++;
      return Promise.resolve(this.release());
    }
    if (this.waiting.length >= this.queued) {
      return Promise.reject(new ExtractionError("CONCURRENCY_LIMIT"));
    }
    return new Promise((resolve, reject) => {
      const waiter: Waiter = {
        signal,
        resolve,
        reject,
        abort: () => {
          const index = this.waiting.indexOf(waiter);
          if (index >= 0) this.waiting.splice(index, 1);
          reject(new ExtractionError("REQUEST_CANCELLED"));
        },
      };
      signal.addEventListener("abort", waiter.abort, { once: true });
      this.waiting.push(waiter);
    });
  }

  private release(): Release {
    let released = false;
    return () => {
      if (released) return;
      released = true;
      const next = this.waiting.shift();
      if (next) {
        next.signal.removeEventListener("abort", next.abort);
        next.resolve(this.release());
      } else this.active--;
    };
  }
}
