
import { spawn, type ChildProcess } from "node:child_process";
import * as path from "node:path";

const ENGINE_TIMEOUT_MS = 30 * 60_000;
const KILL_GRACE_MS = 1_500;
const OUTPUT_CAP_BYTES = 128 * 1024;
const STDERR_TAIL_CHARS = 8 * 1024;

interface EngineResult {
  ok: boolean;
  stdout: string;
}

export function createRouter(ctx: any, scriptsDir: string) {
  async function runPythonModule(
    name: string,
    args: unknown,
  ): Promise<string> {
    if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
      throw new Error(`docs-rag: invalid Python module name: ${name}`);
    }

    const script = path.join(scriptsDir, `${name}.py`);

    const jsonArgs = JSON.stringify(args);
    if (jsonArgs === undefined) {
      throw new Error("docs-rag: arguments are not JSON-serializable");
    }

    const res = await runEngine(
      script,
      name,
      [jsonArgs],
      ctx.signal,
    );

    if (!res.ok) {
      throw new Error(res.stdout);
    }

    return res.stdout;
  }

  return { runPythonModule };
}

function runEngine(
  script: string,
  label: string,
  args: string[],
  signal?: AbortSignal,
): Promise<EngineResult> {
  return new Promise((resolve) => {
    let settled = false;
    let timer: NodeJS.Timeout | undefined;
    let killTimer: NodeJS.Timeout | undefined;
    let child: ChildProcess | undefined;
    let timedOut = false;
    let aborted = false;

    let out = "";
    let outBytes = 0;
    let errTail = "";
    let capped = false;

    const fail = (reason: string, detail?: string): EngineResult => ({
      ok: false,
      stdout: `ERROR: docs-rag ${label} failed (${reason})\n${
        detail?.trim() || "no output"
      }`,
    });

    const finish = (result: EngineResult) => {
      if (settled) return;
      settled = true;

      if (timer) clearTimeout(timer);
      if (killTimer) clearTimeout(killTimer);

      signal?.removeEventListener("abort", abortHandler);
      resolve(result);
    };

    const detail = () => errTail.trim() || out.trim().slice(-2000);

    const terminate = (reason: "timeout" | "aborted") => {
      if (settled) return;

      if (reason === "timeout") timedOut = true;
      else aborted = true;

      if (
        !child ||
        child.exitCode !== null ||
        child.signalCode !== null
      ) {
        finish(fail(reason, detail()));
        return;
      }

      try {
        child.kill("SIGTERM");
      } catch {
        // Continue to the forced-termination fallback.
      }

      killTimer = setTimeout(() => {
        try {
          if (
            child &&
            child.exitCode === null &&
            child.signalCode === null
          ) {
            child.kill("SIGKILL");
          }
        } catch {
          // Report the original timeout/abort below.
        }

        finish(fail(reason, detail()));
      }, KILL_GRACE_MS);

      killTimer.unref?.();
    };

    function abortHandler() {
      terminate("aborted");
    }

    if (signal?.aborted) {
      finish(fail("aborted"));
      return;
    }

    try {
      child = spawn("python3", [script, ...args], {
        stdio: ["ignore", "pipe", "pipe"],
      });
    } catch (err) {
      finish(fail("spawn error", String(err)));
      return;
    }

    signal?.addEventListener("abort", abortHandler, { once: true });

    // Handle a signal that became aborted during startup.
    if (signal?.aborted) {
      abortHandler();
    }

    timer = setTimeout(() => terminate("timeout"), ENGINE_TIMEOUT_MS);
    timer.unref?.();

    child.stdout?.on("data", (chunk: Buffer) => {
      const size = Buffer.isBuffer(chunk)
        ? chunk.length
        : Buffer.byteLength(String(chunk));

      if (outBytes + size > OUTPUT_CAP_BYTES) {
        capped = true;
        return;
      }

      outBytes += size;
      out += String(chunk);
    });

    child.stderr?.on("data", (chunk: Buffer) => {
      errTail = (errTail + String(chunk)).slice(-STDERR_TAIL_CHARS);
    });

    child.on("error", (err) => {
      finish(fail("spawn error", String(err)));
    });

    child.on("close", (code, sig) => {
      if (timedOut) {
        finish(fail("timeout", detail()));
        return;
      }

      if (aborted) {
        finish(fail("aborted", detail()));
        return;
      }

      if (capped) {
        finish(fail("output exceeded 128 KiB"));
        return;
      }

      const trimmed = out.trim();

      if (code === 0) {
        finish({
          ok: true,
          stdout: trimmed || "(no output)",
        });
        return;
      }

      if (code === null && sig) {
        finish(fail(`killed by ${sig}`, detail()));
        return;
      }

      finish(fail(`exit ${String(code)}`, detail()));
    });
  });
}
