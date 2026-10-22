// sanity.js
// Minimal test harness for docs-rag plugin Python modules.

import { spawn } from "node:child_process";
import * as path from "node:path";
import * as fs from "node:fs";

const ROOT = process.cwd();
const PY = path.join(ROOT, "python");

function run(script, args) {
  return new Promise((resolve) => {
    const child = spawn("python3", [path.join(PY, script), JSON.stringify(args)]);
    let out = "";
    let err = "";

    child.stdout.on("data", (d) => (out += String(d)));
    child.stderr.on("data", (d) => (err += String(d)));

    child.on("close", (code) => {
      resolve({ code, out: out.trim(), err: err.trim() });
    });
  });
}

async function main() {
  console.log("=== docs-rag sanity test ===");

  // 1. Check python directory
  if (!fs.existsSync(PY)) {
    console.error("FAIL: python/ directory not found:", PY);
    process.exit(1);
  }
  console.log("OK: python directory exists");

  // 2. Check required modules
  const required = ["ingest.py", "forget.py", "query.py"];
  for (const f of required) {
    const p = path.join(PY, f);
    if (!fs.existsSync(p)) {
      console.error("FAIL: missing python module:", f);
      process.exit(1);
    }
  }
  console.log("OK: python modules present");

  // 3. Create temp subject
  const subject = "sanity";
  const rawDir = path.join(ROOT, "data", "subjects", subject, "raw");
  fs.mkdirSync(rawDir, { recursive: true });

  // 4. Write temp file
  const tmpFile = path.join(rawDir, "test.txt");
  fs.writeFileSync(tmpFile, "hello world");

  // 5. Ingest
  const ingestRes = await run("ingest.py", {
    subject,
    files: [tmpFile],
    description: "sanity test"
  });

  console.log("INGEST:", ingestRes);

  // 6. Query
  const queryRes = await run("query.py", {
    question: "what is in test.txt?",
    subjects: [subject]
  });

  console.log("QUERY:", queryRes);

  // 7. Forget
  const forgetRes = await run("forget.py", {
    subject,
    files: [tmpFile]
  });

  console.log("FORGET:", forgetRes);

  console.log("=== done ===");
}

main().catch((err) => {
  console.error("UNCAUGHT:", err);
  process.exit(1);
});

