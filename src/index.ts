
import { Plugin } from "@opencode/plugin";
import * as path from "node:path";
import * as fs from "node:fs";
import { fileURLToPath } from "node:url";

import { defineTools } from "./tools.js";
import { createRouter } from "./router.js";

function findPkgRoot(): string {
  const moduleDir = path.dirname(fileURLToPath(import.meta.url));

  // Supports both src/index.ts and dist/index.js layouts.
  const candidates = [
    moduleDir,
    path.resolve(moduleDir, ".."),
  ];

  for (const candidate of candidates) {
    if (
      fs.existsSync(path.join(candidate, "python")) &&
      fs.existsSync(path.join(candidate, "package.json"))
    ) {
      return candidate;
    }
  }

  throw new Error(
    `docs-rag: cannot find package root from ${moduleDir}`,
  );
}

export default Plugin.define({
  id: "docs-rag",

  async setup(ctx) {
    const storage = ctx.storage;
    const pkgRoot = findPkgRoot();
    const scriptsDir = path.join(pkgRoot, "python");

    const router = createRouter(ctx, scriptsDir);

    await ctx.tool.transform((editor) => {
      defineTools(editor, storage, router, ctx, pkgRoot);
    });
  },
});
