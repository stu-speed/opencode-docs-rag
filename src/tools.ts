import * as fs from "node:fs";
import * as path from "node:path";

function getStringArray(val: unknown): string[] {
  return Array.isArray(val)
    ? val.filter((item): item is string => typeof item === "string")
    : [];
}

function validateSubject(value: unknown): string {
  if (
    typeof value !== "string" ||
    !/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/.test(value)
  ) {
    throw new Error("docs-rag: invalid subject name");
  }

  return value;
}

function validateFiles(value: unknown): string[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error("docs-rag: at least one file is required");
  }

  const files = value.map((item) => {
    if (typeof item !== "string" || !item.trim()) {
      throw new Error("docs-rag: invalid file path");
    }
    return path.resolve(item);
  });

  if (new Set(files).size !== files.length) {
    throw new Error("docs-rag: duplicate file paths are not allowed");
  }

  return files;
}

export function defineTools(
  editor: any,
  storage: any,
  router: any,
  _ctx: any,
  pkgRoot: string,
) {
  const dataDir = path.resolve(pkgRoot, "data", "subjects");

  function getSubjectDir(subject: string): string {
    const dir = path.resolve(dataDir, subject);
    if (!dir.startsWith(dataDir + path.sep)) {
      throw new Error("docs-rag: invalid subject directory");
    }
    return dir;
  }

  function getRawDir(subject: string): string {
    return path.join(getSubjectDir(subject), "raw");
  }


  // INGEST
  editor.add({
    name: "docs_rag_ingest",
    description: "Ingest files into a docs-rag subject.",
    input: {
      type: "object",
      properties: {
        subject: { type: "string" },
        files: {
          type: "array",
          items: { type: "string" },
          minItems: 1,
        },
        description: { type: "string" },
      },
      required: ["subject", "files"],
      additionalProperties: false,
    },
    async execute(input: any) {
      const subject = validateSubject(input.subject);
      const files = validateFiles(input.files);
      const subjectDir = getRawDir(subject);

      fs.mkdirSync(subjectDir, { recursive: true });

      const copiedFiles: string[] = [];

      for (const source of files) {
        if (!fs.existsSync(source) || !fs.statSync(source).isFile()) {
          throw new Error(`docs-rag: source file not found: ${source}`);
        }

        const dest = path.join(subjectDir, path.basename(source));

        if (fs.existsSync(dest) && path.resolve(source) !== path.resolve(dest)) {
          throw new Error(
            `docs-rag: destination already exists: ${path.basename(source)}`,
          );
        }

        if (path.resolve(source) !== path.resolve(dest)) {
          fs.copyFileSync(source, dest);
        }

        copiedFiles.push(dest);
      }

      const result = await router.runPythonModule("ingest", {
        subject,
        files: copiedFiles,
        description:
          typeof input.description === "string" ? input.description : "",
      });

      return { content: String(result) };
    },
  });

  // FORGET
  editor.add({
    name: "docs_rag_forget",
    description: "Remove files from a docs-rag subject.",
    input: {
      type: "object",
      properties: {
        subject: { type: "string" },
        files: {
          type: "array",
          items: { type: "string" },
          minItems: 1,
        },
      },
      required: ["subject", "files"],
      additionalProperties: false,
    },
    async execute(input: any) {
      const subject = validateSubject(input.subject);
      const files = validateFiles(input.files);
      const rawDir = getRawDir(subject);

      // Only accept filenames belonging to this subject's raw directory.
      const targets = files.map((file) => {
        const target = path.resolve(rawDir, path.basename(file));

        if (!target.startsWith(path.resolve(rawDir) + path.sep)) {
          throw new Error("docs-rag: invalid target path");
        }

        return target;
      });

      // Update the index first. Keep raw files if indexing fails.
      const result = await router.runPythonModule("forget", {
        subject,
        files: targets,
      });

      for (const target of targets) {
        if (fs.existsSync(target) && fs.statSync(target).isFile()) {
          fs.unlinkSync(target);
        }
      }

      return { content: String(result) };
    },
  });

  // QUERY
  editor.add({
    name: "docs_rag_query",
    description: "Search documentation using docs-rag.",
    input: {
      type: "object",
      properties: {
	question: { type: "string", minLength: 1 },
	subject: { type: "string" },
	subjects: {
	  type: "array",
	  items: { type: "string" },
	},
      },
      required: ["question"],
      additionalProperties: false,
    },
    async execute(input: any) {
      const question =
	typeof input.question === "string" ? input.question.trim() : "";

      if (!question) {
	return { content: "ERROR: question cannot be empty." };
      }

      let subjects: string[] = [];

      if (Array.isArray(input.subjects)) {
	subjects = input.subjects.map(validateSubject);
      } else if (typeof input.subject === "string") {
	subjects = [validateSubject(input.subject)];
      }

      if (!subjects.length) {
	return { content: "docs-rag: query missing subject(s)" };
      }

      const result = await router.runPythonModule("query", {
	question,
	subjects,
      });

      return { content: String(result) };
    },
  });

}

