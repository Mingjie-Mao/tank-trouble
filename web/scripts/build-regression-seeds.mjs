import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

function argument(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index === -1 ? fallback : process.argv[index + 1];
}

const output = argument("output", "data/regression/tactical-champion.json");
const slowWins = Number(argument("slow-wins", "16"));
const appendPath = argument("append", null);
const valuedOptions = new Set(["--output", "--slow-wins", "--append"]);
const inputs = [];
for (let index = 2; index < process.argv.length; index += 1) {
  if (valuedOptions.has(process.argv[index])) {
    index += 1;
  } else if (!process.argv[index].startsWith("--")) {
    inputs.push(process.argv[index]);
  }
}
if (inputs.length === 0) {
  throw new Error("usage: node build-regression-seeds.mjs [--output FILE] REPORT...");
}

const reports = await Promise.all(inputs.map(async (input) => ({
  input,
  report: JSON.parse(await readFile(input, "utf8")),
})));
const champion = reports[0].report.candidate;
if (reports.some(({ report }) => report.candidate !== champion)) {
  throw new Error("all reports must describe the same frozen champion");
}

const rows = reports.flatMap(({ input, report }) => report.games.map((row) => ({
  ...row,
  source: input,
})));
const rowsByKey = new Map(rows.map((row) => [
  `${row.opponent}:${row.candidateSide}:${row.seed}`, row,
]));
const appended = appendPath === null
  ? []
  : JSON.parse(await readFile(appendPath, "utf8")).cases.map((item) => {
    const row = rowsByKey.get(item.key);
    if (!row) throw new Error(`append case ${item.key} is absent from the new champion reports`);
    return { ...row, tags: item.tags };
  });
const selected = [
  ...appended,
  ...rows.filter((row) => row.outcome !== "win").map((row) => ({
    ...row,
    tags: ["hard-case", row.outcome],
  })),
  ...rows.filter((row) => row.outcome === "win")
    .sort((left, right) => right.frames - left.frames)
    .slice(0, slowWins)
    .map((row) => ({ ...row, tags: ["slow-win"] })),
];
const cases = [];
const seen = new Set();
for (const row of selected) {
  const key = `${row.opponent}:${row.candidateSide}:${row.seed}`;
  if (seen.has(key)) continue;
  seen.add(key);
  cases.push({
    key,
    opponent: row.opponent,
    candidateSide: row.candidateSide,
    seed: row.seed,
    baselineOutcome: row.outcome,
    baselineFrames: row.frames,
    baselineP95DecisionMs: row.candidateDecisionP95Ms,
    tags: row.tags,
    source: row.source,
  });
}
cases.sort((left, right) => left.key.localeCompare(right.key));

const artifact = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  champion,
  immutableSeedPolicy: "append-only after validation; never tune against one seed",
  sourceReports: inputs,
  counts: {
    total: cases.length,
    hardCases: cases.filter((item) => item.tags.includes("hard-case")).length,
    slowWins: cases.filter((item) => item.tags.includes("slow-win")).length,
  },
  cases,
};
await mkdir(path.dirname(output), { recursive: true });
await writeFile(output, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
console.log(`wrote ${cases.length} permanent regression cases to ${output}`);
