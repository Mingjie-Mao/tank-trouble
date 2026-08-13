import assert from "node:assert/strict";
import test from "node:test";

import { runSuite, summarise } from "../lib/killfield-runtime/test/suite.js";

test("vendored JavaScript mechanics retain upstream parity", () => {
  const results = runSuite();
  const summary = summarise(results);
  const failures = results.flatMap((group) => group.checks
    .filter((check) => !check.pass)
    .map((check) => `${group.name}: ${check.name} ${check.detail}`));

  assert.equal(summary.fail, 0, failures.join("\n"));
  assert.equal(summary.total, 46);
});
