import assert from "node:assert/strict";
import test from "node:test";

import { playLeagueGame } from "../lib/browser-league.js";
import {
  DODGER_INTERCEPT_REGRESSION, DODGER_TIMEOUT_REGRESSION,
} from "../lib/tactical-v2-regression.js";

test("Tactical v2 resolves the frozen Dodger timeout regression set", () => {
  for (const regression of DODGER_TIMEOUT_REGRESSION) {
    const result = playLeagueGame({
      candidate: "p27-js-tactical-v2",
      opponent: "dodger-js",
      candidateSide: regression.candidateSide,
      seed: regression.seed,
      maxFrames: 3000,
    });
    assert.equal(
      result.outcome,
      "win",
      `seed ${regression.seed} side ${regression.candidateSide} regressed`,
    );
  }
});

test("Tactical v2 improves the held-out moving-target interception case", () => {
  for (const regression of DODGER_INTERCEPT_REGRESSION) {
    const result = playLeagueGame({
      candidate: "p27-js-tactical-v2",
      opponent: "dodger-js",
      candidateSide: regression.candidateSide,
      seed: regression.seed,
      maxFrames: 3000,
    });
    if (regression.expected === "non_loss") {
      assert.notEqual(result.outcome, "loss");
    } else {
      assert.equal(result.outcome, "win");
    }
  }
});
