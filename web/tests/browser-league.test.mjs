import assert from "node:assert/strict";
import test from "node:test";

import { playLeagueGame, runBrowserLeague } from "../lib/browser-league.js";

test("league game records the candidate on either color", () => {
  const left = playLeagueGame({
    candidate: "random-js",
    opponent: "idle-js",
    candidateSide: 0,
    seed: 1200000,
    maxFrames: 4,
  });
  const right = playLeagueGame({
    candidate: "random-js",
    opponent: "idle-js",
    candidateSide: 1,
    seed: 1200000,
    maxFrames: 4,
  });
  assert.equal(left.candidateSide, 0);
  assert.equal(right.candidateSide, 1);
  assert.equal(left.outcome, "draw");
  assert.equal(right.outcome, "draw");
  assert.equal(left.winType, null);
  assert.equal(typeof left.candidateFires, "number");
  assert.equal(typeof left.movementSwitchesPer1000, "number");
  assert.equal(typeof left.reversalsPer1000, "number");
});
test("league summary reports per-opponent color balance", () => {
  const report = runBrowserLeague({
    candidate: "random-js",
    opponents: ["idle-js"],
    roundsPerSide: 2,
    seed: 1200100,
    maxFrames: 3,
  });
  assert.equal(report.overall.games, 4);
  assert.equal(report.opponents["idle-js"].bySide.left.games, 2);
  assert.equal(report.opponents["idle-js"].bySide.right.games, 2);
  assert.equal(report.opponents["idle-js"].draw, 4);
  assert.equal(report.opponents["idle-js"].colorGap, 0);
  assert.equal(report.overall.activeWins, 0);
  assert.equal(report.overall.passiveWins, 0);
  assert.equal(typeof report.overall.fireOpportunityCaptureRate, "number");
  assert.equal(typeof report.overall.plannedOpportunityFireRate, "number");
});
