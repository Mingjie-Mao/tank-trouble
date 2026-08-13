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
});
