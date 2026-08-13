import { CHAMPION_BASELINE } from "./champion-baseline.js";

function rowKey(row) {
  return `${row.opponent}:${row.candidateSide}:${row.seed}`;
}

function rate(count, games) {
  return count / Math.max(1, games);
}

function worstColorGap(report) {
  return Math.max(0, ...Object.values(report.opponents).map((item) => item.colorGap));
}

/** Compare a candidate and frozen champion on exactly the same league cases. */
export function summarizeSelfplayFlywheel({
  championReport,
  candidateReport,
  maximumPoolRegression = 0,
  maximumP95Ms = CHAMPION_BASELINE.promotionGate.maximumP95DecisionMs,
  latencyNoiseTolerance = 0,
}) {
  const championRows = new Map(championReport.games.map((row) => [rowKey(row), row]));
  const changes = [];
  const hardCases = [];
  for (const candidate of candidateReport.games) {
    const champion = championRows.get(rowKey(candidate));
    if (!champion) throw new Error(`missing champion pair ${rowKey(candidate)}`);
    const item = {
      opponent: candidate.opponent,
      candidateSide: candidate.candidateSide,
      seed: candidate.seed,
      championOutcome: champion.outcome,
      candidateOutcome: candidate.outcome,
      championFrames: champion.frames,
      candidateFrames: candidate.frames,
    };
    if (candidate.outcome !== champion.outcome) changes.push(item);
    if (candidate.outcome !== "win" || champion.outcome !== "win") hardCases.push(item);
  }

  const regressions = changes.filter((row) => (
    row.championOutcome === "win" && row.candidateOutcome !== "win"
  ));
  const recoveries = changes.filter((row) => (
    row.championOutcome !== "win" && row.candidateOutcome === "win"
  ));
  const candidate = candidateReport.overall;
  const champion = championReport.overall;
  const championColorGap = worstColorGap(championReport);
  const candidateColorGap = worstColorGap(candidateReport);
  const gates = {
    pairedCorrections: recoveries.length >= regressions.length,
    opponentPool: candidate.winRate >= champion.winRate - maximumPoolRegression,
    doubleDeath: rate(candidate.double_death, candidate.games)
      <= rate(champion.double_death, champion.games),
    draw: candidate.draw <= champion.draw,
    // The paired champion is the fairness reference. A previously published
    // absolute target remains telemetry, but a corrected evaluation must not
    // reject a candidate that improves a known champion-side imbalance.
    color: candidateColorGap <= championColorGap + 1e-12,
    latency: candidate.p95DecisionMs <= Math.min(
      maximumP95Ms,
      champion.p95DecisionMs * (1 + latencyNoiseTolerance),
    ),
  };
  return {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    champion: championReport.candidate,
    candidate: candidateReport.candidate,
    seed: candidateReport.seed,
    roundsPerSide: candidateReport.roundsPerSide,
    opponents: Object.keys(candidateReport.opponents),
    championOverall: champion,
    candidateOverall: candidate,
    winRateDelta: candidate.winRate - champion.winRate,
    doubleDeathRateDelta: rate(candidate.double_death, candidate.games)
      - rate(champion.double_death, champion.games),
    colorGapDelta: candidateColorGap - championColorGap,
    p95DecisionMsDelta: candidate.p95DecisionMs - champion.p95DecisionMs,
    regressions,
    recoveries,
    changes,
    hardCases,
    gates,
    readyForLaikaBlind: Object.values(gates).every(Boolean),
  };
}
