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
  maximumPoolRegression = CHAMPION_BASELINE.promotionGate.maximumOpponentPoolRegression,
  maximumColorGap = CHAMPION_BASELINE.promotionGate.maximumColorGap,
  maximumP95Ms = CHAMPION_BASELINE.promotionGate.maximumP95DecisionMs,
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
  const gates = {
    pairedCorrections: recoveries.length >= regressions.length,
    opponentPool: candidate.winRate >= champion.winRate - maximumPoolRegression,
    doubleDeath: rate(candidate.double_death, candidate.games)
      <= rate(champion.double_death, champion.games),
    draw: candidate.draw <= champion.draw,
    color: worstColorGap(candidateReport) <= maximumColorGap,
    latency: candidate.p95DecisionMs <= maximumP95Ms,
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
    regressions,
    recoveries,
    changes,
    hardCases,
    gates,
    readyForLaikaBlind: Object.values(gates).every(Boolean),
  };
}

