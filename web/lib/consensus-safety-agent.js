import { KillFieldAgent } from "./killfield-runtime/src/killfield/teacher.js";
import { CANDIDATES, LIVE_ACTION_INDICES } from "./killfield-runtime/src/killfield/score.js";
import { applyAction, makeSandbox } from "./killfield-runtime/src/killfield/sandbox.js";

function outcome(game, action, horizon, opponentModel, seed) {
  const sandbox = makeSandbox(game, opponentModel, seed);
  applyAction(sandbox, action);
  let meDied = false;
  let enemyDied = false;
  let winner = undefined;
  for (let frame = 0; frame < horizon; frame += 1) {
    if (frame === 1) sandbox.tanks[0].fire = false;
    const events = sandbox.step();
    for (const event of events) {
      if (event[0] === "destroy" && event[1] === 0) meDied = true;
      if (event[0] === "destroy" && event[1] === 1) enemyDied = true;
      if (event[0] === "round_end") winner = event[1];
    }
    if (winner !== undefined) break;
  }
  return {
    unsafe: meDied || winner === null || winner === 1,
    win: winner === 0 || (enemyDied && !meDied),
  };
}

function modelSet(frame) {
  return [
    ["L1", 11003 + frame],
    ["L2", 21011 + frame],
    ["L2", 41017 + frame],
  ];
}

export class ConsensusSafetyAgent extends KillFieldAgent {
  constructor(options = {}) {
    super(options);
    this.safetyHorizon = Number(options.safetyHorizon ?? 75);
    this.maxSafetyCandidates = Number(options.maxSafetyCandidates ?? 4);
    this.lastCandidateScores = null;
    this.consensusAudits = 0;
    this.consensusOverrides = 0;
  }

  reset() {
    super.reset();
    this.lastCandidateScores = null;
    this.consensusAudits = 0;
    this.consensusOverrides = 0;
  }

  scores(game) {
    const values = super.scores(game);
    this.lastCandidateScores = Float64Array.from(values);
    return values;
  }

  act(game) {
    const baseline = super.act(game);
    if (!game.tanks[0].alive || !game.tanks[1].alive || game.frozen
        || this.lastDecisionKind === "hold") return baseline;

    const models = modelSet(game.frame);
    const baselineOutcomes = models.map(([model, seed]) => outcome(
      game, baseline, this.safetyHorizon, model, seed,
    ));
    // Fail closed on the gate: one model believing the baseline survives is
    // enough to keep the proven action and avoid speculative intervention.
    if (!baselineOutcomes.every((result) => result.unsafe)) return baseline;

    this.consensusAudits += 1;
    const ranked = LIVE_ACTION_INDICES.slice().sort((left, right) => {
      if (this.lastCandidateScores === null) return left - right;
      return this.lastCandidateScores[right] - this.lastCandidateScores[left];
    });
    let evaluated = 0;
    for (const index of ranked) {
      const action = CANDIDATES[index];
      if (action[0] === baseline[0] && action[1] === baseline[1] && action[2] === baseline[2]) {
        continue;
      }
      if (evaluated >= this.maxSafetyCandidates) break;
      evaluated += 1;
      const results = models.map(([model, seed]) => outcome(
        game, action, this.safetyHorizon, model, seed,
      ));
      if (results.every((result) => !result.unsafe)) {
        this.consensusOverrides += 1;
        this.commitRemaining = 0;
        return this.emitAction(game, action, "consensus_safety_override");
      }
    }
    return baseline;
  }

  telemetry() {
    return {
      ...super.telemetry(),
      consensusAudits: this.consensusAudits,
      consensusOverrides: this.consensusOverrides,
    };
  }
}
