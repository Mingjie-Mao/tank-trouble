import { KillFieldAgent } from "./killfield-runtime/src/killfield/teacher.js";
import { CANDIDATES, LIVE_ACTION_INDICES } from "./killfield-runtime/src/killfield/score.js";
import { applyAction, makeSandbox } from "./killfield-runtime/src/killfield/sandbox.js";

function terminalValue(game, action, horizon, opponentModel, seed) {
  const sandbox = makeSandbox(game, opponentModel, seed);
  applyAction(sandbox, action);
  let enemyDied = false;
  let meDied = false;
  let scoreWinner = undefined;
  let deathFrame = horizon + 1;
  for (let frame = 0; frame < horizon; frame += 1) {
    if (frame === 1) sandbox.tanks[0].fire = false;
    const events = sandbox.step();
    for (const event of events) {
      if (event[0] === "destroy" && event[1] === 0) {
        meDied = true;
        deathFrame = Math.min(deathFrame, frame);
      }
      if (event[0] === "destroy" && event[1] === 1) enemyDied = true;
      if (event[0] === "round_end") scoreWinner = event[1];
    }
    if (scoreWinner !== undefined) break;
  }
  let value = 0;
  if (scoreWinner === 0) value = 40000;
  else if (scoreWinner === 1) value = -40000;
  else if (scoreWinner === null) value = -30000;
  else if (meDied && enemyDied) value = -25000;
  else if (meDied) value = -35000 + deathFrame;
  else if (enemyDied) value = 25000;
  return { value, meDied, enemyDied, scoreWinner };
}

/**
 * KillField-compatible controller with an independently implemented sparse
 * terminal safety audit. It keeps the baseline choice unless the long-tail
 * rollout predicts our death and finds a strictly safer first action.
 */
export class SaferSearchAgent extends KillFieldAgent {
  constructor(options = {}) {
    super(options);
    this.safetyHorizon = Number(options.safetyHorizon ?? 96);
    this.maxSafetyCandidates = Number(options.maxSafetyCandidates ?? 4);
    this.safetyAudits = 0;
    this.safetyOverrides = 0;
    this.auditMs = [];
    this.lastCandidateScores = null;
  }

  scores(game) {
    const values = super.scores(game);
    this.lastCandidateScores = Float64Array.from(values);
    return values;
  }

  reset() {
    super.reset();
    this.safetyAudits = 0;
    this.safetyOverrides = 0;
    this.auditMs = [];
  }

  act(game) {
    const baseline = super.act(game);
    if (!game.tanks[0].alive || !game.tanks[1].alive || game.frozen) return baseline;
    // Holds are already the result of a recent search; auditing every held
    // frame would spend the whole budget repeating the same calculation.
    if (this.lastDecisionKind === "hold") return baseline;

    const started = performance.now();
    const baselineTerminal = terminalValue(
      game, baseline, this.safetyHorizon, this.oppModel, 7001 + game.frame,
    );
    if (!baselineTerminal.meDied && baselineTerminal.scoreWinner !== null) {
      this.auditMs.push(performance.now() - started);
      if (this.auditMs.length > 600) this.auditMs.shift();
      return baseline;
    }

    this.safetyAudits += 1;
    let bestAction = baseline;
    let best = baselineTerminal;
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
      const result = terminalValue(
        game, action, this.safetyHorizon, this.oppModel,
        7001 + game.frame,
      );
      if (!result.meDied && result.scoreWinner !== null && result.value > best.value) {
        best = result;
        bestAction = action;
        break;
      }
    }

    if (best.value > baselineTerminal.value && !best.meDied) {
      this.safetyOverrides += 1;
      this.commitRemaining = 0;
      const noFire = [bestAction[0], bestAction[1], bestAction[2]];
      const emitted = this.emitAction(game, noFire, "safety_override");
      this.auditMs.push(performance.now() - started);
      if (this.auditMs.length > 600) this.auditMs.shift();
      return emitted;
    }

    this.auditMs.push(performance.now() - started);
    if (this.auditMs.length > 600) this.auditMs.shift();
    return baseline;
  }

  telemetry() {
    const base = super.telemetry();
    const sorted = this.auditMs.slice().sort((a, b) => a - b);
    const at = (q) => sorted.length
      ? sorted[Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))]
      : 0;
    return {
      ...base,
      safetyAudits: this.safetyAudits,
      safetyOverrides: this.safetyOverrides,
      auditP95Ms: at(0.95),
    };
  }
}
