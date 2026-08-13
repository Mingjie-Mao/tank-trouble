import { KillFieldAgent } from "./killfield-runtime/src/killfield/teacher.js";
import {
  CANDIDATES, LIVE_ACTION_INDICES, densityRollout,
} from "./killfield-runtime/src/killfield/score.js";
import { applyAction, makeSandbox } from "./killfield-runtime/src/killfield/sandbox.js";
import { incomingRisk } from "./killfield-runtime/src/killfield/risk.js";

const TERMINAL_WIN = 12000;
const TERMINAL_LOSS = -12000;

function sameAction(left, right) {
  return left[0] === right[0] && left[1] === right[1] && left[2] === right[2];
}

function proposalSeed(game) {
  let value = (game.seed ^ Math.imul(game.frame + 1, 0x9e3779b1)
    ^ Math.imul(game.roundNumber + 1, 0x85ebca6b)) >>> 0;
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d) >>> 0;
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b) >>> 0;
  return (value ^ (value >>> 16)) >>> 0;
}

/** Advance only the first stage; the continuation is evaluated separately. */
function advanceRoot(game, action, frames, opponentModel, seed) {
  const sandbox = makeSandbox(game, opponentModel, seed);
  applyAction(sandbox, action);
  for (let frame = 0; frame < frames; frame += 1) {
    if (frame === 1) sandbox.tanks[0].fire = false;
    const events = sandbox.step();
    if (!sandbox.tanks[0].alive) return { sandbox, terminal: TERMINAL_LOSS + frame };
    if (!sandbox.tanks[1].alive) return { sandbox, terminal: TERMINAL_WIN - frame };
    if (events.some((event) => event[0] === "round_end")) break;
  }
  return { sandbox, terminal: null };
}

/**
 * Sparse two-stage search. H36 remains the authority in clear states. Only a
 * close first-action decision or immediate bullet danger opens the extra
 * branch `a0 × 8 frames -> a1 × 28 frames`.
 */
export class TwoStageSearchAgent extends KillFieldAgent {
  constructor(options = {}) {
    super(options);
    this.splitFrames = Number(options.splitFrames ?? 8);
    this.rootCount = Number(options.rootCount ?? 3);
    this.marginThreshold = Number(options.marginThreshold ?? 30);
    this.riskThreshold = Number(options.riskThreshold ?? 0.45);
    this.shadowOnly = Boolean(options.shadowOnly ?? false);
    this.twoStageCalls = 0;
    this.twoStageChanges = 0;
    this.twoStageBranches = 0;
    this.twoStageMs = [];
    this.lastBaseMargin = Infinity;
    this.changeLog = [];
    this.pendingTwoStageAction = null;
    this.lastProposal = null;
  }

  reset() {
    super.reset();
    this.twoStageCalls = 0;
    this.twoStageChanges = 0;
    this.twoStageBranches = 0;
    this.twoStageMs = [];
    this.lastBaseMargin = Infinity;
    this.changeLog = [];
    this.pendingTwoStageAction = null;
    this.lastProposal = null;
  }

  scores(game) {
    this.lastProposal = null;
    const baseValues = super.scores(game);
    const ranked = LIVE_ACTION_INDICES.slice().sort(
      (left, right) => baseValues[right] - baseValues[left],
    );
    const bestIndex = ranked[0];
    const runnerUp = ranked[1] ?? bestIndex;
    const margin = baseValues[bestIndex] - baseValues[runnerUp];
    this.lastBaseMargin = margin;
    const risk = incomingRisk(game, this.boxes);
    if (margin > this.marginThreshold && risk < this.riskThreshold) return baseValues;

    const started = performance.now();
    this.twoStageCalls += 1;
    const field = this.field;
    const rootIndices = ranked.slice(0, this.rootCount);
    const combined = new Map();
    const secondHorizon = Math.max(8, this.horizon - this.splitFrames);
    const seed = proposalSeed(game);

    for (const rootIndex of rootIndices) {
      const rootAction = CANDIDATES[rootIndex];
      const advanced = advanceRoot(
        game, rootAction, this.splitFrames, this.oppModel, seed,
      );
      if (advanced.terminal !== null) {
        combined.set(rootIndex, advanced.terminal);
        continue;
      }

      let bestContinuation = -Infinity;
      for (const continuationIndex of LIVE_ACTION_INDICES) {
        const continuation = CANDIDATES[continuationIndex];
        const value = densityRollout(
          advanced.sandbox, continuation, field, seed,
          {
            boxes: this.boxes,
            chainState: this.chain,
            horizon: secondHorizon,
            hold: this.hold,
            oppModel: this.oppModel,
          },
        );
        this.twoStageBranches += 1;
        if (value > bestContinuation) bestContinuation = value;
      }
      combined.set(rootIndex, 0.20 * baseValues[rootIndex] + 0.80 * bestContinuation);
    }

    let chosenIndex = bestIndex;
    let chosenValue = combined.get(bestIndex) ?? baseValues[bestIndex];
    for (const rootIndex of rootIndices) {
      const value = combined.get(rootIndex) ?? -Infinity;
      if (value > chosenValue) {
        chosenIndex = rootIndex;
        chosenValue = value;
      }
    }

    const output = Float64Array.from(baseValues);
    this.pendingTwoStageAction = null;
    if (!sameAction(CANDIDATES[chosenIndex], CANDIDATES[bestIndex])) {
      const proposal = {
        frame: game.frame,
        round: game.roundNumber,
        margin,
        risk,
        baseline: Array.from(CANDIDATES[bestIndex]),
        chosen: Array.from(CANDIDATES[chosenIndex]),
        baselineTwoStage: combined.get(bestIndex) ?? null,
        chosenTwoStage: chosenValue,
      };
      this.lastProposal = proposal;
      if (!this.shadowOnly) {
        const separation = Math.max(1, Math.abs(baseValues[bestIndex]) * 1e-6);
        output[chosenIndex] = baseValues[bestIndex] + separation;
        this.twoStageChanges += 1;
        this.pendingTwoStageAction = Array.from(CANDIDATES[chosenIndex]);
      }
      this.changeLog.push(proposal);
      if (this.changeLog.length > 200) this.changeLog.shift();
    }
    this.twoStageMs.push(performance.now() - started);
    if (this.twoStageMs.length > 600) this.twoStageMs.shift();
    return output;
  }

  act(game) {
    const action = super.act(game);
    const pending = this.pendingTwoStageAction;
    this.pendingTwoStageAction = null;
    if (!this.shadowOnly && pending !== null && sameAction(action, pending)
        && this.lastDecisionKind === "plan") {
      // The planner evaluated a0 as an eight-frame first stage. Execute that
      // exact contract live; falling back to the baseline's 2–4 frame commit
      // makes the evaluated continuation describe a state we never reach.
      this.committedAction = Array.from(action);
      this.commitRemaining = Math.max(0, this.splitFrames - 1);
      this.lastDecisionKind = "two_stage_plan";
    }
    return action;
  }

  telemetry() {
    const base = super.telemetry();
    const sorted = this.twoStageMs.slice().sort((a, b) => a - b);
    const at = (q) => sorted.length
      ? sorted[Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))]
      : 0;
    return {
      ...base,
      twoStageCalls: this.twoStageCalls,
      twoStageChanges: this.twoStageChanges,
      twoStageBranches: this.twoStageBranches,
      lastBaseMargin: this.lastBaseMargin,
      twoStageP95Ms: at(0.95),
      changeLog: this.changeLog.slice(),
    };
  }
}
