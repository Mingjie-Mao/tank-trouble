function movementFromTank(tank) {
  return [
    tank.backup ? 0 : tank.forward ? 2 : 1,
    tank.turnLeft ? 0 : tank.turnRight ? 2 : 1,
    tank.fire ? 1 : 0,
  ];
}
function key(action) {
  return action.join(":");
}

function copy(action) {
  return Array.from(action);
}

/**
 * Small online Markov/persistence model using only controls visible in the
 * current frame. It learns action run lengths and transitions within a round;
 * no opponent implementation, hidden state, or seed-specific table is read.
 */
export class VisibleOpponentModel {
  constructor({ minimumTransitions = 3 } = {}) {
    this.minimumTransitions = minimumTransitions;
    this.reset();
  }

  reset() {
    this.roundNumber = null;
    this.lastFrame = null;
    this.current = [1, 1, 0];
    this.currentRun = 0;
    this.duration = new Map();
    this.transitions = new Map();
    this.transitionCount = 0;
  }

  observe(game) {
    if (this.roundNumber !== game.roundNumber || this.lastFrame === null
        || game.frame < this.lastFrame) {
      this.reset();
      this.roundNumber = game.roundNumber;
    }
    if (game.frame === this.lastFrame) return;
    const observed = movementFromTank(game.tanks[1]);
    const previousKey = key(this.current);
    const observedKey = key(observed);
    if (this.lastFrame === null) {
      this.current = observed;
      this.currentRun = 1;
    } else if (previousKey === observedKey) {
      this.currentRun += Math.max(1, game.frame - this.lastFrame);
    } else {
      const history = this.duration.get(previousKey) ?? { total: 0, count: 0 };
      history.total += this.currentRun;
      history.count += 1;
      this.duration.set(previousKey, history);
      const targets = this.transitions.get(previousKey) ?? new Map();
      targets.set(observedKey, (targets.get(observedKey) ?? 0) + 1);
      this.transitions.set(previousKey, targets);
      this.transitionCount += 1;
      this.current = observed;
      this.currentRun = 1;
    }
    this.lastFrame = game.frame;
  }

  predict(frameOffset = 1) {
    if (this.transitionCount < this.minimumTransitions) return copy(this.current);
    const currentKey = key(this.current);
    const stats = this.duration.get(currentKey);
    const expectedRun = stats?.count ? stats.total / stats.count : Infinity;
    const remaining = Math.max(1, Math.round(expectedRun - this.currentRun));
    if (frameOffset <= remaining) return copy(this.current);
    const targets = this.transitions.get(currentKey);
    if (!targets?.size) return copy(this.current);
    let bestKey = currentKey;
    let bestCount = -1;
    for (const [target, count] of targets) {
      if (count > bestCount) {
        bestKey = target;
        bestCount = count;
      }
    }
    return bestKey.split(":").map(Number);
  }

  telemetry() {
    return {
      visibleOpponentTransitions: this.transitionCount,
      visibleOpponentModelReady: this.transitionCount >= this.minimumTransitions,
    };
  }
}
