import { LEARNED_SEARCH_MODEL } from "./learned-search-model.js";

function sigmoid(value) {
  return value >= 0
    ? 1 / (1 + Math.exp(-value))
    : Math.exp(value) / (1 + Math.exp(value));
}
function softmax(logits) {
  const maximum = Math.max(...logits);
  const values = logits.map((value) => Math.exp(value - maximum));
  const total = values.reduce((sum, value) => sum + value, 0);
  return values.map((value) => value / total);
}

/** Browser-native forward pass for the small shared gate/action-prior MLP. */
export function inferLearnedSearch(features) {
  const model = LEARNED_SEARCH_MODEL;
  if (features.length !== model.featureCount) {
    throw new Error(`learned-search feature mismatch: ${features.length} != ${model.featureCount}`);
  }
  const hidden = model.hiddenBias.slice();
  for (let h = 0; h < model.hiddenCount; h += 1) {
    for (let j = 0; j < model.featureCount; j += 1) {
      const value = (features[j] - model.mean[j]) / model.scale[j];
      hidden[h] += model.inputWeights[h][j] * value;
    }
    hidden[h] = Math.max(0, hidden[h]);
  }

  let gateLogit = model.gateBias;
  for (let h = 0; h < model.hiddenCount; h += 1) {
    gateLogit += model.gateWeights[h] * hidden[h];
  }
  const logits = model.priorBias.slice();
  for (let action = 0; action < model.classes; action += 1) {
    for (let h = 0; h < model.hiddenCount; h += 1) {
      logits[action] += model.priorWeights[action][h] * hidden[h];
    }
  }
  return { gate: sigmoid(gateLogit), prior: softmax(logits) };
}
