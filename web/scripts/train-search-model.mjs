import { readFile, writeFile } from "node:fs/promises";

const input = process.argv[2] ?? "data/search-teacher.json";
const output = process.argv[3] ?? "lib/learned-search-model.js";
const reportPath = process.argv[4] ?? "data/search-model-report.json";
const dataset = JSON.parse(await readFile(input, "utf8"));
const samples = dataset.samples.filter((sample) => (
  sample.fallbackClass >= 0 && Number.isFinite(sample.teacherRegret)
));
const validation = samples.filter((sample) => sample.seed % 5 === 0);
const training = samples.filter((sample) => sample.seed % 5 !== 0);
const classes = 10;
const baseFeatureCount = dataset.featureCount;
const featureCount = baseFeatureCount + classes;
const hiddenCount = 48;
const safeRegret = 5;
const mean = new Array(featureCount).fill(0);
const scale = new Array(featureCount).fill(0);

function rawVector(sample) {
  const values = sample.features.slice();
  for (let index = 0; index < classes; index += 1) {
    values.push(index === sample.fallbackClass ? 1 : 0);
  }
  return values;
}

for (const sample of training) {
  const values = rawVector(sample);
  for (let j = 0; j < featureCount; j += 1) mean[j] += values[j];
}
for (let j = 0; j < featureCount; j += 1) mean[j] /= Math.max(1, training.length);
for (const sample of training) {
  const values = rawVector(sample);
  for (let j = 0; j < featureCount; j += 1) {
    const delta = values[j] - mean[j];
    scale[j] += delta * delta;
  }
}
for (let j = 0; j < featureCount; j += 1) {
  scale[j] = Math.sqrt(scale[j] / Math.max(1, training.length));
  if (scale[j] < 1e-6) scale[j] = 1;
}

function vector(sample) {
  return rawVector(sample).map((value, index) => (value - mean[index]) / scale[index]);
}

function sigmoid(value) {
  return value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value));
}

function softmax(logits) {
  const maximum = Math.max(...logits);
  const values = logits.map((value) => Math.exp(value - maximum));
  const total = values.reduce((sum, value) => sum + value, 0);
  return values.map((value) => value / total);
}

function random(state) {
  state.value = (1664525 * state.value + 1013904223) >>> 0;
  return state.value / 0x100000000;
}

function shuffledIndices(length, state) {
  const indices = Array.from({ length }, (_, index) => index);
  for (let i = indices.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random(state) * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  return indices;
}

const state = { value: 20260813 };
const inputWeights = Array.from({ length: hiddenCount }, () => (
  Array.from({ length: featureCount }, () => (random(state) - 0.5) * 0.2)
));
const hiddenBias = new Array(hiddenCount).fill(0);
const gateWeights = Array.from({ length: hiddenCount }, () => (random(state) - 0.5) * 0.2);
let gateBias = 0;
const priorWeights = Array.from({ length: classes }, () => (
  Array.from({ length: hiddenCount }, () => (random(state) - 0.5) * 0.2)
));
const priorBias = new Array(classes).fill(0);
const trainVectors = training.map(vector);
const epochs = 240;
const batchSize = 64;

function zeros(rows, columns) {
  return Array.from({ length: rows }, () => new Array(columns).fill(0));
}

function forward(x) {
  const hidden = hiddenBias.slice();
  for (let h = 0; h < hiddenCount; h += 1) {
    for (let j = 0; j < featureCount; j += 1) hidden[h] += inputWeights[h][j] * x[j];
    hidden[h] = Math.max(0, hidden[h]);
  }
  let gateLogit = gateBias;
  for (let h = 0; h < hiddenCount; h += 1) gateLogit += gateWeights[h] * hidden[h];
  const logits = priorBias.slice();
  for (let k = 0; k < classes; k += 1) {
    for (let h = 0; h < hiddenCount; h += 1) logits[k] += priorWeights[k][h] * hidden[h];
  }
  return { hidden, gate: sigmoid(gateLogit), prior: softmax(logits) };
}

for (let epoch = 0; epoch < epochs; epoch += 1) {
  const order = shuffledIndices(training.length, state);
  const learningRate = 0.025 * (1 - 0.8 * epoch / epochs);
  for (let start = 0; start < order.length; start += batchSize) {
    const batch = order.slice(start, start + batchSize);
    const inputGradient = zeros(hiddenCount, featureCount);
    const hiddenBiasGradient = new Array(hiddenCount).fill(0);
    const gateGradient = new Array(hiddenCount).fill(0);
    let gateBiasGradient = 0;
    const priorGradient = zeros(classes, hiddenCount);
    const priorBiasGradient = new Array(classes).fill(0);

    for (const index of batch) {
      const x = trainVectors[index];
      const sample = training[index];
      const result = forward(x);
      const gateError = result.gate - sample.skipTarget;
      gateBiasGradient += gateError;
      const hiddenGradient = new Array(hiddenCount).fill(0);
      for (let h = 0; h < hiddenCount; h += 1) {
        gateGradient[h] += gateError * result.hidden[h];
        hiddenGradient[h] += gateError * gateWeights[h];
      }

      for (let k = 0; k < classes; k += 1) {
        const error = result.prior[k] - (sample.actionClass === k ? 1 : 0);
        priorBiasGradient[k] += error;
        for (let h = 0; h < hiddenCount; h += 1) {
          priorGradient[k][h] += error * result.hidden[h];
          hiddenGradient[h] += 0.5 * error * priorWeights[k][h];
        }
      }
      for (let h = 0; h < hiddenCount; h += 1) {
        if (result.hidden[h] <= 0) continue;
        hiddenBiasGradient[h] += hiddenGradient[h];
        for (let j = 0; j < featureCount; j += 1) {
          inputGradient[h][j] += hiddenGradient[h] * x[j];
        }
      }
    }

    const denominator = Math.max(1, batch.length);
    gateBias -= learningRate * gateBiasGradient / denominator;
    for (let h = 0; h < hiddenCount; h += 1) {
      hiddenBias[h] -= learningRate * hiddenBiasGradient[h] / denominator;
      gateWeights[h] -= learningRate
        * (gateGradient[h] / denominator + 1e-4 * gateWeights[h]);
      for (let j = 0; j < featureCount; j += 1) {
        inputWeights[h][j] -= learningRate
          * (inputGradient[h][j] / denominator + 1e-4 * inputWeights[h][j]);
      }
    }
    for (let k = 0; k < classes; k += 1) {
      priorBias[k] -= learningRate * priorBiasGradient[k] / denominator;
      for (let h = 0; h < hiddenCount; h += 1) {
        priorWeights[k][h] -= learningRate
          * (priorGradient[k][h] / denominator + 1e-4 * priorWeights[k][h]);
      }
    }
  }
}

function predict(sample) {
  return forward(vector(sample));
}

const predictions = validation.map((sample) => ({ sample, ...predict(sample) }));
let gateThreshold = 1;
let bestCoverage = 0;
let gateSafeRate = 1;
let gateMeanRegret = 0;
for (let step = 500; step <= 999; step += 1) {
  const threshold = step / 1000;
  const selected = predictions.filter((row) => row.gate >= threshold);
  if (selected.length < 20) continue;
  const safeRate = selected.filter((row) => row.sample.teacherRegret <= safeRegret).length
    / selected.length;
  const coverage = selected.length / Math.max(1, predictions.length);
  if (safeRate >= 0.99 && coverage > bestCoverage) {
    gateThreshold = threshold;
    bestCoverage = coverage;
    gateSafeRate = safeRate;
    gateMeanRegret = selected.reduce((sum, row) => sum + row.sample.teacherRegret, 0)
      / selected.length;
  }
}

function topKHit(row, k) {
  const ranking = row.prior.map((probability, index) => ({ probability, index }))
    .sort((left, right) => right.probability - left.probability)
    .slice(0, k);
  return ranking.some((entry) => entry.index === row.sample.actionClass);
}

const priorTopK = {};
for (let k = 1; k < classes; k += 1) {
  priorTopK[k] = predictions.filter((row) => topKHit(row, k)).length
    / Math.max(1, predictions.length);
}

const highMargin = predictions.filter((row) => row.sample.teacherMargin >= 100);
const highMarginTopK = {};
for (let k = 1; k < classes; k += 1) {
  highMarginTopK[k] = highMargin.filter((row) => topKHit(row, k)).length
    / Math.max(1, highMargin.length);
}

const report = {
  schemaVersion: 3,
  input,
  generatedAt: new Date().toISOString(),
  trainingSamples: training.length,
  validationSamples: validation.length,
  trainingSeeds: [...new Set(training.map((sample) => sample.seed))].length,
  validationSeeds: [...new Set(validation.map((sample) => sample.seed))].length,
  safeRegret,
  skipBaseRate: validation.filter((sample) => sample.skipTarget === 1).length
    / Math.max(1, validation.length),
  gateThreshold,
  gateSafeRate,
  gateMeanRegret,
  gateCoverage: bestCoverage,
  priorTopK,
  highMarginSamples: highMargin.length,
  highMarginTopK,
};

const model = {
  schemaVersion: 3,
  featureCount,
  hiddenCount,
  classes,
  mean,
  scale,
  inputWeights,
  hiddenBias,
  gateWeights,
  gateBias,
  gateThreshold,
  priorWeights,
  priorBias,
  report,
};
await writeFile(output, `// Generated by scripts/train-search-model.mjs\nexport const LEARNED_SEARCH_MODEL = ${JSON.stringify(model)};\n`, "utf8");
await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
console.log(JSON.stringify(report, null, 2));
