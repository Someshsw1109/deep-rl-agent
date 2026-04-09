export interface RLConfig {
  episodes: number;
  maxSteps: number;
  gamma: number;
  learningRate: number;
  epsilonStart: number;
  epsilonEnd: number;
  epsilonDecay: number;
  batchSize: number;
  bufferCapacity: number;
  targetUpdateFreq: number;
  hiddenSize: number;
}

export const DEFAULT_CONFIG: RLConfig = {
  episodes: 300,
  maxSteps: 500,
  gamma: 0.99,
  learningRate: 0.001,
  epsilonStart: 1.0,
  epsilonEnd: 0.01,
  epsilonDecay: 0.995,
  batchSize: 64,
  bufferCapacity: 10000,
  targetUpdateFreq: 10,
  hiddenSize: 128,
};
