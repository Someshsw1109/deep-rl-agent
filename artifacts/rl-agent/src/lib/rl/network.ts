export class Matrix {
  data: number[][];
  rows: number;
  cols: number;

  constructor(rows: number, cols: number) {
    this.rows = rows;
    this.cols = cols;
    this.data = Array.from({ length: rows }, () => Array(cols).fill(0));
  }

  static random(rows: number, cols: number): Matrix {
    const m = new Matrix(rows, cols);
    const scale = Math.sqrt(2 / rows);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        m.data[i][j] = (Math.random() * 2 - 1) * scale;
      }
    }
    return m;
  }

  static zeros(rows: number, cols: number): Matrix {
    return new Matrix(rows, cols);
  }

  clone(): Matrix {
    const m = new Matrix(this.rows, this.cols);
    m.data = this.data.map((row) => [...row]);
    return m;
  }
}

function relu(x: number): number {
  return Math.max(0, x);
}

function reluGrad(x: number): number {
  return x > 0 ? 1 : 0;
}

interface LayerState {
  input: number[];
  preActivation: number[];
  output: number[];
}

export class QNetwork {
  weights: Matrix[];
  biases: number[][];
  private layerSizes: number[];
  private layerStates: LayerState[] = [];

  constructor(inputSize: number, hiddenSize: number, outputSize: number) {
    this.layerSizes = [inputSize, hiddenSize, hiddenSize, outputSize];
    this.weights = [];
    this.biases = [];

    for (let i = 0; i < this.layerSizes.length - 1; i++) {
      this.weights.push(Matrix.random(this.layerSizes[i + 1], this.layerSizes[i]));
      this.biases.push(Array(this.layerSizes[i + 1]).fill(0));
    }
  }

  forward(input: number[]): number[] {
    this.layerStates = [];
    let current = [...input];

    for (let l = 0; l < this.weights.length; l++) {
      const W = this.weights[l];
      const b = this.biases[l];
      const preAct: number[] = [];

      for (let i = 0; i < W.rows; i++) {
        let sum = b[i];
        for (let j = 0; j < W.cols; j++) {
          sum += W.data[i][j] * current[j];
        }
        preAct.push(sum);
      }

      const isLast = l === this.weights.length - 1;
      const output = isLast ? preAct : preAct.map(relu);

      this.layerStates.push({ input: [...current], preActivation: preAct, output });
      current = output;
    }

    return current;
  }

  copyFrom(other: QNetwork): void {
    this.weights = other.weights.map((w) => w.clone());
    this.biases = other.biases.map((b) => [...b]);
  }

  update(targets: number[], actions: number[], learningRate: number): number {
    const batchSize = targets.length;
    let totalLoss = 0;

    const weightGrads = this.weights.map((w) => Matrix.zeros(w.rows, w.cols));
    const biasGrads = this.biases.map((b) => Array(b.length).fill(0));

    for (let s = 0; s < batchSize; s++) {
      const target = targets[s];
      const action = actions[s];
      const states = this.layerStates;

      if (!states || states.length === 0) continue;

      const outputIdx = action;
      const predicted = states[states.length - 1].output[outputIdx];
      const error = predicted - target;
      totalLoss += error * error;

      let delta = Array(this.layerSizes[this.layerSizes.length - 1]).fill(0);
      delta[outputIdx] = (2 * error) / batchSize;

      for (let l = this.weights.length - 1; l >= 0; l--) {
        const layerInput = states[l].input;
        const preAct = states[l].preActivation;

        for (let i = 0; i < this.weights[l].rows; i++) {
          biasGrads[l][i] += delta[i];
          for (let j = 0; j < this.weights[l].cols; j++) {
            weightGrads[l].data[i][j] += delta[i] * layerInput[j];
          }
        }

        if (l > 0) {
          const prevDelta = Array(this.weights[l].cols).fill(0);
          for (let j = 0; j < this.weights[l].cols; j++) {
            for (let i = 0; i < this.weights[l].rows; i++) {
              prevDelta[j] += this.weights[l].data[i][j] * delta[i];
            }
            prevDelta[j] *= reluGrad(preAct[j]);
          }
          delta = prevDelta;
        }
      }
    }

    for (let l = 0; l < this.weights.length; l++) {
      for (let i = 0; i < this.weights[l].rows; i++) {
        this.biases[l][i] -= learningRate * biasGrads[l][i];
        for (let j = 0; j < this.weights[l].cols; j++) {
          this.weights[l].data[i][j] -= learningRate * weightGrads[l].data[i][j];
        }
      }
    }

    return totalLoss / batchSize;
  }
}
