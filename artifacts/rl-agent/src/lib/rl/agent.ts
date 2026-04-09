import { QNetwork } from "./network";
import { ReplayBuffer, Transition } from "./replay-buffer";
import { RLConfig } from "./config";

export type AgentType = "dqn" | "double-dqn";

export interface TrainingStep {
  episode: number;
  reward: number;
  loss: number;
  epsilon: number;
  steps: number;
}

export class DQNAgent {
  private qNet: QNetwork;
  private targetNet: QNetwork;
  private buffer: ReplayBuffer;
  private epsilon: number;
  private episodeCount: number = 0;
  private updateCount: number = 0;
  readonly type: AgentType;
  readonly config: RLConfig;
  private stateSize: number;
  private actionSize: number;

  constructor(stateSize: number, actionSize: number, config: RLConfig, type: AgentType = "dqn") {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    this.config = config;
    this.type = type;
    this.epsilon = config.epsilonStart;
    this.qNet = new QNetwork(stateSize, config.hiddenSize, actionSize);
    this.targetNet = new QNetwork(stateSize, config.hiddenSize, actionSize);
    this.targetNet.copyFrom(this.qNet);
    this.buffer = new ReplayBuffer(config.bufferCapacity);
  }

  selectAction(state: number[]): number {
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * this.actionSize);
    }
    const qValues = this.qNet.forward(state);
    return qValues.indexOf(Math.max(...qValues));
  }

  store(transition: Transition): void {
    this.buffer.add(transition);
  }

  train(): number {
    if (this.buffer.size < this.config.batchSize) return 0;

    const batch = this.buffer.sample(this.config.batchSize);
    const targets: number[] = [];
    const actions: number[] = [];

    for (const t of batch) {
      const currentQ = this.qNet.forward(t.state);
      let targetQ: number;

      if (t.done) {
        targetQ = t.reward;
      } else if (this.type === "double-dqn") {
        const nextQOnline = this.qNet.forward(t.nextState);
        const bestAction = nextQOnline.indexOf(Math.max(...nextQOnline));
        const nextQTarget = this.targetNet.forward(t.nextState);
        targetQ = t.reward + this.config.gamma * nextQTarget[bestAction];
      } else {
        const nextQTarget = this.targetNet.forward(t.nextState);
        targetQ = t.reward + this.config.gamma * Math.max(...nextQTarget);
      }

      currentQ[t.action] = targetQ;
      targets.push(targetQ);
      actions.push(t.action);
    }

    const loss = this.qNet.update(targets, actions, this.config.learningRate);

    this.updateCount++;
    if (this.updateCount % this.config.targetUpdateFreq === 0) {
      this.targetNet.copyFrom(this.qNet);
    }

    this.epsilon = Math.max(
      this.config.epsilonEnd,
      this.epsilon * this.config.epsilonDecay
    );

    return loss;
  }

  get currentEpsilon(): number {
    return this.epsilon;
  }

  get episodes(): number {
    return this.episodeCount;
  }

  incrementEpisode(): void {
    this.episodeCount++;
  }

  evaluate(env: { reset: () => number[]; step: (a: number) => { nextState: number[]; reward: number; done: boolean } }): number {
    let state = env.reset();
    let totalReward = 0;
    let done = false;

    while (!done) {
      const qValues = this.qNet.forward(state);
      const action = qValues.indexOf(Math.max(...qValues));
      const result = env.step(action);
      totalReward += result.reward;
      state = result.nextState;
      done = result.done;
    }

    return totalReward;
  }
}
