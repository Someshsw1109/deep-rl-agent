import { CartPoleEnv } from "./cartpole";
import { DQNAgent, AgentType, TrainingStep } from "./agent";
import { RLConfig } from "./config";

export interface TrainingResult {
  steps: TrainingStep[];
  evalRewards: number[];
  finalAvgReward: number;
}

export type ProgressCallback = (step: TrainingStep, progress: number) => void;

export async function trainAgent(
  config: RLConfig,
  type: AgentType,
  onProgress: ProgressCallback,
  signal?: AbortSignal
): Promise<TrainingResult> {
  const env = new CartPoleEnv();
  const agent = new DQNAgent(env.stateSize, env.actionSize, config, type);

  const steps: TrainingStep[] = [];
  const evalRewards: number[] = [];

  for (let episode = 0; episode < config.episodes; episode++) {
    if (signal?.aborted) break;

    let state = env.reset();
    let episodeReward = 0;
    let episodeLoss = 0;
    let trainSteps = 0;
    let done = false;

    while (!done) {
      const action = agent.selectAction(state);
      const { nextState, reward, done: isDone } = env.step(action);

      agent.store({ state, action, reward, nextState, done: isDone });

      const loss = agent.train();
      if (loss > 0) {
        episodeLoss += loss;
        trainSteps++;
      }

      episodeReward += reward;
      state = nextState;
      done = isDone;
    }

    agent.incrementEpisode();

    const avgLoss = trainSteps > 0 ? episodeLoss / trainSteps : 0;
    const step: TrainingStep = {
      episode: episode + 1,
      reward: episodeReward,
      loss: avgLoss,
      epsilon: agent.currentEpsilon,
      steps: trainSteps,
    };

    steps.push(step);

    if ((episode + 1) % 10 === 0) {
      const evalEnv = new CartPoleEnv();
      const evalReward = agent.evaluate(evalEnv);
      evalRewards.push(evalReward);
    }

    onProgress(step, (episode + 1) / config.episodes);

    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  const last50 = steps.slice(-50).map((s) => s.reward);
  const finalAvgReward = last50.reduce((a, b) => a + b, 0) / (last50.length || 1);

  return { steps, evalRewards, finalAvgReward };
}
