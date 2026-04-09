import { TrainingStep } from "./agent";

export function movingAverage(data: number[], window: number): number[] {
  return data.map((_, i) => {
    const start = Math.max(0, i - window + 1);
    const slice = data.slice(start, i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

export function computeMetrics(steps: TrainingStep[]) {
  const rewards = steps.map((s) => s.reward);
  const losses = steps.map((s) => s.loss);

  const maxReward = Math.max(...rewards);
  const avgReward = rewards.reduce((a, b) => a + b, 0) / rewards.length;
  const last50Avg = rewards.slice(-50).reduce((a, b) => a + b, 0) / Math.min(50, rewards.length);
  const solvedEpisode = rewards.findIndex((_, i, arr) => {
    if (i < 99) return false;
    const avg = arr.slice(i - 99, i + 1).reduce((a, b) => a + b, 0) / 100;
    return avg >= 195;
  });

  const avgLoss = losses.filter((l) => l > 0).reduce((a, b) => a + b, 0) / (losses.filter((l) => l > 0).length || 1);

  return {
    maxReward,
    avgReward: Math.round(avgReward * 10) / 10,
    last50Avg: Math.round(last50Avg * 10) / 10,
    solvedEpisode: solvedEpisode >= 0 ? solvedEpisode + 1 : null,
    avgLoss: Math.round(avgLoss * 10000) / 10000,
  };
}

export function formatNumber(n: number, decimals = 2): string {
  return n.toFixed(decimals);
}
