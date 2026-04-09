export interface Transition {
  state: number[];
  action: number;
  reward: number;
  nextState: number[];
  done: boolean;
}

export class ReplayBuffer {
  private buffer: Transition[] = [];
  private readonly capacity: number;

  constructor(capacity: number) {
    this.capacity = capacity;
  }

  add(transition: Transition): void {
    if (this.buffer.length >= this.capacity) {
      this.buffer.shift();
    }
    this.buffer.push(transition);
  }

  sample(batchSize: number): Transition[] {
    const shuffled = [...this.buffer].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, batchSize);
  }

  get size(): number {
    return this.buffer.length;
  }
}
