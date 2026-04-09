export interface CartPoleState {
  x: number;
  xDot: number;
  theta: number;
  thetaDot: number;
}

export interface StepResult {
  nextState: number[];
  reward: number;
  done: boolean;
}

export class CartPoleEnv {
  private state: CartPoleState;
  private readonly gravity = 9.8;
  private readonly massCart = 1.0;
  private readonly massPole = 0.1;
  private readonly totalMass = 1.1;
  private readonly length = 0.5;
  private readonly poleMassLength = 0.05;
  private readonly forceMag = 10.0;
  private readonly tau = 0.02;
  private readonly xThreshold = 2.4;
  private readonly thetaThreshold = 12 * (Math.PI / 180);
  private stepCount = 0;
  readonly stateSize = 4;
  readonly actionSize = 2;

  constructor() {
    this.state = this.randomState();
  }

  private randomState(): CartPoleState {
    return {
      x: (Math.random() - 0.5) * 0.1,
      xDot: (Math.random() - 0.5) * 0.1,
      theta: (Math.random() - 0.5) * 0.1,
      thetaDot: (Math.random() - 0.5) * 0.1,
    };
  }

  reset(): number[] {
    this.state = this.randomState();
    this.stepCount = 0;
    return this.getObs();
  }

  private getObs(): number[] {
    return [this.state.x, this.state.xDot, this.state.theta, this.state.thetaDot];
  }

  step(action: number): StepResult {
    const force = action === 1 ? this.forceMag : -this.forceMag;
    const { x, xDot, theta, thetaDot } = this.state;

    const cosTheta = Math.cos(theta);
    const sinTheta = Math.sin(theta);

    const temp = (force + this.poleMassLength * thetaDot * thetaDot * sinTheta) / this.totalMass;
    const thetaAcc =
      (this.gravity * sinTheta - cosTheta * temp) /
      (this.length * (4.0 / 3.0 - (this.massPole * cosTheta * cosTheta) / this.totalMass));
    const xAcc = temp - (this.poleMassLength * thetaAcc * cosTheta) / this.totalMass;

    const newX = x + this.tau * xDot;
    const newXDot = xDot + this.tau * xAcc;
    const newTheta = theta + this.tau * thetaDot;
    const newThetaDot = thetaDot + this.tau * thetaAcc;

    this.state = { x: newX, xDot: newXDot, theta: newTheta, thetaDot: newThetaDot };
    this.stepCount++;

    const done =
      Math.abs(newX) > this.xThreshold ||
      Math.abs(newTheta) > this.thetaThreshold ||
      this.stepCount >= 500;

    return {
      nextState: this.getObs(),
      reward: done && this.stepCount < 500 ? 0 : 1,
      done,
    };
  }
}
