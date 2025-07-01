import * as GaussianSplats3D from "gaussian-splat-renderer-for-lam"
//import bsData from "../asset/test_expression_1s.json"
import bsData from "../asset/asr_example_expression.json"

export class GaussianAvatar {
  private _avatarDivEle: HTMLDivElement;
  private _assetsPath = "";
  public curState = "Idle";
  private _renderer!: GaussianSplats3D.GaussianSplatRenderer;
  constructor(container: HTMLDivElement, assetsPath: string) {
    this._avatarDivEle = container;
    this._assetsPath = assetsPath;
    this._init();
  }
  private _init() {
    if (!this._avatarDivEle || !this._assetsPath) {
      throw new Error("Lack of necessary initialization parameters");
    }
  }

  public start() {
    this.render();
  }

  public async render() {
    this._renderer = await GaussianSplats3D.GaussianSplatRenderer.getInstance(
      this._avatarDivEle,
      this._assetsPath,
      {
        getChatState: this.getChatState.bind(this),
        getExpressionData: this.getArkitFaceFrame.bind(this),
        backgroundColor: "0x000000"
      },
    );
    this.startTime = performance.now() / 1000;
    setTimeout(() => {
      this.curState = "Listening"
    }, 5000);
    setTimeout(() => {
      this.curState = "Thinking"
    }, 6000);
    setTimeout(() => {
      this.curState = "Responding"
    }, 10000);

  }

  expressitionData: any;
  startTime = 0
  public getChatState() {
    return this.curState;
  }
  public getArkitFaceFrame() {
    const length = bsData["frames"].length

    const frameInfoInternal = 1.0 / 30.0;
    const currentTime = performance.now() / 1000;
    const calcDelta = (currentTime - this.startTime) % (length * frameInfoInternal);
    const frameIndex = Math.floor(calcDelta / frameInfoInternal)
    this.expressitionData = {};


    bsData["names"].forEach((name: string, index: number) => {
      this.expressitionData[name] = bsData["frames"][frameIndex]["weights"][index]
    })
    return this.expressitionData;
  }
}