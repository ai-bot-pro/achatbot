import * as GaussianSplats3D from "gaussian-splat-renderer-for-lam"

export class GaussianAvatar {
  private _currentAnimationData: any = null;
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
        //backgroundColor: "0x000000"
        backgroundColor: "0xffffff"
      },
    );
    this.startTime = performance.now() / 1000;
    /*
    setTimeout(() => {
      this.curState = "Listening"
    }, 5000);
    setTimeout(() => {
      this.curState = "Thinking"
    }, 6000);
    setTimeout(() => {
      this.curState = "Responding"
    }, 10000);
    */
  }

  public updateAvatarStatus(status: string) {
    if (status) {
      this.curState = status;
    }
  }

  expressitionData: any;
  startTime = 0;
  private _lastAnimationTime = 0;

  public getChatState() {
    return this.curState;
  }

  public updateAnimationData(animationJson: any) {
    this._currentAnimationData = animationJson;
    this._lastAnimationTime = performance.now() / 1000;
  }

  public getArkitFaceFrame() {
    if (!this._currentAnimationData) {
      return {};
    }

    const length = this._currentAnimationData["frames"].length;
    const frameInfoInternal = 1.0 / 30.0;
    const currentTime = performance.now() / 1000;
    const calcDelta = (currentTime - this._lastAnimationTime) % (length * frameInfoInternal);
    const frameIndex = Math.floor(calcDelta / frameInfoInternal);

    if (frameIndex >= length) {
      return {};
    }

    this.expressitionData = {};
    this._currentAnimationData["names"].forEach((name: string, index: number) => {
      this.expressitionData[name] = this._currentAnimationData["frames"][frameIndex]["weights"][index];
    });

    return this.expressitionData;
  }
}