import * as GaussianSplats3D from "gaussian-splat-renderer-for-lam"

export class GaussianAvatar {
  private _currentAnimationData: any = null;
  private _avatarDivEle: HTMLDivElement;
  private _assetsPath = "";
  public curState = "Idle";
  private _renderer!: GaussianSplats3D.GaussianSplatRenderer;
  // 添加一个标志，表示是否有音频正在播放
  private _isAudioPlaying: boolean = false;
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

  public updateAnimationData(animationJson: any, audioContextTime?: number) {
    this._currentAnimationData = animationJson;
    // 如果提供了音频上下文时间，则使用它作为动画的时间基准
    // 否则使用性能时间
    this._lastAnimationTime = audioContextTime || performance.now() / 1000;
  }

  // 添加一个属性来存储音频上下文
  private _audioContext: AudioContext | null = null;

  // 设置音频上下文的方法
  public setAudioContext(audioContext: AudioContext) {
    this._audioContext = audioContext;
  }

  // 设置音频播放状态的方法
  public setAudioPlayingState(isPlaying: boolean) {
    this._isAudioPlaying = isPlaying;
    if (!isPlaying) {
      // 如果音频停止播放，重置动画数据
      this._currentAnimationData = null;
    }
  }

  public getArkitFaceFrame() {
    // 如果没有音频播放，或者没有动画数据，返回空对象
    if (!this._isAudioPlaying || !this._currentAnimationData) {
      return {};
    }

    if (!this._currentAnimationData["frames"]) {
      return {};
    }

    const length = this._currentAnimationData["frames"].length;
    const frameInfoInternal = 1.0 / this._currentAnimationData["metadata"]["fps"];

    // 使用音频上下文的时间（如果可用）或性能时间
    let currentTime = this._audioContext ? this._audioContext.currentTime : performance.now() / 1000;
    currentTime = currentTime > this._lastAnimationTime ? currentTime : this._lastAnimationTime

    // 计算从动画开始到现在经过的时间，并确定当前应该显示哪一帧
    const calcDelta = (currentTime - this._lastAnimationTime) % (length * frameInfoInternal);
    const frameIndex = Math.floor(calcDelta / frameInfoInternal);

    // 添加更详细的日志以便调试
    if (frameIndex % 10 === 0) { // 每10帧记录一次，避免日志过多
      console.log(`Frame: ${frameIndex}/${length}, Time: ${currentTime.toFixed(3)}, AnimStart: ${this._lastAnimationTime.toFixed(3)}, Delta: ${calcDelta.toFixed(3)}`);
    }

    // 确保frameIndex在有效范围内
    if (frameIndex >= length || frameIndex < 0) {
      console.warn(`Invalid frameIndex: ${frameIndex}, length: ${length}`);
      return {};
    }

    // 检查当前帧是否存在
    const currentFrame = this._currentAnimationData["frames"][frameIndex];
    if (!currentFrame || !currentFrame.weights) {
      console.warn(`Frame ${currentFrame} at index ${frameIndex} is invalid or missing weights`);
      return {};
    }

    this.expressitionData = {};
    this._currentAnimationData["names"].forEach((name: string, index: number) => {
      // 确保index在weights数组范围内
      if (index < currentFrame.weights.length) {
        this.expressitionData[name] = currentFrame.weights[index];
      } else {
        console.warn(`Weight index ${index} out of bounds for frame ${frameIndex}`);
        this.expressitionData[name] = 0; // 使用默认值
      }
    });

    return this.expressitionData;
  }
}