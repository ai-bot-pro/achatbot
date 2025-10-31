## Project Structure

<img width="742" height="626" alt="image" src="https://github.com/user-attachments/assets/8ec37a1e-0251-4ce1-8721-55de25a84733" />

achatbot 的结构主要分为5个各部分，其中一个components是公用组件(包括一些多模态音频图片处理工具，配置，日志等组件)，其他4个分层设计，而且通过接口隔离进行解耦这4个分别是：

- cmd 命令程序启动不同的任务场景的chatbot，结合bot config 进行命令程序的启动，包括：
  - 服务交互的API, 通过http请求或者websocket消息来启对应任务bot, bot通过websocket/webrtc协议来通信交互多模态信息(文本、语音、图片)； 
  - 通过pipeline processor 使用不同传输协议组装而成的任务bot, pipeline的组装定义，以及processor中的模型引擎都可以有config来控制；
- processors 主要是处理不同类型frame: StartFrame 初始启动，调用modules层和core/llm层对多模态frame进行处理(主要是 TextFrame/AudioRawFrame/ImageRawFrame 这类多模态frame)；也会调用三方开源/闭源多模态模型API服务；
- modules 主要分为4个模块 多模态处理模块(文本，音频，图片) 以及 工具函数模块 （用于agent 启动时注册工具， 运行时 llm调用，执行工具）
- core/llm 核心模块，主要是对接开源多模态模型，比如端到端audio模型，全模态omni模型；都是自回归(AR)模型；并使用开源的推理引擎进行推理加速

### components
<img width="1305" height="781" alt="image" src="https://github.com/user-attachments/assets/9119f87c-8079-47e4-87d3-029b3298a5a2" />


### cmd (chat bot factory)
<img width="1238" height="753" alt="image" src="https://github.com/user-attachments/assets/ee6dc23a-a65f-400e-85e0-2dc02eb01d99" />


### processors
<img width="874" height="569" alt="image" src="https://github.com/user-attachments/assets/9c7def05-c5b1-486e-8964-6467c03620de" />


### modules
<img width="1105" height="857" alt="image" src="https://github.com/user-attachments/assets/3202a86e-9457-44eb-8a88-03f373256098" />


### core/llm
<img width="1164" height="342" alt="image" src="https://github.com/user-attachments/assets/ad0f4be9-85ef-48b3-929e-cca6085cf375" />
