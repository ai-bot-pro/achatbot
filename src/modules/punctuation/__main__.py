from src.modules.punctuation import PuncEnvInit
from src.common.session import Session, SessionCtx

engine = PuncEnvInit.initEngine()
session = Session(**SessionCtx("test_client_id", 16000, 2).__dict__)
# session.ctx.state["punc_cache"] = {}
session.ctx.state.update({"punc_cache": {}})
inputs = "你好|你叫什么名字|讲一个故事|境河流是养育沿岸|人民的生命之源长期以来为帮助下游地区防灾减灾中方技术人员|在上游地区极为恶劣的自然条件下克服巨大困难甚至冒着生命危险|向印方提供汛期水文资料处理紧急事件中方重视印方在跨境河流问题上的关切|愿意进一步完善双方联合工作机制|凡是|中方能做的我们|都会去做而且会做得更好我请印度朋友们放心中国在上游的|任何开发利用都会经过科学|规划和论证兼顾上下游的利益"
inputs = "你好|你叫什么名字|讲一个故事|"
vads = inputs.split("|")
rec_result_all = "outputs: "
for text in vads:
    if not text:
        continue
    session.ctx.state["text"] = text
    punc_text = engine.generate(session)
    print(session.ctx.state.get("punc_cache"), punc_text)
    rec_result_all += punc_text

print(rec_result_all)

"""
# pytorch
python -m src.modules.punctuation
PUNC_TAG=punc_ct_tranformer_offline python -m src.modules.punctuation

# onnx
PUNC_TAG=punc_ct_tranformer_onnx python -m src.modules.punctuation
PUNC_TAG=punc_ct_tranformer_onnx_offline python -m src.modules.punctuation

"""
