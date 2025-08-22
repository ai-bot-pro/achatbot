from src.modules.text_processing import TextProcessingEnvInit
from src.common.session import Session, SessionCtx

session = Session(**SessionCtx("test_client_id", 16000, 2).__dict__)

zh_tn_text = "你好 WeTextProcessing 1.0，船新版本儿，船新体验儿，简直666，9和10"
engine = TextProcessingEnvInit.initEngine(
    tag="we_text_processing",
    language="zh",
    normalize_type="tn",
)
session.ctx.state["text"] = zh_tn_text
zh_tn_res = engine.normalize(session)
print(f"{zh_tn_text} --> {zh_tn_res=}")


zh_itn_text = "你好 WeTextProcessing 一点零，船新版本儿，船新体验儿，简直六六六，九和六"
engine = TextProcessingEnvInit.initEngine(
    tag="we_text_processing",
    language="zh",
    normalize_type="itn",
)
session.ctx.state["text"] = zh_itn_text
zh_itn_res = engine.normalize(session)
print(f"{zh_itn_text=} --> {zh_itn_res=}")


en_tn_text = "Hello WeTextProcessing 1.0, life is short, just use wetext, 666, 9 and 10"
engine = TextProcessingEnvInit.initEngine(
    tag="we_text_processing",
    language="en",
    normalize_type="tn",
)
session.ctx.state["text"] = en_tn_text
en_tn_res = engine.normalize(session)
print(f"{en_tn_text} --> {en_tn_res=}")
