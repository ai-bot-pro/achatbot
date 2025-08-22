from itn.chinese.inverse_normalizer import InverseNormalizer
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer

# When the parameters differ from the defaults, it is mandatory to re-compose. To re-compose, please ensure you specify `overwrite_cache=True`.

zh_tn_text = "你好 WeTextProcessing 1.0，船新版本儿，船新体验儿，简直666，9和10"
zh_itn_text = "你好 WeTextProcessing 一点零，船新版本儿，船新体验儿，简直六六六，九和六"
en_tn_text = "Hello WeTextProcessing 1.0, life is short, just use wetext, 666, 9 and 10"
zh_tn_model = ZhNormalizer(remove_erhua=True, overwrite_cache=True)
zh_itn_model = InverseNormalizer(enable_0_to_9=False, overwrite_cache=True)
en_tn_model = EnNormalizer(overwrite_cache=True)
print(
    "中文 TN (去除儿化音，重新在线构图):\n\t{} => {}".format(
        zh_tn_text, zh_tn_model.normalize(zh_tn_text)
    )
)
print(
    "中文ITN (小于10的单独数字不转换，重新在线构图):\n\t{} => {}".format(
        zh_itn_text, zh_itn_model.normalize(zh_itn_text)
    )
)
print(
    "英文 TN (暂时还没有可控的选项，后面会加...):\n\t{} => {}\n".format(
        en_tn_text, en_tn_model.normalize(en_tn_text)
    )
)

zh_tn_model = ZhNormalizer(overwrite_cache=False)
zh_itn_model = InverseNormalizer(overwrite_cache=False)
en_tn_model = EnNormalizer(overwrite_cache=False)
print(
    "中文 TN (复用之前编译好的图):\n\t{} => {}".format(
        zh_tn_text, zh_tn_model.normalize(zh_tn_text)
    )
)
print(
    "中文ITN (复用之前编译好的图):\n\t{} => {}".format(
        zh_itn_text, zh_itn_model.normalize(zh_itn_text)
    )
)
print(
    "英文 TN (复用之前编译好的图):\n\t{} => {}\n".format(
        en_tn_text, en_tn_model.normalize(en_tn_text)
    )
)

zh_tn_model = ZhNormalizer(remove_erhua=False, overwrite_cache=True)
zh_itn_model = InverseNormalizer(enable_0_to_9=True, overwrite_cache=True)
print(
    "中文 TN (不去除儿化音，重新在线构图):\n\t{} => {}".format(
        zh_tn_text, zh_tn_model.normalize(zh_tn_text)
    )
)
print(
    "中文ITN (小于10的单独数字也进行转换，重新在线构图):\n\t{} => {}\n".format(
        zh_itn_text, zh_itn_model.normalize(zh_itn_text)
    )
)
