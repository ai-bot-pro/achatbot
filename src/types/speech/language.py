#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import sys

from enum import Enum

if sys.version_info < (3, 11):

    class StrEnum(str, Enum):
        def __new__(cls, value):
            obj = str.__new__(cls, value)
            obj._value_ = value
            return obj
else:
    from enum import StrEnum


class Language(StrEnum):
    BG = "bg"  # Bulgarian
    CA = "ca"  # Catalan
    ZH = "zh"  # Chinese simplified
    ZH_CN = "zh-CN"  # Chinese simplified
    ZH_TW = "zh-TW"  # Chinese traditional
    ZN = "zn"  # Chinese and English
    CS = "cs"  # Czech
    DA = "da"  # Danish
    NL = "nl"  # Dutch
    EN = "en"  # English
    EN_US = "en-US"  # English (USA)
    EN_AU = "en-AU"  # English (Australia)
    EN_GB = "en-GB"  # English (Great Britain)
    EN_NZ = "en-NZ"  # English (New Zealand)
    EN_IN = "en-IN"  # English (India)
    ET = "et"  # Estonian
    FI = "fi"  # Finnish
    NL_BE = "nl-BE"  # Flemmish
    FR = "fr"  # French
    FR_CA = "fr-CA"  # French (Canada)
    DE = "de"  # German
    DE_CH = "de-CH"  # German (Switzerland)
    EL = "el"  # Greek
    HI = "hi"  # Hindi
    HU = "hu"  # Hungarian
    ID = "id"  # Indonesian
    IT = "it"  # Italian
    JA = "ja"  # Japanese
    KO = "ko"  # Korean
    LV = "lv"  # Latvian
    LT = "lt"  # Lithuanian
    MS = "ms"  # Malay
    NO = "no"  # Norwegian
    PL = "pl"  # Polish
    PT = "pt"  # Portuguese
    PT_BR = "pt-BR"  # Portuguese (Brazil)
    RO = "ro"  # Romanian
    RU = "ru"  # Russian
    SK = "sk"  # Slovak
    ES = "es"  # Spanish
    SV = "sv"  # Swedish
    TH = "th"  # Thai
    TR = "tr"  # Turkish
    UK = "uk"  # Ukrainian
    VI = "vi"  # Vietnamese


TO_LLM_LANGUAGE = {
    "bg": "Bulgarian",
    "ca": "Catalan",
    "": "Chinese simplified(简体中文)",
    "zh": "Chinese simplified(简体中文)",
    "zh-CN": "Chinese simplified(简体中文)",
    "zh-TW": "Chinese traditional(繁体中文)",
    "zn": "Chinese and English",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "en-US": "English (USA)",
    "en-AU": "English (Australia)",
    "en-GB": "English (Great Britain)",
    "en-NZ": "English (New Zealand)",
    "en-IN": "English (India)",
    "et": "Estonian",
    "fi": "Finnish",
    "nl-BE": "Flemmish",
    "fr": "French",
    "fr-CA": "French (Canada)",
    "de": "German",
    "de-CH": "German (Switzerland)",
    "el": "Greek",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "ms": "Malay",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "pt-BR": "Portuguese (Brazil)",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "es": "Spanish",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
}


WHISPER_LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}


TRANSLATE_LANGUAGE = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "nb": "Norwegian Bokmal",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
}
