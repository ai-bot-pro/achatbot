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
