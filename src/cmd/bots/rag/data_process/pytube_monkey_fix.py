"""
this is pytube monkey fix, just adapter youtube js change
[!TIPS]: u can chat with chatGPT to fix it, use js code and bug info
"""

import logging
import re
from typing import List

from pytube import YouTube, cipher
from pytube.innertube import _default_clients
from pytube.exceptions import RegexMatchError

YouTube = YouTube

# issue: https://github.com/pytube/pytube/issues/1973
_default_clients["ANDROID"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["ANDROID_EMBED"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS_EMBED"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS_MUSIC"]["context"]["client"]["clientVersion"] = "6.41"
_default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID_CREATOR"]


def get_throttling_function_name(js: str) -> str:
    """Extract the name of the function that computes the throttling parameter.

    :param str js:
        The contents of the base.js asset file.
    :rtype: str
    :returns:
        The name of the function used to compute the throttling parameter.
    """
    function_patterns = [
        r'a\.[a-zA-Z]\s*&&\s*\([a-z]\s*=\s*a\.get\("n"\)\)\s*&&.*?\|\|\s*([a-z]+)',
        r'a\.[a-zA-Z]\s*&&\s*\([a-z]\s*=\s*a\.get\("n"\)\)\s*&&\s*'
        r"\([a-z]\s*=\s*([a-zA-Z0-9$]+)(\[\d+\])?\([a-z]\)",
        r"\([a-z]\s*=\s*([a-zA-Z0-9$]+)(\[\d+\])\([a-z]\)",
    ]
    # logging.debug('Finding throttling function name')
    for pattern in function_patterns:
        regex = re.compile(pattern)
        function_match = regex.search(js)
        if function_match:
            logging.debug("finished regex search, matched: %s", pattern)
            if len(function_match.groups()) == 1:
                return function_match.group(1)
            idx = function_match.group(2)
            if idx:
                idx = idx.strip("[]")
                array = re.search(
                    r"var {nfunc}\s*=\s*(\[.+?\]);".format(
                        nfunc=re.escape(function_match.group(1))
                    ),
                    js,
                )
                if array:
                    array = array.group(1).strip("[]").split(",")
                    array = [x.strip() for x in array]
                    return array[int(idx)]

    raise RegexMatchError(caller="get_throttling_function_name", pattern="multiple")


cipher.get_throttling_function_name = get_throttling_function_name


# https://github.com/pytube/pytube/issues/2006
def get_transform_plan(js: str) -> List[str]:
    """Extract the "transform plan".

    The "transform plan" is the functions that the ciphered signature is
    cycled through to obtain the actual signature.

    :param str js:
        The contents of the base.js asset file.

    **Example**:

    ['DE.AJ(a,15)',
    'DE.VR(a,3)',
    'DE.AJ(a,51)',
    'DE.VR(a,3)',
    'DE.kT(a,51)',
    'DE.kT(a,8)',
    'DE.VR(a,3)',
    'DE.kT(a,21)']
    """
    name = re.escape(cipher.get_initial_function_name(js))
    # pattern = r"%s=function\(\w\){[a-z=\.\(\"\)]*;(.*);(?:.+)}" % name
    pattern = (
        r"%s=function\(\w\){[a-z=\.\(\"\)]*;((\w+\.\w+\([\w\"\'\[\]\(\)\.\,\s]*\);)+)(?:.+)}" % name
    )
    logging.debug("getting transform plan")
    return cipher.regex_search(pattern, js, group=1).split(";")


cipher.get_transform_plan = get_transform_plan
