from threading import Lock

_COUNTS = {}
_COUNTS_MUTEX = Lock()

_ID = 0
_ID_MUTEX = Lock()


def obj_id() -> int:
    global _ID, _ID_MUTEX
    with _ID_MUTEX:
        _ID += 1
        return _ID


def obj_count(obj) -> int:
    global _COUNTS, COUNTS_MUTEX
    name = obj.__class__.__name__
    with _COUNTS_MUTEX:
        if name not in _COUNTS:
            _COUNTS[name] = 0
        else:
            _COUNTS[name] += 1
        return _COUNTS[name]
