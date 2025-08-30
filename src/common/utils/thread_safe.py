import threading
from typing import Dict, Any, List, Tuple


class ThreadSafeDict(Dict):
    """线程安全的字典实现

    使用 RLock 确保所有操作的线程安全性
    支持标准字典操作和额外的便利方法
    """

    def __init__(self):
        self._map = {}  # 内部存储
        self._lock = threading.RLock()  # 可重入锁

    def __setitem__(self, key: Any, value: Any) -> None:
        """支持 dict[key] = value 语法"""
        self.set(key, value)

    def __getitem__(self, key: Any) -> Any:
        """支持 dict[key] 语法"""
        return self.get(key)

    def __delitem__(self, key: Any) -> None:
        """支持 del dict[key] 语法"""
        self.delete(key)

    def __contains__(self, key: Any) -> bool:
        """支持 key in dict 语法"""
        return self.contains(key)

    def __len__(self) -> int:
        """支持 len(dict) 语法"""
        return self.size()

    def __repr__(self) -> str:
        """字符串表示"""
        if hasattr(self, "_lock") is False:
            return repr(self._map)
        with self._lock:
            return repr(self._map)

    def set(self, key: Any, value: Any) -> None:
        """设置键值对"""
        if hasattr(self, "_lock") is False:  # init
            if hasattr(self, "_map") is False:
                self._map = {}
            self._map[key] = value
            return
        with self._lock:
            self._map[key] = value

    def get(self, key: Any, default: Any = None) -> Any:
        """获取值，支持默认值"""
        if hasattr(self, "_lock") is False:
            return self._map.get(key, default)
        with self._lock:
            return self._map.get(key, default)

    def pop(self, key: Any, default: Any = None) -> Any:
        """弹出并返回值"""
        if hasattr(self, "_lock") is False:
            return self._map.pop(key, default)
        with self._lock:
            return self._map.pop(key, default)

    def delete(self, key: Any) -> None:
        """删除键值对"""
        if hasattr(self, "_lock") is False:
            if key in self._map:
                del self._map[key]
            return
        with self._lock:
            if key in self._map:
                del self._map[key]

    def contains(self, key: Any) -> bool:
        """检查键是否存在"""
        if hasattr(self, "_lock") is False:
            return key in self._map
        with self._lock:
            return key in self._map

    def clear(self) -> None:
        """清空字典"""
        if hasattr(self, "_lock") is False:
            self._map.clear()
            return
        with self._lock:
            self._map.clear()

    def size(self) -> int:
        """返回字典大小"""
        if hasattr(self, "_lock") is False:
            return len(self._map)
        with self._lock:
            return len(self._map)

    def keys(self) -> List[Any]:
        """返回所有键的列表"""
        if hasattr(self, "_lock") is False:
            return list(self._map.keys())
        with self._lock:
            return list(self._map.keys())

    def values(self) -> List[Any]:
        """返回所有值的列表"""
        if hasattr(self, "_lock") is False:
            return list(self._map.values())
        with self._lock:
            return list(self._map.values())

    def items(self) -> List[Tuple[Any, Any]]:
        """返回所有键值对的列表"""
        if hasattr(self, "_lock") is False:
            return list(self._map.items())
        with self._lock:
            return list(self._map.items())

    def update(self, kwargs: Dict) -> None:
        """更新字典"""
        if hasattr(self, "_lock") is False:
            self._map.update(kwargs)
            return
        with self._lock:
            self._map.update(kwargs)

    def __getstate__(self):
        """序列化时排除不可pickle的属性"""
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state):
        """反序列化时重建对象状态"""
        self._lock = threading.RLock()
        self.__dict__.update(state)


"""
python -m src.common.utils.thread_safe
"""
if __name__ == "__main__":

    def test_set_get_item(test_dict=None):
        test_dict = test_dict or ThreadSafeDict()
        test_dict["a"] = "123"
        test_dict.update({"a": "321", "b": "123"})
        print(test_dict)

    def test_concurency(safe_dict=None):
        import time
        from concurrent.futures import ThreadPoolExecutor

        def write_task(thread_dict, thread_id):
            """写入任务"""
            for i in range(100):
                key = f"thread_{thread_id}_{i}"
                thread_dict.set(key, f"value_{thread_id}_{i}")
                time.sleep(0.01)  # 模拟耗时操作

        def read_task(thread_dict, thread_id):
            """读取任务"""
            count = 0
            for _ in range(50):
                keys = thread_dict.keys()
                for key in keys:
                    value = thread_dict.get(key)
                    if value:
                        count += 1
                time.sleep(0.02)  # 模拟耗时操作
            print(f"线程 {thread_id} 读取到 {count} 个值")

        # 创建线程安全字典
        safe_dict = safe_dict or ThreadSafeDict()

        # 创建线程池
        with ThreadPoolExecutor(max_workers=10) as executor:
            # 启动5个写线程
            write_futures = [executor.submit(write_task, safe_dict, i) for i in range(5)]

            # 启动3个读线程
            read_futures = [executor.submit(read_task, safe_dict, i) for i in range(3)]

            # 等待所有任务完成
            for future in write_futures + read_futures:
                future.result()

        # 验证最终结果
        print(f"\n测试完成:")
        print(f"字典最终大小: {safe_dict.size()}")
        print(f"理论写入数量: {5 * 100}")  # 5个线程各写入100个值

    def test_pickling():
        import pickle

        # 创建并填充字典
        ts_dict = ThreadSafeDict()
        ts_dict["test"] = "value"

        # 序列化
        pickled = pickle.dumps(ts_dict)

        # 反序列化
        unpickled = pickle.loads(pickled)

        # 验证状态
        assert unpickled.get("test") == "value"
        print(type(unpickled._lock), type(threading.RLock))
        # assert isinstance(unpickled._lock, threading.RLock)
        assert type(unpickled._lock) is type(threading.RLock())
        print("Pickle test passed!")

        return unpickled

    test_set_get_item()
    test_concurency()
    safe_dict = test_pickling()
    test_set_get_item(safe_dict)
    test_concurency(safe_dict)
