import unittest
import queue
import threading
import time
import asyncio
from src.common.utils.task import fetch_async_items

r"""
python -m unittest test.common.test_task
"""


class TestFetchAsyncItems(unittest.TestCase):
    async def mock_async_generator(self, items, delay=0.1, raise_exception=False):
        """模拟异步生成器函数"""
        for i, item in enumerate(items):
            await asyncio.sleep(delay)  # 模拟异步操作
            if raise_exception and i == len(items) // 2:
                raise ValueError("测试异常")
            yield item

    def test_normal_case(self):
        """测试正常情况"""
        test_items = ["item1", "item2", "item3", "item4", "item5"]
        q = queue.Queue()

        # 在线程中运行 fetch_async_items
        thread = threading.Thread(
            target=fetch_async_items, args=(q, self.mock_async_generator, test_items)
        )
        thread.start()

        # 从队列中获取项目
        received_items = []
        while True:
            item = q.get()
            if item is None:  # 默认结束标记是 None
                break
            received_items.append(item)

        thread.join()

        # 验证结果
        self.assertEqual(received_items, test_items)

    def test_custom_end_marker(self):
        """测试自定义结束标记"""
        test_items = ["item1", "item2", "item3"]
        q = queue.Queue()
        end_marker = "END"

        # 在线程中运行 fetch_async_items，使用自定义结束标记
        thread = threading.Thread(
            target=fetch_async_items,
            args=(q, self.mock_async_generator, test_items),
            kwargs={"end": end_marker},
        )
        thread.start()

        # 从队列中获取项目
        received_items = []
        while True:
            item = q.get()
            if item == end_marker:  # 使用自定义结束标记
                break
            received_items.append(item)

        thread.join()

        # 验证结果
        self.assertEqual(received_items, test_items)

    def test_exception_case(self):
        """测试异常情况"""
        test_items = ["item1", "item2", "item3", "item4", "item5"]
        q = queue.Queue()

        # 在线程中运行 fetch_async_items，设置 raise_exception=True
        thread = threading.Thread(
            target=fetch_async_items,
            args=(q, self.mock_async_generator, test_items),
            kwargs={"raise_exception": True},
        )
        thread.start()

        # 从队列中获取项目
        received_items = []
        while True:
            item = q.get()
            if item is None:  # 异常情况下也会发送结束标记
                break
            received_items.append(item)

        thread.join()

        # 验证结果 - 应该只收到一半的项目
        self.assertEqual(len(received_items), len(test_items) // 2)
        self.assertEqual(received_items, test_items[: len(test_items) // 2])


if __name__ == "__main__":
    unittest.main()
