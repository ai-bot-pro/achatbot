import os
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Type, Any, Optional
import queue

from src.common.interface import IPoolInstance
from src.common.register import Register


# 用于存储创建引擎函数的全局字典 use engine factory to create with bot config
# _new_func_map: Dict[str, callable] = {}


class PoolInstanceInfo:
    """池中对象实例的包装信息，包含使用状态等。"""

    def __init__(self, instance: IPoolInstance):
        self.instance_id = None  # 由外部设置
        self._is_in_use = False
        self.last_used = time.time_ns()
        self.instance = instance
        self._lock = threading.Lock()

    def get_instance(self) -> IPoolInstance:
        return self.instance

    def mark_in_use(self) -> bool:
        """尝试标记实例为正在使用。"""
        with self._lock:
            if not self._is_in_use:
                self._is_in_use = True
                self.last_used = time.time_ns()
                return True
            return False

    def mark_available(self) -> bool:
        """尝试标记实例为可用。"""
        with self._lock:
            if self._is_in_use:
                self._is_in_use = False
                self.last_used = time.time_ns()
                return True
            return False

    def is_in_use(self) -> bool:
        with self._lock:
            return self._is_in_use


class EngineProviderPool:
    """engine provider pool (EngineClass)"""

    def __init__(self, pool_size: int, new_func: callable, init_worker_num: int | None = None):
        if pool_size <= 0:
            pool_size = 1
        if init_worker_num and init_worker_num > pool_size:
            init_worker_num = pool_size
        self._pool_instances: queue.Queue[PoolInstanceInfo] = queue.Queue(maxsize=pool_size)
        self.pool_size = pool_size
        self.init_worker_num = init_worker_num or os.cpu_count()  # logic cpu count
        self._new_func = new_func
        self._stop_event = threading.Event()
        self._lock = threading.RLock()  # 用于保护共享计数器

        # 统计信息
        self._total_created = 0
        self._total_reused = 0
        self._total_active = 0

        if not self._new_func:
            raise ValueError(f"No NewFunc registered for {new_func=}")

    def _create_new_instance_info(self) -> PoolInstanceInfo:
        """创建新的实例信息对象。"""
        try:
            instance = self._new_func()
            if not isinstance(instance, IPoolInstance):
                raise TypeError(f"NewFunc for {self._new_func} did not return an IPoolInstance")
        except Exception as e:
            raise RuntimeError(f"Failed to create new instance for {self._new_func}: {e}")

        with self._lock:
            instance_id = self._total_created + 1
            self._total_created = instance_id

        instance_info = PoolInstanceInfo(instance)
        instance_info.instance_id = instance_id
        logging.info(f"Created New InstanceInfo# {instance_info.instance_id}")

        return instance_info

    def initialize(self) -> bool:
        """并行初始化池中的对象。"""
        logging.info(
            f"Initializing pool with {self.init_worker_num=} {self.pool_size} instances..."
        )
        success_count = 0
        init_errors = []

        # 使用 ThreadPoolExecutor 来并行创建实例
        with ThreadPoolExecutor(max_workers=self.init_worker_num) as executor:
            futures = [
                executor.submit(self._create_new_instance_info) for _ in range(self.pool_size)
            ]

            for i, future in enumerate(futures):
                try:
                    instance_info = future.result()
                    try:
                        self._pool_instances.put_nowait(instance_info)
                        logging.info(f"{self._new_func} Instance#{i} Initialized and put into pool")
                        success_count += 1
                    except queue.Full:
                        # 如果队列满了，释放实例
                        instance_info.instance.release()
                        init_errors.append(
                            RuntimeError(
                                f"Pool queue full during initialization, instance#{i} released"
                            )
                        )
                        logging.warning(
                            f"Pool queue full during initialization, instance#{i} released"
                        )
                except Exception as e:
                    init_errors.append(e)
                    logging.warning(f"Initialization warning for instance#{i}: {e}")

        logging.info(
            f"Pool initialized with {success_count}/{self.pool_size} {self._new_func} instances"
        )

        if init_errors and success_count == 0:
            logging.error(f"Failed to initialize any {self._new_func} instances")
            return False

        return True

    def get(self) -> Optional[PoolInstanceInfo]:
        """从池中获取一个实例。"""
        logging.info(
            f"Attempting to get {self._new_func} instance from pool (available: {self._pool_instances.qsize()})"
        )

        # 首先尝试从池中获取
        try:
            instance_info = self._pool_instances.get_nowait()
            if instance_info.mark_in_use():
                with self._lock:
                    self._total_reused += 1
                    self._total_active += 1
                logging.info(
                    f"Got {self._new_func} instanceInfo# {instance_info.instance_id} from pool and marked as in-use (active: {self._total_active})"
                )
                return instance_info
            else:
                # 如果获取到的实例正在被使用，理论上不应该发生，但为了安全，放回队列
                logging.warning(
                    f"Instance# {instance_info.instance_id} was marked as in-use immediately after being taken from pool. This is unexpected."
                )
                try:
                    self._pool_instances.put_nowait(instance_info)
                except queue.Full:
                    # 如果放不回去，释放它
                    logging.warning(
                        f"Could not return unexpected in-use instance# {instance_info.instance_id} to pool, releasing it."
                    )
                    instance_info.instance.release()
        except queue.Empty:
            pass  # 池中没有可用实例

        # 如果池中没有可用实例，等待一段时间看是否有实例被归还
        timeout = 0.1  # 100ms
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                instance_info = self._pool_instances.get(
                    timeout=timeout - (time.time() - start_time)
                )
                if instance_info.mark_in_use():
                    with self._lock:
                        self._total_reused += 1
                        self._total_active += 1
                    logging.info(
                        f"Got {self._new_func} instanceInfo# {instance_info.instance_id} from pool after waiting and marked as in-use (active: {self._total_active})"
                    )
                    return instance_info
                else:
                    # 实例被占用，尝试放回
                    try:
                        self._pool_instances.put_nowait(instance_info)
                    except queue.Full:
                        instance_info.instance.release()
            except queue.Empty:
                break  # 超时

        # 如果等待后仍未获取到，创建一个新实例（超出池大小）
        logging.warning("Pool timeout, creating new beyond instance")
        try:
            instance_info = self._create_new_instance_info()
            instance_info.mark_in_use()  # 新实例直接标记为使用中
            with self._lock:
                self._total_active += 1
            logging.info(
                f"Created new beyond instance# {instance_info.instance_id} and marked as in-use (active: {self._total_active})"
            )
            return instance_info
        except Exception as e:
            logging.error(f"Failed to create new beyond instance: {e}")
            return None

    def put(self, instance_info: PoolInstanceInfo) -> None:
        """将实例归还到池中。"""
        if not instance_info:
            logging.warning("Attempted to put nil instance")
            return

        if self._stop_event.is_set():
            logging.warning("Pool is shutting down, releasing instance")
            instance_info.instance.release()
            return

        logging.info(f"Returning instance# {instance_info.instance_id} to pool({self._new_func})")

        if not instance_info.mark_available():
            logging.warning(
                f"Instance# {instance_info.instance_id} was not in use, cannot return to pool. This is unexpected."
            )
            return

        # 重置实例
        try:
            instance_info.instance.reset()
        except Exception as e:
            logging.warning(
                f"Failed to reset instance# {instance_info.instance_id} to pool({self._new_func}) err: {e}"
            )

        try:
            self._pool_instances.put_nowait(instance_info)
            with self._lock:
                self._total_active -= 1
            logging.info(
                f"Instance# {instance_info.instance_id} marked as available (active: {self._total_active}) to pool({self._new_func})"
            )
        except queue.Full:
            logging.warning(f"Pool queue full, releasing instance# {instance_info.instance_id}")
            instance_info.instance.release()

    def get_stats(self) -> Dict[str, Any]:
        """获取池的统计信息。"""
        with self._lock:
            return {
                "pool_size": self.pool_size,
                "total_instances": self._pool_instances.qsize(),
                "active_count": self._total_active,
                "total_created": self._total_created,
                "total_reused": self._total_reused,
            }

    def close(self) -> None:
        """关闭池，释放所有资源。"""
        logging.info("Pool Closing...")
        self._stop_event.set()

        # 清空队列并释放所有实例
        while True:
            try:
                instance_info = self._pool_instances.get_nowait()
                if instance_info:
                    instance_info.instance.release()
            except queue.Empty:
                break

        logging.info("Pool Closed")


"""
python -m src.common.pool_module
"""
if __name__ == "__main__":
    # --- 示例用法 ---
    class ExamplePoolableResource(IPoolInstance):
        """一个示例的可池化资源类。"""

        def __init__(self, resource_id: int):
            self.resource_id = resource_id
            self.data = f"Resource-{resource_id}-Data"
            logging.info(f"Created ExamplePoolableResource #{self.resource_id}")

        def reset(self) -> None:
            logging.info(f"Resetting ExamplePoolableResource #{self.resource_id}")
            # 模拟重置操作
            self.data = f"Resource-{self.resource_id}-Data-Reset"

        def release(self) -> None:
            logging.info(f"Releasing ExamplePoolableResource #{self.resource_id}")
            # 模拟释放资源操作
            self.data = None

    # 1. 创建函数
    def create_example_resource() -> IPoolInstance:
        """创建 ExamplePoolableResource 的工厂函数。"""
        # 这里可以使用一个静态计数器来模拟 ID 分配
        if not hasattr(create_example_resource, "counter"):
            create_example_resource.counter = 0
        create_example_resource.counter += 1
        return ExamplePoolableResource(create_example_resource.counter)

    # 2. 创建并初始化池
    pool = EngineProviderPool(pool_size=3, new_func=create_example_resource)
    if not pool.initialize():
        logging.error("Failed to initialize pool")
        exit(1)

    print("\n--- Pool Stats After Initialization ---")
    print(pool.get_stats())

    # 3. 获取实例
    print("\n--- Getting Instances ---")
    instance_info1 = pool.get()
    instance_info2 = pool.get()
    instance_info3 = pool.get()
    instance_info4 = pool.get()  # 这个会超出池大小创建

    print(
        f"Got instances: {instance_info1.instance_id}, {instance_info2.instance_id}, {instance_info3.instance_id}, {instance_info4.instance_id}"
    )

    print("\n--- Pool Stats After Getting ---")
    print(pool.get_stats())

    # 4. 模拟使用实例
    print(f"\n--- Using Instance 1 ---")
    resource1 = instance_info1.get_instance()
    print(f"Resource 1 Data: {resource1.data}")

    # 5. 归还实例
    print(f"\n--- Returning Instances 1, 2, 3 ---")
    pool.put(instance_info1)
    pool.put(instance_info2)
    pool.put(instance_info3)
    # 5.1. 归还超出池大小的实例
    print(f"\n--- Returning Beyond-Pool Instance 4 ---")
    pool.put(instance_info4)  # 这个会因为队列满而被释放

    print("\n--- Pool Stats After Returning 3 ---")
    print(pool.get_stats())

    # 6. 尝试获取一个刚归还的实例
    print(f"\n--- Getting Another Instance (should reuse) ---")
    instance_info5 = pool.get()
    print(f"Got instance: {instance_info5.instance_id} (Should be one of 1, 2, or 3)")

    print("\n--- Pool Stats After Re-getting ---")
    print(pool.get_stats())

    # 6.1. 已释放的不能放入
    print(f"\n--- Returning Beyond-Pool Instance 4 ---")
    pool.put(instance_info4)

    print("\n--- Final Pool Stats ---")
    print(pool.get_stats())

    # 7. 关闭池
    pool.close()
