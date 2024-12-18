import concurrent.futures


class ClassObject:
    def __init__(self, cls, **kwargs):
        self._count = 0
        self.obj = cls(**kwargs)


class ClassObjectPool:
    def __init__(self, size, cls, multi_thread_init: bool = False, **kwargs):
        """
        Base obj pool
        Initialize the ClassObjectPool with a specified size and configs.

        Parameters:
        - size (int): The number of objects to initialize in the pool.
        - cls: the obj class
        - multi_thread_init: use multi thread to init class instance obj.
        - kwargs: The args of object.

        Returns:
        - None
        """
        self.cls = cls
        self.pool = self._initialize_pool(size, cls, multi_thread_init, **kwargs)

    def _initialize_pool(self, size, cls, multi_thread_init: bool = False, **kwargs):
        if not multi_thread_init:
            pool = [ClassObject(cls, **kwargs) for _ in range(size)]
            return pool
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(ClassObject, cls, **kwargs) for _ in range(size)]
                return [future.result() for future in concurrent.futures.as_completed(futures)]

    def acquire(self):
        # Find the object with the minimum count
        min_cn_obj = min(self.pool, key=lambda obj: obj._count)
        min_cn_obj._count += 1
        return min_cn_obj

    def release(self, obj):
        if obj._count > 0:
            obj._count -= 1

    def print_info(self):
        for i, obj in enumerate(self.pool):
            print(f"ClassObject {self.cls} {i} use count: {obj._count}")


class OneClassObjectPool(ClassObjectPool):
    """
    using obj don't reuse
    """

    def acquire(self):
        for obj in self.pool:
            if obj._count == 0:
                obj._count = 1
                return obj
        raise Exception("No available objects in the pool")
