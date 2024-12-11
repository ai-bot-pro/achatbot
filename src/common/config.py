import logging
import os
import inspect
import asyncio


from .types import CONFIG_DIR


class Conf:
    @staticmethod
    async def save_to_yaml(args, file_path: str):
        from omegaconf import OmegaConf

        conf = OmegaConf.create(args)
        logging.debug(f"save config:{conf} to {file_path}")
        yaml_str = OmegaConf.to_yaml(conf)
        with open(file_path, "w") as f:
            f.write(yaml_str)

    @staticmethod
    async def save_obj_to_yaml(name, obj, tag=None):
        if "init" not in name and "Engine" not in name:
            return
        logging.info(f"name:{name}, obj:{obj} save to yaml")
        engine = obj()
        if engine is None:
            return
        if tag and tag not in engine.TAG:
            return
        # u can use local, dev, test, stress, gray, online
        env = os.getenv("CONF_ENV", "env")
        file_dir = os.path.join(CONFIG_DIR, env)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        file_path = os.path.join(file_dir, f"{engine.SELECTED_TAG}.yaml")
        logging.debug(f"obj: {obj}, engine: {engine}, file_path: {file_path}")
        await Conf.save_to_yaml(engine.args.__dict__, file_path)
        return name[4 : len(name) - 6].lower(), file_path, engine.SELECTED_TAG

    @staticmethod
    async def save_to_yamls(object, tag=None):
        async_tasks = []
        for name, obj in inspect.getmembers(object, inspect.isfunction):
            async_task = asyncio.create_task(Conf.save_obj_to_yaml(name, obj, tag))
            async_tasks.append(async_task)
            # await async_task
        logging.debug(f"{async_tasks}, len(async_tasks):{len(async_tasks)}")
        res = await asyncio.gather(*async_tasks, return_exceptions=False)
        manifests = {}
        items = []
        for item in res:
            if item is None:
                continue

            print("item------------>", type(item), item)
            items.append(item)
            name, file_path, _tag = item
            manifests[name] = {"file_path": file_path, "tag": _tag}

        env = os.getenv("CONF_ENV", "local")
        file_path = os.path.join(CONFIG_DIR, env, "manifests.yaml")
        await Conf.save_to_yaml(manifests, file_path)
        items.append(("manifests", file_path, "manifests"))

        return items

    @staticmethod
    async def load_from_yaml(file_path: str):
        from omegaconf import OmegaConf

        conf = OmegaConf.load(file_path)
        return conf
