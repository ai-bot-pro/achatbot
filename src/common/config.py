import logging


class Conf:
    @staticmethod
    async def save_to_yaml(args, file_path: str):
        from omegaconf import OmegaConf
        conf = OmegaConf.create(args)
        logging.debug(f"save config:{conf} to {file_path}")
        yaml_str = OmegaConf.to_yaml(conf)
        with open(file_path, 'w') as f:
            f.write(yaml_str)

    @staticmethod
    async def load_from_yaml(file_path: str):
        from omegaconf import OmegaConf
        conf = OmegaConf.load(file_path)
        return conf
