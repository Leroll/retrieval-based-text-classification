import yaml
import os
from pydantic import BaseModel
from enum import Enum
import uuid
from loguru import logger
from pathlib import Path
import traceback
from dotenv import load_dotenv
load_dotenv()

from retrieval_based_text_classification.utils import Singleton  


class AppConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8042
    workers: int = 2  # Number of workers for uvicorn
    log_dir: str = "logs"


class ConfigData(BaseModel):
    app: AppConfig = AppConfig()


@Singleton
class Config(object):
    def __init__(self, config_path: str = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径，如果为None则从环境变量CONFIG_PATH获取，
                        如果环境变量也没有则使用默认配置
        """
        self.config_path = config_path or os.getenv("CONFIG_PATH")
        self.config = self._load_config()
        self.config_id = str(uuid.uuid4().hex)
        logger.info(f"Config ID: {self.config_id}")
        if self.config_path:
            logger.info(f"Loaded config from: {self.config_path}")
        else:
            logger.info("Using default configuration")

    def _load_config(self) -> ConfigData:
        """加载配置文件"""
        if not self.config_path:
            logger.info("No config path specified, using default values")
            return ConfigData()
            
        config_path = Path(self.config_path)
        
        try:
            if not config_path.exists():
                logger.warning(f"Config file {config_path} not found, using default values")
                return ConfigData()
                
            config_dict = self._yaml_load(config_path)
            config = ConfigData(**config_dict)
            logger.info(f"Successfully loaded config from {config_path}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {config_path}: {e}, using default values")
            return ConfigData()
        except Exception as e:
            logger.error(f"Unexpected error loading config from {config_path}, using default values:\n{traceback.format_exc()}")
            return ConfigData()

    def _yaml_load(self, path: Path):
        """加载YAML文件"""
        with open(path, "r", encoding="utf-8") as fin:
            return yaml.safe_load(fin)

    def get_config(self) -> ConfigData:
        """获取配置对象"""
        return self.config


if __name__ == "__main__":
    # 测试不同的配置加载方式
    print("=== Testing config loading ===")
    
    # 1. 默认方式（无配置文件）
    cfg1 = Config()
    print(f"Default config ID: {cfg1.config_id}")
    
    # 2. 指定配置文件
    cfg2 = Config("configs/prod.yaml")
    print(f"Specific file config ID: {cfg2.config_id}")
    
    # 3. 测试单例模式
    cfg3 = Config()
    cfg4 = Config()
    print(f"Singleton test - cfg3 ID: {cfg3.config_id}, cfg4 ID: {cfg4.config_id}")
    print(f"Are they the same instance? {cfg3 is cfg4}")


