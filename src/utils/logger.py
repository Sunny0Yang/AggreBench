import logging
import logging.config
import os
import yaml

def setup_logging():
    """配置日志系统"""
    log_config = os.getenv('LOG_CONFIG_PATH', 
                          f"{os.getenv('CONFIG_DIR')}/logging.yaml")
    
    # 基础配置
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_file = f"{os.getenv('LOG_DIR','logs')}/{os.getenv('LOG_FILE', 'pipeline.log')}"
    
    # 加载YAML配置或使用默认
    try:
        with open(log_config) as f:
            config = yaml.safe_load(f)
            config['handlers']['file']['filename'] = log_file
            config['root']['level'] = log_level
            logging.config.dictConfig(config)
    except FileNotFoundError:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    return logging.getLogger(__name__)