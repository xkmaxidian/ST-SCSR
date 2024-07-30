import logging
import os.path
from logging.handlers import RotatingFileHandler

logger = logging.getLogger('my_logger')

def initlog(log_path :str):
    # 创建文件处理器
    logger.setLevel(level=logging.INFO)


    file_handler = RotatingFileHandler(os.path.join(log_path, 'log.log'), maxBytes=10 * 1024 * 1024, backupCount=3)

    # 创建控制台处理器
    stream_handler = logging.StreamHandler()

    # 配置处理器的格式
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s:%(lineno)d  :  %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.debug("log init success")


