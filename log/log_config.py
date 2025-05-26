from loguru import logger


def create_log():
    # logger.remove(0)
    logger.add("../log/log_file/log_file_1.log", rotation="10 MB", level="DEBUG")
    print("日志框架已经创建")
    return logger

log_driver = create_log()