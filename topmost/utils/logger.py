import logging  # 导入日志模块


class Logger:  # 定义日志记录器类
    def __init__(self, level):  # 初始化方法
        self.logger = logging.getLogger('TopMost')  # 创建TopMost日志记录器
        self.set_level(level)  # 设置日志级别
        self._add_handler()  # 添加处理器
        self.logger.propagate = False  # 禁止日志传播

    def info(self, message):  # 信息日志方法
        self.logger.info(f"{message}")  # 记录信息日志

    def warning(self, message):  # 警告日志方法
        self.logger.warning(f"WARNING: {message}")  # 记录警告日志

    def set_level(self, level):  # 设置日志级别方法
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]  # 定义有效日志级别
        if level in levels:  # 如果级别有效
            self.logger.setLevel(level)  # 设置日志级别

    def _add_handler(self):  # 添加处理器方法
        sh = logging.StreamHandler()  # 创建流处理器
        sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))  # 设置格式化器
        self.logger.addHandler(sh)  # 添加处理器到日志记录器

        # Remove duplicate handlers  # 移除重复处理器
        if len(self.logger.handlers) > 1:  # 如果处理器数量大于1
            self.logger.handlers = [self.logger.handlers[0]]  # 只保留第一个处理器
