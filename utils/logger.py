import sys
import enum
from datetime import datetime

class Color(enum.Enum):
    """颜色枚举"""
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'

class Logger:
    """自定义logger"""
    def __init__(self, stream=sys.stdout, date_format='%Y-%m-%d %H:%M:%S'):
        self.stream = stream
        self.date_format = date_format
        self.level_colors = {
            'info': Color.GREEN,
            'debug': Color.BLUE,
            'warning': Color.YELLOW,
            'error': Color.RED,
            'critical': Color.MAGENTA
        }

    def log(self, message, level, end='\n', flush=False):
        """打印消息"""
        color = self.level_colors.get(level, None)
        if end == '\n':
            current_time = datetime.now().strftime(self.date_format)
            log_message = f"{current_time} - {message}"
            self.iter_print = False
        elif not self.iter_print:
            current_time = datetime.now().strftime(self.date_format)
            log_message = f"{current_time} - {message}"
            self.iter_print = True
        else:
            log_message = message
        
        if color:
            log_message = f"{color.value}{log_message}{Color.RESET.value}"
        self.stream.write(log_message + end)
        if flush:
            self.stream.flush()

    def info(self, message, end='\n', flush=False):
        """信息日志"""
        self.log(message, 'info', end, flush)

    def debug(self, message, end='\n', flush=False):
        """调试日志"""
        self.log(message, 'debug', end, flush)

    def warning(self, message, end='\n', flush=False):
        """警告日志"""
        self.log(message, 'warning', end, flush)

    def error(self, message, end='\n', flush=False):
        """错误日志"""
        self.log(message, 'error', end, flush)

    def critical(self, message, end='\n', flush=False):
        """严重错误日志"""
        self.log(message, 'critical', end, flush)

# 使用示例
logger = Logger()

# # 打印不同颜色的消息
# logger.error("这是一条红色的消息", end='\n')
# logger.debug("这是一条蓝色的消息", end=' ')
# logger.info("这是一条绿色的消息", end=' ')
# logger.warning("这是一条黄色的消息", end='\n')

# # 使用end和flush
# logger.debug("没有换行", end=' ')
# logger.info("还是没有换行", end=' ', flush=True)
# logger.warning("接着打印", end='\n', flush=True)