from examples.pipeline_simple import PipelineManager
from utils.logger import logger
import asyncio

if __name__ == "__main__":
    # 创建pipeline管理器实例
    pipeline = PipelineManager()
    
    # 运行事件循环
    try:
        asyncio.run(pipeline.start())
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}") 