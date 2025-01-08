from typing import AsyncIterator
import asyncio

class TextSegmenter:
    def __init__(self, min_length: int = 10):
        """
        简单的文本分段器
        
        Args:
            min_length: 最小句子长度
        """
        self.min_length = min_length
        self.delimiters = "。！？；：\\n"
        self.pending = ""
        
    async def process(self, text_iterator: AsyncIterator[str]) -> AsyncIterator[str]:
        """处理文本流"""
        async for chunk in text_iterator:
            self.pending += chunk
            
            # 查找所有分隔符位置
            positions = []
            for i, char in enumerate(self.pending):
                if char in self.delimiters:
                    positions.append(i)
            
            if positions:
                last_pos = 0
                current = ""
                
                # 处理每个分隔位置
                for pos in positions:
                    segment = self.pending[last_pos:pos + 1]
                    current += segment
                    
                    # 当累积文本长度超过最小长度时输出
                    if len(current) >= self.min_length:
                        if current.strip():
                            yield current.strip()
                        current = ""
                    
                    last_pos = pos + 1
                
                # 保留未处理完的文本
                self.pending = self.pending[last_pos:]
        
        # 处理剩余文本
        if self.pending and len(self.pending.strip()) >= self.min_length:
            yield self.pending.strip()
            self.pending = ""

    async def process_text(self, text: str) -> AsyncIterator[str]:
        """处理单个文本字符串
        
        Args:
            text: 需要分段的文本字符串
        
        Returns:
            AsyncIterator[str]: 分段后的文本迭代器
        """
        if not text or not text.strip():
            return
        
        self.pending += text
        
        # 查找所有分隔符位置
        positions = []
        for i, char in enumerate(self.pending):
            if char in self.delimiters:
                positions.append(i)
            
        if positions:
            last_pos = 0
            current = ""
            
            # 处理每个分隔位置
            for pos in positions:
                segment = self.pending[last_pos:pos + 1]
                current += segment
                
                # 当累积文本长度达到要求时输出
                if len(current.strip()) >= self.min_length:
                    yield current.strip()
                    current = ""
                    
                last_pos = pos + 1
            
            # 保留未处理完的文本
            self.pending = current + self.pending[last_pos:]
        
        # 如果剩余文本超过最小长度且包含分隔符,输出剩余文本
        if len(self.pending.strip()) >= self.min_length and any(c in self.pending for c in self.delimiters):
            yield self.pending.strip()
            self.pending = ""

async def demo():
    """示例"""
    import asyncio
    
    segmenter = TextSegmenter(min_length=10)
    
    async def text_stream():
        text_parts = [
            "我",
            "能够",
            "做",
            "很多事情，主要包括以下几个",
            "方面：\n",
            "1.**回答问题和",
            "提供信息**\n",
            "无论是科学知识、",
            "历史事件、实用",
            "技巧还是其他领域的",
            "信息查询，我",
            "都可以提供详细的解答\n",
            "2.**创作文字内容**\n",
            "\n我可以帮助你",
            "写故事、公",
            "文、邮件、",
            "剧本等，根据",
            "你的需求生成高质量",
            "的文字内容。"
        ]
        for part in text_parts:
            yield part
            await asyncio.sleep(0.1)
    
    async for sentence in segmenter.process(text_stream()):
        print(f"输出: {sentence}")

if __name__ == "__main__":
    asyncio.run(demo())
