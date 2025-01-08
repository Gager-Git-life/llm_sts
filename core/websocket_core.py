from abc import abstractmethod
import asyncio
import websockets
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketServer():
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.send_queues = {}

    @abstractmethod
    def data_process(self, websocket, data):
        pass    

    async def sender(self, websocket):
        """处理发送消息的协程"""
        try:
            queue = self.send_queues[websocket]
            while True:
                message = await queue.get()
                if message is None:
                    break
                await websocket.send(message)
                queue.task_done()
        finally:
            if websocket in self.send_queues:
                del self.send_queues[websocket]

    async def receiver(self, websocket):
        """处理接收消息的协程"""
        try:
            async for message in websocket:
                await self.data_process(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"Receiver error: {str(e)}")

    async def handle_client(self, websocket):
        """处理客户端连接"""
        self.clients.add(websocket)
        self.send_queues[websocket] = asyncio.Queue()
        
        try:
            sender_task = asyncio.create_task(self.sender(websocket))
            receiver_task = asyncio.create_task(self.receiver(websocket))
            await asyncio.gather(sender_task, receiver_task)
        except Exception as e:
            logger.error(f"Handler error: {str(e)}")
        finally:
            self.clients.remove(websocket)
            if websocket in self.send_queues:
                await self.send_queues[websocket].put(None)

    async def start(self):
        """启动WebSocket服务器"""
        async with websockets.serve(
            self.handle_client, 
            self.host, 
            self.port,
            ping_interval=300,    # 5分钟发送一次 ping
            ping_timeout=120      # 2分钟的 ping 超时时间
        ):
            logger.info(f"Server started on ws://{self.host}:{self.port}")
            await asyncio.Future()

class WebSocketClient:
    def __init__(self, uri="ws://localhost:8765"):
        self.uri = uri
        self.websocket = None
        self.send_queue = asyncio.Queue()
        self.receive_queue = asyncio.Queue()
        self.running = False

    async def connect(self):
        """连接到WebSocket服务器"""
        try:
            self.websocket = await websockets.connect(
                self.uri,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5
            )
            self.running = True
            return True
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            return False

    async def sender(self):
        """发送消息的协程"""
        try:
            while self.running:
                try:
                    message = await asyncio.wait_for(self.send_queue.get(), timeout=1.0)
                    if message is None:
                        break
                    await self.websocket.send(message)
                    self.send_queue.task_done()
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Failed to send message: {str(e)}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.running = False

    async def receiver(self):
        """接收消息的协程"""
        try:
            while self.running:
                try:
                    message = await self.websocket.recv()
                    await self.receive_queue.put(message)
                except websockets.exceptions.ConnectionClosed:
                    break
                except Exception as e:
                    logger.error(f"Receiver error: {str(e)}")
                    break
        finally:
            self.running = False
            await self.send_queue.put(None)

    async def send_message(self, message):
        """将消息放入发送队列"""
        if self.running:
            await self.send_queue.put(message)

    async def get_message(self):
        """从接收队列获取消息"""
        return await self.receive_queue.get()

    async def run(self):
        """运行客户端"""
        if not await self.connect():
            return

        try:
            sender_task = asyncio.create_task(self.sender())
            receiver_task = asyncio.create_task(self.receiver())
            await asyncio.gather(sender_task, receiver_task)
        except Exception as e:
            logger.error(f"Run error: {str(e)}")
        finally:
            self.running = False
            await self.close()

    async def close(self):
        """关闭连接"""
        self.running = False
        if self.websocket:
            await self.websocket.close()
            self.websocket = None 