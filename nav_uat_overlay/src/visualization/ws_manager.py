from queue import Empty, Queue
from threading import Lock


class WebSocketManager:
    def __init__(self):
        self._lock = Lock()
        self._queues = []

    def register(self):
        queue = Queue()
        with self._lock:
            self._queues.append(queue)
        return queue

    def unregister(self, queue):
        with self._lock:
            if queue in self._queues:
                self._queues.remove(queue)

    def broadcast(self, message):
        with self._lock:
            queues = list(self._queues)
        for queue in queues:
            queue.put(message)

    @staticmethod
    def get_nowait(queue):
        try:
            return queue.get_nowait()
        except Empty:
            return None
