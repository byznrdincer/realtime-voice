from queue_service.redis_client import text_redis, TEXT_QUEUE


MIN_CHARS = 40


class SentenceBuffer:

    def __init__(self):
        self.buffer = ""

    def add(self, token):

        self.buffer += token

        should_send = (
            len(self.buffer) >= MIN_CHARS
            and self.buffer.endswith((".", "!", "?"))
        )

        if should_send:

            chunk = self.buffer.strip()

            text_redis.lpush(
                TEXT_QUEUE,
                chunk
            )

            print(f"[REDIS] {chunk}")

            self.buffer = ""

    def flush(self):

        if self.buffer.strip():

            text_redis.lpush(
                TEXT_QUEUE,
                self.buffer.strip()
            )

            print(f"[REDIS] {self.buffer.strip()}")

            self.buffer = ""

