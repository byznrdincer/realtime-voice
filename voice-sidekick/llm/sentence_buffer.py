import re

from queue_service.redis_client import text_redis, TEXT_QUEUE

MIN_SENTENCES = 2
MAX_CHARS = 220

SENTENCE_END_PATTERN = re.compile(
    r"(?<!\d)[.!?](?=\s|$)"
)


class SentenceBuffer:

    def __init__(self):
        self.buffer = ""

    def _send(self, cutoff):

        chunk = self.buffer[:cutoff].strip()
        self.buffer = self.buffer[cutoff:].lstrip()

        if not chunk:
            return

        text_redis.lpush(
            TEXT_QUEUE,
            chunk
        )

        print(f"[REDIS] {chunk}")

    def _emit_ready(self):

        while self.buffer:

            matches = list(
                SENTENCE_END_PATTERN.finditer(
                    self.buffer
                )
            )

            cutoff = None

            # Normal durumda iki cümleyi birlikte gönder.
            if len(matches) >= MIN_SENTENCES:
                cutoff = matches[
                    MIN_SENTENCES - 1
                ].end()

            # Metin fazla büyüdüyse mevcut en uygun
            # cümle sınırından gönder.
            elif len(self.buffer) >= MAX_CHARS:

                if matches:
                    cutoff = matches[-1].end()
                else:
                    cutoff = self.buffer.rfind(
                        " ",
                        0,
                        MAX_CHARS,
                    )

                    if cutoff <= 0:
                        cutoff = MAX_CHARS

            if cutoff is None:
                return

            self._send(cutoff)

    def add(self, token):

        self.buffer += token
        self._emit_ready()

    def flush(self):

        self._emit_ready()

        if self.buffer.strip():
            self._send(len(self.buffer))
