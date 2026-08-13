import re

from queue_service.messages import encode_text_message
from queue_service.redis_client import (
    text_redis,
    TEXT_QUEUE,
)

MIN_SENTENCES = 2
FIRST_CHUNK_MIN_CHARS = 55
MAX_CHARS = 220

SENTENCE_END_PATTERN = re.compile(
    r"(?<!\d)[.!?](?=\s|$)"
)


class SentenceBuffer:

    def __init__(self, turn_id=None):
        self.turn_id = turn_id
        self.buffer = ""
        self.sent_first_chunk = False

    def _send(self, cutoff):

        chunk = self.buffer[:cutoff].strip()
        self.buffer = (
            self.buffer[cutoff:].lstrip()
        )

        if not chunk:
            return

        payload = (
            encode_text_message(
                self.turn_id,
                chunk,
            )
            if self.turn_id is not None
            else chunk
        )

        text_redis.lpush(
            TEXT_QUEUE,
            payload,
        )

        print(f"[REDIS] {chunk}")

        self.sent_first_chunk = True

    def _emit_ready(self):

        while self.buffer:

            matches = list(
                SENTENCE_END_PATTERN.finditer(
                    self.buffer
                )
            )

            cutoff = None

            # İlk yeterince uzun cümleyi
            # ikinci cümleyi beklemeden gönder.
            if (
                not self.sent_first_chunk
                and matches
                and matches[0].end()
                >= FIRST_CHUNK_MIN_CHARS
            ):
                cutoff = matches[0].end()

            # Sonraki parçalarda iki cümleyi
            # birlikte gönder.
            elif len(matches) >= MIN_SENTENCES:
                cutoff = matches[
                    MIN_SENTENCES - 1
                ].end()

            # Buffer fazla büyürse uygun
            # cümle veya kelime sınırından gönder.
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

    def reset(self):
        self.buffer = ""
        self.sent_first_chunk = False

    def flush(self):

        self._emit_ready()

        if self.buffer.strip():
            self._send(len(self.buffer))

        # Yeni kullanıcı turunda ilk cümle
        # yeniden hızlı gönderilsin.
        self.sent_first_chunk = False
