import unittest
from unittest.mock import patch

from llm.sentence_buffer import (
    FIRST_CHUNK_MIN_CHARS,
    TEXT_QUEUE,
    SentenceBuffer,
)


class SentenceBufferTests(unittest.TestCase):

    def setUp(self):
        patcher = patch(
            "llm.sentence_buffer.text_redis"
        )
        self.redis = patcher.start()
        self.addCleanup(patcher.stop)

    def sent_chunks(self):
        return [
            call.args[1]
            for call in self.redis.lpush.call_args_list
        ]

    def test_long_first_sentence_is_sent_immediately(self):
        sentence = (
            "Bu ilk cümle, kullanıcıya hızlı bir başlangıç "
            "sunabilecek kadar uzun hazırlanmıştır."
        )

        self.assertGreaterEqual(
            len(sentence),
            FIRST_CHUNK_MIN_CHARS,
        )

        buffer = SentenceBuffer()
        buffer.add(sentence)

        self.redis.lpush.assert_called_once_with(
            TEXT_QUEUE,
            sentence,
        )

    def test_two_short_sentences_are_sent_together(self):
        text = "Merhaba. Nasılsın?"

        buffer = SentenceBuffer()
        buffer.add(text)

        self.redis.lpush.assert_called_once_with(
            TEXT_QUEUE,
            text,
        )

    def test_decimal_dot_is_not_sentence_end(self):
        text = "Sürüm 1.5 kararlı."

        buffer = SentenceBuffer()
        buffer.add(text)

        self.redis.lpush.assert_not_called()

        buffer.flush()

        self.redis.lpush.assert_called_once_with(
            TEXT_QUEUE,
            text,
        )

    def test_max_length_splits_at_word_boundary(self):
        text = ("kelime " * 45).strip()

        buffer = SentenceBuffer()
        buffer.add(text)
        buffer.flush()

        chunks = self.sent_chunks()

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(
            " ".join(chunks),
            text,
        )

    def test_flush_sends_remaining_text_and_resets_turn(self):
        buffer = SentenceBuffer()

        buffer.add("Kısa bir parça")
        buffer.flush()

        self.assertFalse(
            buffer.sent_first_chunk
        )
        self.assertEqual(
            self.sent_chunks(),
            ["Kısa bir parça"],
        )


if __name__ == "__main__":
    unittest.main()
