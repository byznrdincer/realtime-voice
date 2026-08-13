import unittest
from unittest.mock import patch

from llm.sentence_buffer import SentenceBuffer
from queue_service.messages import (
    decode_audio_message,
    decode_text_message,
    encode_audio_message,
)


class TurnIsolationTests(unittest.TestCase):

    @patch(
        "llm.sentence_buffer.text_redis"
    )
    def test_sentence_buffer_attaches_turn_id(
        self,
        redis_mock,
    ):
        buffer = SentenceBuffer(turn_id=17)
        buffer.add("Merhaba.")
        buffer.flush()

        payload = (
            redis_mock.lpush
            .call_args.args[1]
        )

        turn_id, text = decode_text_message(
            payload
        )

        self.assertEqual(turn_id, 17)
        self.assertEqual(text, "Merhaba.")

    def test_old_audio_is_rejected(self):
        payload = encode_audio_message(
            4,
            b"old",
        )

        turn_id, audio = decode_audio_message(
            payload
        )

        current_turn_id = 5

        self.assertNotEqual(
            turn_id,
            current_turn_id,
        )
        self.assertEqual(audio, b"old")

    def test_new_audio_survives_old_pop_state(self):
        pop_started_turn_id = 4
        current_turn_id = 5

        payload = encode_audio_message(
            current_turn_id,
            b"new",
        )

        turn_id, audio = decode_audio_message(
            payload
        )

        self.assertNotEqual(
            pop_started_turn_id,
            current_turn_id,
        )
        self.assertEqual(
            turn_id,
            current_turn_id,
        )
        self.assertEqual(audio, b"new")


if __name__ == "__main__":
    unittest.main()
