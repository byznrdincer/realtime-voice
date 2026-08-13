import unittest

from queue_service.messages import (
    decode_audio_message,
    decode_text_message,
    encode_audio_message,
    encode_text_message,
)


class QueueMessageTests(unittest.TestCase):

    def test_text_round_trip(self):
        encoded = encode_text_message(
            12,
            "Merhaba Beyzanur.",
        )

        self.assertEqual(
            decode_text_message(encoded),
            (12, "Merhaba Beyzanur."),
        )

    def test_legacy_text_is_supported(self):
        self.assertEqual(
            decode_text_message(
                "Eski düz metin"
            ),
            (None, "Eski düz metin"),
        )

    def test_audio_round_trip(self):
        pcm = b"\x00\x01\x02\xff"

        encoded = encode_audio_message(
            34,
            pcm,
        )

        self.assertEqual(
            decode_audio_message(encoded),
            (34, pcm),
        )

    def test_legacy_audio_is_supported(self):
        pcm = b"\x00\x00\x01\x00"

        self.assertEqual(
            decode_audio_message(pcm),
            (None, pcm),
        )

    def test_invalid_turn_id_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            encode_text_message(
                -1,
                "Geçersiz",
            )


if __name__ == "__main__":
    unittest.main()
