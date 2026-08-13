import unittest

from stt.whisper_worker import (
    choose_vad_threshold,
)


class VADEchoGuardTests(unittest.TestCase):

    def test_normal_threshold_without_playback(self):
        self.assertEqual(
            choose_vad_threshold(
                0.0,
                normal_threshold=0.015,
                playback_threshold=0.035,
                enabled=True,
            ),
            0.015,
        )

    def test_higher_threshold_during_playback(self):
        self.assertEqual(
            choose_vad_threshold(
                1.0,
                normal_threshold=0.015,
                playback_threshold=0.035,
                enabled=True,
            ),
            0.035,
        )

    def test_disabled_guard_keeps_normal_threshold(self):
        self.assertEqual(
            choose_vad_threshold(
                1.0,
                normal_threshold=0.015,
                playback_threshold=0.035,
                enabled=False,
            ),
            0.015,
        )


if __name__ == "__main__":
    unittest.main()
