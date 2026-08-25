import unittest

from stt.echo_guard import (
    EchoGuard,
    choose_vad_threshold,
    is_echo_transcript,
    normalize_echo_text,
)


class VADEchoGuardTests(unittest.TestCase):

    def test_normal_threshold_without_playback(self):
        self.assertEqual(
            choose_vad_threshold(
                0.0,
                normal_threshold=0.015,
                playback_threshold=0.10,
                enabled=True,
            ),
            0.015,
        )

    def test_higher_threshold_during_playback(self):
        self.assertEqual(
            choose_vad_threshold(
                1.0,
                normal_threshold=0.015,
                playback_threshold=0.10,
                enabled=True,
            ),
            0.10,
        )

    def test_disabled_guard_keeps_normal_threshold(self):
        self.assertEqual(
            choose_vad_threshold(
                1.0,
                normal_threshold=0.015,
                playback_threshold=0.10,
                enabled=False,
            ),
            0.015,
        )


class EchoHangoverTests(unittest.TestCase):

    def test_hangover_keeps_guard_after_buffer_drops(self):
        guard = EchoGuard(
            hangover_seconds=0.7,
            barge_in_threshold=0.10,
            normal_threshold=0.025,
        )
        now = 100.0
        guard.observe_playback(1.2, now=now)
        later = now + 0.4
        self.assertTrue(guard.is_hot(now=later))
        self.assertEqual(
            guard.threshold(now=later),
            0.10,
        )

    def test_ignore_quiet_echo_during_hot_window(self):
        guard = EchoGuard(
            hangover_seconds=0.7,
            barge_in_threshold=0.10,
            normal_threshold=0.025,
        )
        guard.observe_playback(0.8, now=10.0)
        self.assertTrue(
            guard.should_ignore_start(
                0.036,
                now=10.2,
            )
        )

    def test_allow_loud_barge_in_during_playback(self):
        guard = EchoGuard(
            hangover_seconds=0.7,
            barge_in_threshold=0.10,
            normal_threshold=0.025,
        )
        guard.observe_playback(0.8, now=10.0)
        self.assertFalse(
            guard.should_ignore_start(
                0.14,
                now=10.1,
            )
        )

    def test_idle_after_hangover(self):
        guard = EchoGuard(
            hangover_seconds=0.7,
            barge_in_threshold=0.10,
            normal_threshold=0.025,
        )
        guard.observe_playback(0.8, now=10.0)
        self.assertFalse(
            guard.is_hot(now=10.8)
        )
        self.assertEqual(
            guard.threshold(now=10.8),
            0.025,
        )


class EchoTranscriptTests(unittest.TestCase):

    def test_prefix_of_spoken_line_is_echo_while_hot(self):
        spoken = [
            {
                "text": "Merhaba, nasılsın? Yardımcı olabileceğim bir konu var mı?",
                "ts": 20.0,
            }
        ]
        self.assertTrue(
            is_echo_transcript(
                "Merhaba.",
                spoken,
                playback_hot=True,
                now=20.4,
            )
        )

    def test_same_greeting_allowed_when_idle(self):
        spoken = [
            {
                "text": "Merhaba, nasılsın? Yardımcı olabileceğim bir konu var mı?",
                "ts": 10.0,
            }
        ]
        self.assertFalse(
            is_echo_transcript(
                "Merhaba.",
                spoken,
                playback_hot=False,
                now=16.0,
            )
        )

    def test_near_duplicate_sentence_is_echo(self):
        spoken = [
            {
                "text": "Harika! Bugün sana nasıl yardımcı olabilirim?",
                "ts": 5.0,
            }
        ]
        self.assertTrue(
            is_echo_transcript(
                "Harika bugün sana nasıl yardımcı olabilirim",
                spoken,
                playback_hot=False,
                now=5.5,
            )
        )

    def test_unrelated_reply_is_kept(self):
        spoken = [
            {
                "text": "Harika! Bugün sana nasıl yardımcı olabilirim?",
                "ts": 5.0,
            }
        ]
        self.assertFalse(
            is_echo_transcript(
                "saat kaç",
                spoken,
                playback_hot=True,
                now=5.3,
            )
        )

    def test_normalize_strips_punctuation(self):
        self.assertEqual(
            normalize_echo_text("Merhaba!"),
            "merhaba",
        )


if __name__ == "__main__":
    unittest.main()
