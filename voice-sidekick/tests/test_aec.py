import time
import unittest
from unittest.mock import patch

import numpy as np

from stt.aec import AcousticEchoCanceller, ReferenceBuffer


def make_buffer(
    sample_rate=1000,
    delay_seconds=0.0,
    max_age_seconds=2.0,
):
    with patch("stt.aec.audio_redis") as mock_redis:
        mock_redis.xrevrange.return_value = []
        buf = ReferenceBuffer(
            sample_rate, delay_seconds, max_age_seconds
        )
    return buf


class ReferenceBufferAlignmentTests(unittest.TestCase):

    def test_returns_none_when_empty(self):
        buf = make_buffer()
        self.assertIsNone(
            buf.get_aligned(time.time(), 100)
        )

    def test_aligns_within_single_chunk(self):
        buf = make_buffer(
            sample_rate=1000, delay_seconds=0.0
        )
        chunk = np.arange(1000, dtype=np.int16)
        buf._chunks.append((100.0, chunk))

        result = buf.get_aligned(100.2, 50)
        np.testing.assert_array_equal(
            result, chunk[200:250]
        )

    def test_applies_delay_offset(self):
        buf = make_buffer(
            sample_rate=1000, delay_seconds=0.05
        )
        chunk = np.arange(1000, dtype=np.int16)
        buf._chunks.append((100.0, chunk))

        result = buf.get_aligned(100.25, 50)
        np.testing.assert_array_equal(
            result, chunk[200:250]
        )

    def test_spans_two_chunks(self):
        buf = make_buffer(
            sample_rate=1000, delay_seconds=0.0
        )
        chunk1 = np.arange(0, 500, dtype=np.int16)
        chunk2 = np.arange(500, 1000, dtype=np.int16)
        buf._chunks.append((100.0, chunk1))
        buf._chunks.append((100.5, chunk2))

        result = buf.get_aligned(100.48, 40)
        expected = np.arange(480, 520, dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

    def test_none_when_not_enough_trailing_history(self):
        buf = make_buffer(
            sample_rate=1000, delay_seconds=0.0
        )
        chunk = np.arange(100, dtype=np.int16)
        buf._chunks.append((100.0, chunk))

        result = buf.get_aligned(100.09, 50)
        self.assertIsNone(result)

    def test_trim_drops_old_chunks(self):
        buf = make_buffer(
            sample_rate=1000, max_age_seconds=1.0
        )
        buf._chunks.append(
            (100.0, np.zeros(10, dtype=np.int16))
        )
        buf._chunks.append(
            (101.5, np.zeros(10, dtype=np.int16))
        )
        buf._trim(now=102.0)

        self.assertEqual(len(buf._chunks), 1)
        self.assertAlmostEqual(
            buf._chunks[0][0], 101.5
        )


class AcousticEchoCancellerTests(unittest.TestCase):

    def test_passthrough_when_reference_unavailable(
        self,
    ):
        with patch(
            "stt.aec.audio_redis"
        ) as mock_redis:
            mock_redis.xrevrange.return_value = []
            mock_redis.xread.return_value = []
            aec = AcousticEchoCanceller(
                sample_rate=16000,
                frame_len=160,
                filter_length=512,
                delay_ms=50,
            )

        frame = (
            np.random.default_rng(0)
            .uniform(-0.1, 0.1, 160)
            .astype(np.float32)
        )
        clean, applied = aec.process(
            frame, time.time()
        )

        self.assertFalse(applied)
        np.testing.assert_array_equal(clean, frame)

    def test_cancels_aligned_echo(self):
        sample_rate = 16000
        frame_len = 256

        with patch(
            "stt.aec.audio_redis"
        ) as mock_redis:
            mock_redis.xrevrange.return_value = []
            aec = AcousticEchoCanceller(
                sample_rate=sample_rate,
                frame_len=frame_len,
                filter_length=1024,
                delay_ms=0,
            )

        if not aec.enabled:
            self.skipTest("pyaec not available")

        base_time = 1000.0
        samples = frame_len * 30
        tone = (
            0.3
            * np.sin(
                2
                * np.pi
                * 440
                * np.arange(samples)
                / sample_rate
            )
        ).astype(np.float32)

        chunk = (tone * 32767).astype(np.int16)
        aec.reference._chunks.append(
            (base_time, chunk)
        )
        aec.reference.ingest_new = lambda: None

        mic_rms_before = []
        out_rms_after = []

        for i in range(
            0, samples - frame_len, frame_len
        ):
            frame = tone[i:i + frame_len]
            frame_time = (
                base_time + i / sample_rate
            )
            clean, applied = aec.process(
                frame, frame_time
            )
            self.assertTrue(applied)

            mic_rms_before.append(
                float(
                    np.sqrt(
                        np.mean(
                            frame.astype(np.float64)
                            ** 2
                        )
                    )
                )
            )
            out_rms_after.append(
                float(
                    np.sqrt(
                        np.mean(
                            clean.astype(np.float64)
                            ** 2
                        )
                    )
                )
            )

        # Skip the first few blocks while the adaptive filter
        # converges; after that, residual echo should be
        # substantially quieter than the raw mic signal.
        self.assertLess(
            np.mean(out_rms_after[10:]),
            np.mean(mic_rms_before[10:]) * 0.5,
        )


if __name__ == "__main__":
    unittest.main()
