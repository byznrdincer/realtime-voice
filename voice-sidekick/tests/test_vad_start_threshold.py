import unittest

import numpy as np

from stt.vad import EnergyVAD


def constant_frame(level, samples=1440):
    return np.full(samples, level, dtype=np.float32)


class StartThresholdTests(unittest.TestCase):

    def test_defaults_to_shared_threshold(self):
        vad = EnergyVAD(sample_rate=48000, threshold=0.05)
        self.assertEqual(vad.start_threshold, 0.05)

    def test_quiet_onset_can_start_with_lower_start_threshold(self):
        vad = EnergyVAD(
            sample_rate=48000,
            threshold=0.10,
            start_threshold=0.02,
            start_ms=120,
            frame_ms=30,
        )
        frame = constant_frame(0.03)
        events = [vad.process(frame) for _ in range(4)]
        self.assertEqual(events[-1][0], "start")

    def test_onset_still_blocked_below_start_threshold(self):
        vad = EnergyVAD(
            sample_rate=48000,
            threshold=0.10,
            start_threshold=0.05,
            start_ms=120,
            frame_ms=30,
        )
        frame = constant_frame(0.03)
        events = [vad.process(frame) for _ in range(6)]
        self.assertTrue(
            all(event[0] != "start" for event in events)
        )

    def test_active_phase_keeps_using_normal_threshold(self):
        vad = EnergyVAD(
            sample_rate=48000,
            threshold=0.10,
            start_threshold=0.02,
            start_ms=120,
            end_ms=90,
            frame_ms=30,
        )
        loud = constant_frame(0.15)
        for _ in range(4):
            vad.process(loud)
        self.assertTrue(vad.active)

        quiet = constant_frame(0.03)
        event = None
        for _ in range(3):
            event, _audio, _level = vad.process(quiet)

        self.assertEqual(event, "end")


if __name__ == "__main__":
    unittest.main()
