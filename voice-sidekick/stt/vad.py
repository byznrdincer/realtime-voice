import math
from collections import deque

import numpy as np


class EnergyVAD:
    def __init__(
        self,
        sample_rate=48000,
        frame_ms=30,
        threshold=0.015,
        start_ms=120,
        end_ms=700,
        pre_roll_ms=300,
        max_speech_seconds=20,
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.threshold = threshold

        self.frame_samples = int(
            sample_rate * frame_ms / 1000
        )

        self.start_frames = max(
            1,
            math.ceil(start_ms / frame_ms),
        )
        self.end_frames = max(
            1,
            math.ceil(end_ms / frame_ms),
        )
        self.max_frames = max(
            1,
            math.ceil(
                max_speech_seconds
                * 1000
                / frame_ms
            ),
        )

        pre_roll_frames = max(
            1,
            math.ceil(pre_roll_ms / frame_ms),
        )

        self.pre_roll = deque(
            maxlen=pre_roll_frames
        )

        self.reset()

    def reset(self):
        self.active = False
        self.speech_run = 0
        self.silence_run = 0
        self.active_frames = 0
        self.recorded_frames = []
        self.pre_roll.clear()

    @staticmethod
    def rms(frame):
        frame = np.asarray(
            frame,
            dtype=np.float32,
        ).reshape(-1)

        if frame.size == 0:
            return 0.0

        return float(
            np.sqrt(
                np.mean(
                    np.square(frame)
                )
            )
        )

    def process(self, frame):
        frame = np.asarray(
            frame,
            dtype=np.float32,
        ).reshape(-1)

        level = self.rms(frame)
        voiced = level >= self.threshold

        if not self.active:
            self.pre_roll.append(
                frame.copy()
            )

            if voiced:
                self.speech_run += 1
            else:
                self.speech_run = 0

            if (
                self.speech_run
                >= self.start_frames
            ):
                self.active = True
                self.recorded_frames = list(
                    self.pre_roll
                )
                self.active_frames = len(
                    self.recorded_frames
                )
                self.pre_roll.clear()
                self.silence_run = 0

                return "start", None, level

            return None, None, level

        self.recorded_frames.append(
            frame.copy()
        )
        self.active_frames += 1

        if voiced:
            self.silence_run = 0
        else:
            self.silence_run += 1

        finished = (
            self.silence_run
            >= self.end_frames
            or self.active_frames
            >= self.max_frames
        )

        if not finished:
            return "speech", None, level

        audio = np.concatenate(
            self.recorded_frames
        )

        self.reset()

        return "end", audio, level
