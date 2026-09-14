import time
from collections import deque

import numpy as np
import redis

from queue_service.redis_client import (
    audio_redis,
    AEC_REFERENCE_STREAM_KEY,
)

try:
    from pyaec import Aec
except ImportError:
    Aec = None


# Below this int16-scale RMS, treat the reference as "no real
# playback" rather than feeding it to the filter. Running the
# adaptive filter against a near-silent reference is a degenerate
# case for this library: it can end up aggressively attenuating any
# stationary/tonal content in the mic signal (which includes voiced
# speech), not just genuine echo, even though there's nothing to
# actually cancel. Measured empirically: ~50 safely separates true
# silence from an audibly present (even quiet) TTS tail at 48kHz.
MIN_REFERENCE_RMS = 50.0


class ReferenceBuffer:
    """Tracks recently-played TTS audio (the AEC far-end/reference
    signal), received from the audio player process over a Redis
    stream, indexed by wall-clock timestamp.

    The audio player's output callback runs continuously (it emits
    silence between TTS turns, not gaps), so the stream is a gapless
    timeline. Reads are polled from whisper_worker's own per-frame
    loop rather than blocking, so a slow/late reference never stalls
    mic processing.
    """

    def __init__(
        self,
        sample_rate,
        delay_seconds,
        max_age_seconds=2.0,
    ):
        self.sample_rate = sample_rate
        self.delay_seconds = delay_seconds
        self.max_age_seconds = max_age_seconds
        self._chunks = deque()
        self._last_id = self._latest_id()

    def _latest_id(self):
        try:
            tail = audio_redis.xrevrange(
                AEC_REFERENCE_STREAM_KEY,
                count=1,
            )
        except redis.exceptions.RedisError:
            return "0"

        if not tail:
            return "0"

        return tail[0][0]

    def ingest_new(self):
        try:
            entries = audio_redis.xread(
                {
                    AEC_REFERENCE_STREAM_KEY: (
                        self._last_id
                    )
                },
                count=64,
            )
        except redis.exceptions.RedisError:
            return

        if not entries:
            return

        for _stream_key, messages in entries:
            for entry_id, fields in messages:
                self._last_id = entry_id

                ts_raw = fields.get(b"ts")
                pcm_raw = fields.get(b"pcm")

                if ts_raw is None or pcm_raw is None:
                    continue

                try:
                    ts = float(ts_raw)
                except ValueError:
                    continue

                samples = np.frombuffer(
                    pcm_raw, dtype="<i2"
                )

                if samples.size == 0:
                    continue

                self._chunks.append((ts, samples))

        self._trim(time.time())

    def _trim(self, now):
        cutoff = now - self.max_age_seconds

        while (
            self._chunks
            and self._chunks[0][0] < cutoff
        ):
            self._chunks.popleft()

    def get_aligned(self, frame_time, frame_len):
        """Return `frame_len` int16 reference samples covering the
        time window `[frame_time - delay, ...)`, or None if the
        buffer doesn't have enough history there yet (e.g. TTS
        hasn't produced output covering that window, or AEC just
        started up).
        """
        if not self._chunks:
            return None

        target_start = (
            frame_time - self.delay_seconds
        )
        target_end = (
            target_start
            + frame_len / self.sample_rate
        )

        collected = []
        collected_start = None

        for chunk_ts, samples in self._chunks:
            chunk_end = (
                chunk_ts
                + len(samples) / self.sample_rate
            )

            if chunk_end <= target_start:
                continue
            if chunk_ts >= target_end:
                break

            if collected_start is None:
                collected_start = chunk_ts
            collected.append(samples)

        if not collected:
            return None

        merged = np.concatenate(collected)
        offset = int(
            round(
                (target_start - collected_start)
                * self.sample_rate
            )
        )

        if offset < 0:
            pad = np.zeros(
                -offset, dtype=np.int16
            )
            merged = np.concatenate([pad, merged])
            offset = 0

        if offset + frame_len > len(merged):
            return None

        return merged[offset:offset + frame_len]


class AcousticEchoCanceller:
    """Cancels TTS speaker bleed-through from mic frames using an
    adaptive (NLMS-style) filter against the actual audio sent to
    the speaker, instead of only gating on RMS thresholds.

    Falls back to passing the frame through unchanged whenever a
    reference isn't available (TTS not playing, or not enough
    history yet) or the pyaec dependency/library failed to load, so
    this is always safe to call unconditionally.
    """

    def __init__(
        self,
        sample_rate,
        frame_len,
        filter_length,
        delay_ms,
        max_age_seconds=2.0,
    ):
        self.enabled = Aec is not None
        self.sample_rate = sample_rate
        self.frame_len = frame_len
        self.applied_count = 0
        self.skipped_count = 0

        self.reference = ReferenceBuffer(
            sample_rate,
            delay_ms / 1000.0,
            max_age_seconds,
        )

        self._aec = (
            Aec(
                frame_len,
                filter_length,
                sample_rate,
                True,
            )
            if self.enabled
            else None
        )

    def process(self, frame, frame_time):
        """frame: float32 numpy array, one VAD frame, in [-1, 1].
        Returns (possibly-cleaned frame, applied: bool).
        """
        if not self.enabled:
            return frame, False

        self.reference.ingest_new()

        reference = self.reference.get_aligned(
            frame_time,
            len(frame),
        )

        if reference is None:
            self.skipped_count += 1
            return frame, False

        reference_rms = float(
            np.sqrt(
                np.mean(
                    reference.astype(np.float64) ** 2
                )
            )
        )

        if reference_rms < MIN_REFERENCE_RMS:
            self.skipped_count += 1
            return frame, False

        mic_i16 = np.clip(
            frame * 32767.0, -32768, 32767
        ).astype(np.int16)

        try:
            cancelled = self._aec.cancel_echo(
                mic_i16.tolist(),
                reference.tolist(),
            )
        except Exception as error:
            print("[AEC ERROR]", error)
            return frame, False

        self.applied_count += 1

        clean = (
            np.asarray(cancelled, dtype=np.float32)
            / 32768.0
        )
        return clean, True
