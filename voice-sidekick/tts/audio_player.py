import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd


class ContinuousAudioPlayer:

    def __init__(
        self,
        sample_rate=48000,
        prebuffer_seconds=3.0,
    ):
        self.sample_rate = sample_rate

        self.prebuffer_samples = int(
            sample_rate * prebuffer_seconds
        )

        self.chunks = deque()

        self.lock = threading.Lock()

        self.current_chunk = None
        self.current_position = 0
        self.queued_samples = 0

        self.stream = None
        self.started = False

        self.underrun_count = 0

    def start(self):

        if self.stream is not None:
            return

        def callback(
            outdata,
            frames,
            time_info,
            status,
        ):

            if status:
                print(
                    "[AUDIO STATUS]",
                    status
                )

            output = np.zeros(
                frames,
                dtype=np.float32
            )

            filled = 0

            while filled < frames:

                with self.lock:

                    if (
                        self.current_chunk is None
                        or
                        self.current_position >= len(self.current_chunk)
                    ):

                        if self.chunks:

                            self.current_chunk = (
                                self.chunks.popleft()
                            )

                            self.current_position = 0

                        else:

                            self.underrun_count += 1

                            # Boştayken terminal girişini bozacak log yazdırma.
                            break

                    available = (
                        len(self.current_chunk)
                        - self.current_position
                    )

                    needed = frames - filled

                    take = min(
                        available,
                        needed
                    )

                    output[
                        filled:filled + take
                    ] = self.current_chunk[
                        self.current_position:
                        self.current_position + take
                    ]

                    self.current_position += take
                    self.queued_samples -= take

                filled += take

            outdata[:, 0] = output

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
            blocksize=4096,
        )

    def add_audio(self, pcm_bytes):

        if not pcm_bytes:
            return

        usable_length = len(pcm_bytes) - (
            len(pcm_bytes) % 2
        )

        if usable_length <= 0:
            return

        audio = (
            np.frombuffer(
                pcm_bytes[:usable_length],
                dtype="<i2",
            )
            .astype(np.float32)
            / 32768.0
        )

        with self.lock:

            self.chunks.append(audio)

            self.queued_samples += len(audio)

            buffered_samples = self.queued_samples

            buffered_seconds = (
                buffered_samples / self.sample_rate
            )

            print(

                "[PLAYER QUEUE]",

                len(self.chunks),

                "| samples=",

                buffered_samples,

                "| seconds=",

                round(buffered_seconds, 2)

            )

        if (
            not self.started
            and buffered_samples >= self.prebuffer_samples
        ):

            self.stream.start()

            self.started = True

            print(

                "[AUDIO START]",

                round(
                    buffered_samples / self.sample_rate,
                    2
                ),

                "seconds"

            )

    def clear(self):
        """Barge-in sırasında bekleyen sesi anında temizler."""

        with self.lock:
            discarded_samples = self.queued_samples

            self.chunks.clear()
            self.current_chunk = None
            self.current_position = 0
            self.queued_samples = 0

        discarded_seconds = (
            discarded_samples / self.sample_rate
        )

        print(
            f"[AUDIO CLEAR] "
            f"silinen={discarded_seconds:.2f}s"
        )

        return discarded_seconds

    def get_buffered_seconds(self):

        with self.lock:
            return (
                self.queued_samples
                / self.sample_rate
            )

    def stop(self):

        while True:

            with self.lock:
                remaining = self.queued_samples

            if remaining <= 0:
                break

            time.sleep(0.05)

        if self.stream is not None:

            if self.started:
                self.stream.stop()

            self.stream.close()

        self.stream = None
        self.started = False