import io
import os
import queue
import threading
import time
import wave

import numpy as np
import redis
import requests
import sounddevice as sd
from dotenv import load_dotenv

from queue_service.messages import (
    encode_text_message,
)
from queue_service.interrupt import (
    get_interrupt_version,
    request_interrupt,
)
from queue_service.worker_signals import (
    install_clean_shutdown,
)
from queue_service.redis_client import (
    audio_redis,
    text_redis,
    USER_INPUT_QUEUE,
    PLAYBACK_BUFFER_SECONDS_KEY,
)
from stt.vad import EnergyVAD


load_dotenv()

WHISPER_URL = os.getenv(
    "WHISPER_URL",
    "http://127.0.0.1:8002",
).rstrip("/")

SAMPLE_RATE = int(
    os.getenv("MIC_SAMPLE_RATE", "48000")
)
VAD_THRESHOLD = float(
    os.getenv("VAD_RMS_THRESHOLD", "0.015")
)
MIN_AUDIO_SECONDS = 0.5

ECHO_GUARD_ENABLED = (
    os.getenv("VAD_ECHO_GUARD", "true")
    .strip()
    .lower()
    not in {"0", "false", "no", "off"}
)

PLAYBACK_VAD_THRESHOLD = float(
    os.getenv(
        "VAD_PLAYBACK_RMS_THRESHOLD",
        "0.035",
    )
)

PLAYBACK_ACTIVE_SECONDS = float(
    os.getenv(
        "VAD_PLAYBACK_ACTIVE_SECONDS",
        "0.05",
    )
)


def choose_vad_threshold(
    playback_seconds,
    normal_threshold=VAD_THRESHOLD,
    playback_threshold=PLAYBACK_VAD_THRESHOLD,
    enabled=ECHO_GUARD_ENABLED,
):
    if (
        enabled
        and playback_seconds
        > PLAYBACK_ACTIVE_SECONDS
    ):
        return max(
            normal_threshold,
            playback_threshold,
        )

    return normal_threshold


def get_playback_seconds():
    try:
        raw = audio_redis.get(
            PLAYBACK_BUFFER_SECONDS_KEY
        )

        return (
            float(raw)
            if raw is not None
            else 0.0
        )

    except (
        redis.exceptions.RedisError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "[ECHO GUARD ERROR]",
            error,
        )
        return 0.0


def audio_to_wav(audio):
    audio = np.asarray(
        audio,
        dtype=np.float32,
    ).reshape(-1)

    pcm16 = (
        np.clip(audio, -1.0, 1.0)
        * 32767
    ).astype("<i2")

    output = io.BytesIO()

    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm16.tobytes())

    return output.getvalue()


def transcribe(audio):
    wav_bytes = audio_to_wav(audio)
    started = time.perf_counter()

    with requests.post(
        f"{WHISPER_URL}/transcribe",
        files={
            "file": (
                "speech.wav",
                wav_bytes,
                "audio/wav",
            )
        },
        timeout=(5, 120),
    ) as response:
        response.raise_for_status()
        payload = response.json()

    text = payload.get(
        "text",
        "",
    ).strip()

    print(
        f"[WHISPER] "
        f"{time.perf_counter() - started:.3f}s "
        f"| {text!r}"
    )

    return text


def transcription_loop(recordings):
    while True:
        version, audio = recordings.get()

        try:
            duration = (
                len(audio) / SAMPLE_RATE
            )

            if duration < MIN_AUDIO_SECONDS:
                continue

            text = transcribe(audio)

            if not text:
                continue

            if (
                get_interrupt_version()
                != version
            ):
                print(
                    "[WHISPER] Eski konuşma "
                    "sonucu atlandı."
                )
                continue

            text_redis.rpush(
                USER_INPUT_QUEUE,
                encode_text_message(
                    version,
                    text,
                ),
            )

            print(
                f"[VOICE INPUT] {text}"
            )

        except Exception as error:
            print(
                "[WHISPER ERROR]",
                error,
            )

        finally:
            recordings.task_done()


def check_whisper():
    response = requests.get(
        f"{WHISPER_URL}/health",
        timeout=5,
    )
    response.raise_for_status()

    print(
        "[WHISPER HEALTH]",
        response.json(),
    )


def main():
    install_clean_shutdown()
    check_whisper()

    vad = EnergyVAD(
        sample_rate=SAMPLE_RATE,
        threshold=VAD_THRESHOLD,
    )

    audio_frames = queue.Queue(
        maxsize=300
    )
    recordings = queue.Queue()

    transcription_thread = threading.Thread(
        target=transcription_loop,
        args=(recordings,),
        daemon=True,
    )
    transcription_thread.start()

    active_version = None
    echo_guard_active = False

    def callback(
        indata,
        frames,
        time_info,
        status,
    ):
        if status:
            print("[MIC STATUS]", status)

        try:
            audio_frames.put_nowait(
                indata[:, 0].copy()
            )
        except queue.Full:
            print("[MIC] Frame kuyruğu doldu.")

    print(
        f"[VAD] Hazır "
        f"| sample_rate={SAMPLE_RATE} "
        f"| threshold={VAD_THRESHOLD}"
    )

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=vad.frame_samples,
        callback=callback,
    ):
        while True:
            frame = audio_frames.get()

            playback_seconds = (
                get_playback_seconds()
            )

            new_threshold = (
                choose_vad_threshold(
                    playback_seconds
                )
            )

            guard_is_active = (
                new_threshold
                > VAD_THRESHOLD
            )

            if (
                guard_is_active
                != echo_guard_active
            ):
                echo_guard_active = (
                    guard_is_active
                )

                print(
                    "[ECHO GUARD] "
                    f"{'aktif' if echo_guard_active else 'pasif'} "
                    f"| threshold={new_threshold:.4f}"
                )

            vad.threshold = new_threshold

            event, audio, level = (
                vad.process(frame)
            )

            if event == "start":
                active_version = (
                    request_interrupt(
                        "speech-start"
                    )
                )

                print(
                    f"[VAD START] "
                    f"rms={level:.4f}"
                )

            elif event == "end":
                duration = (
                    len(audio) / SAMPLE_RATE
                )

                print(
                    f"[VAD END] "
                    f"audio={duration:.2f}s"
                )

                if active_version is not None:
                    recordings.put(
                        (
                            active_version,
                            audio,
                        )
                    )

                active_version = None


if __name__ == "__main__":
    main()
