import io
import os
import queue
import re
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
from queue_service.events import publish_event
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
    TTS_ACTIVE_KEY,
)
from stt.aec import AcousticEchoCanceller
from stt.echo_guard import (
    EchoGuard,
    is_echo_transcript,
    recent_spoken_texts,
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
VAD_END_MS = int(
    os.getenv("VAD_END_MS", "700")
)
MIN_AUDIO_SECONDS = 0.5

STT_PARTIAL_ENABLED = (
    os.getenv(
        "STT_PARTIAL_ENABLED",
        "false",
    )
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)

STT_ENDPOINT_PRECOMPUTE_ENABLED = (
    os.getenv(
        "STT_ENDPOINT_PRECOMPUTE_ENABLED",
        "true",
    )
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)

STT_ENDPOINT_SILENCE_MS = int(
    os.getenv(
        "STT_ENDPOINT_SILENCE_MS",
        "150",
    )
)

PARTIAL_INTERVAL_SECONDS = float(
    os.getenv(
        "STT_PARTIAL_INTERVAL_SECONDS",
        "0.75",
    )
)
FINAL_OVERLAP_SECONDS = float(
    os.getenv(
        "STT_FINAL_OVERLAP_SECONDS",
        "1.25",
    )
)
PARTIAL_WAIT_SECONDS = float(
    os.getenv(
        "STT_PARTIAL_WAIT_SECONDS",
        "2.0",
    )
)

ECHO_GUARD_ENABLED = (
    os.getenv("VAD_ECHO_GUARD", "true")
    .strip()
    .lower()
    not in {"0", "false", "no", "off"}
)

PLAYBACK_VAD_THRESHOLD = float(
    os.getenv(
        "VAD_BARGE_IN_RMS_THRESHOLD",
        os.getenv(
            "VAD_PLAYBACK_RMS_THRESHOLD",
            "0.10",
        ),
    )
)

PLAYBACK_ACTIVE_SECONDS = float(
    os.getenv(
        "VAD_PLAYBACK_ACTIVE_SECONDS",
        "0.0",
    )
)

ECHO_HANGOVER_SECONDS = float(
    os.getenv(
        "VAD_ECHO_HANGOVER_SECONDS",
        "0.7",
    )
)

# Minimum sustained duration a sub-barge-in-threshold voice must hold
# during TTS playback before it is trusted as real user speech instead
# of TTS bleed. See the candidate/confirm barge-in logic in main().
# Must stay below VAD_END_MS (default 700ms): a candidate is watched
# using the elevated echo threshold, so it needs to confirm before
# VAD's own silence timeout would otherwise kill it as "end".
VAD_BARGE_IN_CONFIRM_SECONDS = float(
    os.getenv(
        "VAD_BARGE_IN_CONFIRM_MS",
        "350",
    )
) / 1000.0

# Real acoustic echo cancellation (adaptive filter against the actual
# TTS output), separate from and in addition to the RMS-threshold
# echo guard above. Requires an acoustic + device delay estimate
# (AEC_REFERENCE_DELAY_MS): the adaptive filter can absorb some
# misalignment on its own (bounded by AEC_FILTER_LENGTH_SAMPLES /
# SAMPLE_RATE of "reach"), but the base delay still needs calibrating
# against real Mac hardware to get good cancellation.
AEC_ENABLED = (
    os.getenv("AEC_ENABLED", "true")
    .strip()
    .lower()
    not in {"0", "false", "no", "off"}
)
AEC_REFERENCE_DELAY_MS = float(
    os.getenv("AEC_REFERENCE_DELAY_MS", "180")
)
AEC_FILTER_LENGTH_SAMPLES = int(
    os.getenv(
        "AEC_FILTER_LENGTH_SAMPLES", "4096"
    )
)


def get_playback_seconds():
    try:
        raw = audio_redis.get(
            PLAYBACK_BUFFER_SECONDS_KEY
        )
        playback_seconds = (
            float(raw)
            if raw is not None
            else 0.0
        )

        if audio_redis.get(TTS_ACTIVE_KEY):
            playback_seconds = max(
                playback_seconds,
                0.5,
            )

        return playback_seconds

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


def transcribe(audio, stage="FINAL"):
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
        f"[WHISPER {stage}] "
        f"{time.perf_counter() - started:.3f}s "
        f"| {text!r}"
    )

    return text


def normalize_word(word):
    return re.sub(
        r"[^\wçğıöşüÇĞİÖŞÜ]+",
        "",
        word,
    ).casefold()


def merge_transcripts(prefix, tail):
    prefix_words = prefix.split()
    tail_words = tail.split()

    prefix_normalized = [
        normalize_word(word)
        for word in prefix_words
    ]
    tail_normalized = [
        normalize_word(word)
        for word in tail_words
    ]

    maximum = min(
        16,
        len(prefix_words),
        len(tail_words),
    )

    for size in range(maximum, 1, -1):
        if (
            prefix_normalized[-size:]
            == tail_normalized[:size]
        ):
            return " ".join(
                prefix_words
                + tail_words[size:]
            ).strip()

    return None


def classify_speech_onset(
    is_hot,
    level,
    threshold,
    candidate_elapsed,
    confirm_seconds,
):
    """Decide what to do with a detected (or continuing) speech onset.

    Returns "confirm" (treat as a real user turn now), or "candidate"
    (not yet trustworthy while TTS is playing; keep watching).

    Idle (not hot) speech always confirms immediately, unchanged from
    prior behavior. During hot/playback state, a level that already
    clears the echo-defense threshold confirms immediately (fast
    path, same as before). A quieter level is trusted only once it
    has sustained for `confirm_seconds` without dropping to silence,
    to reduce the chance a brief TTS bleed blip counts as barge-in.
    """
    if not is_hot:
        return "confirm"

    if level >= threshold:
        return "confirm"

    if candidate_elapsed >= confirm_seconds:
        return "confirm"

    return "candidate"


def remaining_audio_is_silence(
    audio,
    threshold=VAD_THRESHOLD,
):
    audio = np.asarray(
        audio,
        dtype=np.float32,
    ).reshape(-1)

    frame_samples = max(
        1,
        int(SAMPLE_RATE * 0.03),
    )

    for start in range(
        0,
        len(audio),
        frame_samples,
    ):
        frame = audio[
            start:start + frame_samples
        ]

        if (
            frame.size
            and EnergyVAD.rms(frame)
            >= threshold
        ):
            return False

    return True


class PartialTranscriber:
    def __init__(self):
        self.lock = threading.Lock()
        self.version = None
        self.text = ""
        self.sample_count = 0
        self.thread = None

    def begin(self, version):
        with self.lock:
            self.version = version
            self.text = ""
            self.sample_count = 0

    def schedule(self, version, audio):
        with self.lock:
            if self.version != version:
                return False

            if (
                self.thread is not None
                and self.thread.is_alive()
            ):
                return False

            snapshot = np.asarray(
                audio,
                dtype=np.float32,
            ).copy()

            self.thread = threading.Thread(
                target=self._run,
                args=(version, snapshot),
                daemon=True,
            )
            self.thread.start()
            return True

    def _run(self, version, audio):
        try:
            text = transcribe(
                audio,
                stage="PARTIAL",
            )

            if not text:
                return

            if (
                get_interrupt_version()
                != version
            ):
                return

            with self.lock:
                if self.version != version:
                    return

                self.text = text
                self.sample_count = len(audio)

            print(
                "[WHISPER PARTIAL READY] "
                f"turn_id={version} "
                f"| covered="
                f"{len(audio) / SAMPLE_RATE:.2f}s"
            )
            publish_event(
                "partial",
                text=text,
                turn_id=version,
            )

        except Exception as error:
            print(
                "[WHISPER PARTIAL ERROR]",
                error,
            )

    def wait(self, version):
        with self.lock:
            if self.version != version:
                return
            thread = self.thread

        if (
            thread is not None
            and thread.is_alive()
        ):
            thread.join(
                timeout=PARTIAL_WAIT_SECONDS
            )

    def result(self, version):
        with self.lock:
            if self.version != version:
                return "", 0

            return self.text, self.sample_count

    def clear(self, version):
        with self.lock:
            if self.version == version:
                self.version = None
                self.text = ""
                self.sample_count = 0


def finalize_transcript(
    version,
    audio,
    partial_transcriber,
):
    partial_transcriber.wait(version)

    partial_text, covered_samples = (
        partial_transcriber.result(version)
    )

    if not partial_text or covered_samples <= 0:
        return transcribe(
            audio,
            stage="FINAL FULL",
        )

    covered_samples = min(
        covered_samples,
        len(audio),
    )

    remaining_audio = audio[
        covered_samples:
    ]

    if remaining_audio_is_silence(
        remaining_audio
    ):
        print(
            "[WHISPER ENDPOINT REUSE] "
            f"turn_id={version} "
            f"| covered="
            f"{covered_samples / SAMPLE_RATE:.2f}s"
        )
        return partial_text

    overlap_samples = int(
        FINAL_OVERLAP_SECONDS
        * SAMPLE_RATE
    )
    tail_start = max(
        0,
        min(covered_samples, len(audio))
        - overlap_samples,
    )
    tail_audio = audio[tail_start:]
    tail_text = transcribe(
        tail_audio,
        stage="FINAL TAIL",
    )

    if tail_start == 0:
        return tail_text

    merged = merge_transcripts(
        partial_text,
        tail_text,
    )

    if merged is not None:
        print(
            "[WHISPER REUSE] "
            f"turn_id={version} "
            f"| reused="
            f"{covered_samples / SAMPLE_RATE:.2f}s"
        )
        return merged

    print(
        "[WHISPER REUSE FALLBACK] "
        "Örtüşme bulunamadı; tam doğrulama."
    )
    return transcribe(
        audio,
        stage="FINAL FALLBACK",
    )


def transcription_loop(
    recordings,
    partial_transcriber,
    echo_guard,
):
    while True:
        version, audio = recordings.get()

        try:
            duration = (
                len(audio) / SAMPLE_RATE
            )

            if duration < MIN_AUDIO_SECONDS:
                continue

            started = time.perf_counter()

            text = finalize_transcript(
                version,
                audio,
                partial_transcriber,
            )

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

            if is_echo_transcript(
                text,
                recent_spoken_texts(),
                playback_hot=echo_guard.is_hot(),
            ):
                print(
                    "[ECHO FILTER] "
                    f"kendi sesi atlandı | {text!r}"
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
                "[WHISPER FINAL] "
                f"{time.perf_counter() - started:.3f}s "
                f"| {text!r}"
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
            partial_transcriber.clear(
                version
            )
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
        start_threshold=VAD_THRESHOLD,
        end_ms=VAD_END_MS,
    )

    audio_frames = queue.Queue(
        maxsize=300
    )
    recordings = queue.Queue()
    partial_transcriber = (
        PartialTranscriber()
    )
    echo_guard = EchoGuard(
        hangover_seconds=ECHO_HANGOVER_SECONDS,
        active_seconds=PLAYBACK_ACTIVE_SECONDS,
        barge_in_threshold=PLAYBACK_VAD_THRESHOLD,
        normal_threshold=VAD_THRESHOLD,
        enabled=ECHO_GUARD_ENABLED,
    )
    aec = (
        AcousticEchoCanceller(
            sample_rate=SAMPLE_RATE,
            frame_len=vad.frame_samples,
            filter_length=AEC_FILTER_LENGTH_SAMPLES,
            delay_ms=AEC_REFERENCE_DELAY_MS,
        )
        if AEC_ENABLED
        else None
    )

    transcription_thread = threading.Thread(
        target=transcription_loop,
        args=(
            recordings,
            partial_transcriber,
            echo_guard,
        ),
        daemon=True,
    )
    transcription_thread.start()

    active_version = None
    echo_guard_active = False
    # A "candidate" is a VAD-detected onset that happened during hot
    # (TTS playback) state below the barge-in confirm threshold. It is
    # not yet treated as a real turn (no interrupt/version bump) until
    # it either gets loud enough or sustains long enough to be trusted
    # as real user speech rather than TTS bleed-through.
    pending_candidate = False
    candidate_started_at = 0.0
    candidate_peak_level = 0.0
    partial_interval_samples = max(
        1,
        int(
            PARTIAL_INTERVAL_SECONDS
            * SAMPLE_RATE
        ),
    )
    next_partial_samples = (
        partial_interval_samples
    )

    endpoint_silence_frames = max(
        1,
        (
            STT_ENDPOINT_SILENCE_MS
            + vad.frame_ms
            - 1
        )
        // vad.frame_ms,
    )
    endpoint_candidate_scheduled = False

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
                (
                    time.time(),
                    indata[:, 0].copy(),
                )
            )
        except queue.Full:
            print("[MIC] Frame kuyruğu doldu.")

    print(
        f"[VAD] Hazır "
        f"| sample_rate={SAMPLE_RATE} "
        f"| threshold={VAD_THRESHOLD} "
        f"| end_ms={VAD_END_MS}"
    )
    print(
        "[WHISPER STREAMING] "
        f"partial={PARTIAL_INTERVAL_SECONDS:.2f}s "
        f"| overlap={FINAL_OVERLAP_SECONDS:.2f}s"
    )

    print(
        "[WHISPER ENDPOINT PRECOMPUTE] "
        f"{'aktif' if STT_ENDPOINT_PRECOMPUTE_ENABLED else 'pasif'} "
        f"| silence={STT_ENDPOINT_SILENCE_MS}ms"
    )
    print(
        "[VAD BARGE-IN CONFIG] "
        f"barge_in_threshold={PLAYBACK_VAD_THRESHOLD:.4f} "
        f"| confirm_after={VAD_BARGE_IN_CONFIRM_SECONDS:.2f}s"
    )
    if aec is not None and aec.enabled:
        print(
            "[AEC] aktif "
            f"| delay={AEC_REFERENCE_DELAY_MS:.0f}ms "
            f"| filter_length={AEC_FILTER_LENGTH_SAMPLES} "
            f"samples"
        )
    elif AEC_ENABLED:
        print(
            "[AEC] pasif | pyaec kütüphanesi "
            "yüklenemedi, mikrofon işlenmemiş "
            "geçecek"
        )
    else:
        print("[AEC] pasif (AEC_ENABLED=false)")

    try:
        stream_context = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=vad.frame_samples,
            callback=callback,
        )
    except Exception as error:
        print("[WHISPER] Mikrofon açılamadı", error)
        raise

    aec_active_logged = False

    with stream_context:
        while True:
            frame_time, frame = (
                audio_frames.get()
            )

            if aec is not None:
                frame, aec_applied = (
                    aec.process(
                        frame, frame_time
                    )
                )

                if aec_applied != aec_active_logged:
                    aec_active_logged = aec_applied
                    print(
                        "[AEC] "
                        f"{'uygulanıyor' if aec_applied else 'referans yok, atlanıyor'} "
                        f"| applied={aec.applied_count} "
                        f"skipped={aec.skipped_count}"
                    )

            playback_seconds = (
                get_playback_seconds()
            )
            echo_guard.observe_playback(
                playback_seconds
            )

            if active_version is not None:
                # A turn is already confirmed as real user speech;
                # keep endpoint/silence detection at normal
                # sensitivity so the elevated echo-defense threshold
                # can't clip the user's own (possibly moderate
                # volume) speech mid-turn.
                new_threshold = VAD_THRESHOLD
            else:
                new_threshold = echo_guard.threshold()

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
                is_hot = echo_guard.is_hot()
                onset = classify_speech_onset(
                    is_hot,
                    level,
                    new_threshold,
                    0.0,
                    VAD_BARGE_IN_CONFIRM_SECONDS,
                )

                if onset == "candidate":
                    # Onset is real (crossed the normal VAD gate) but
                    # still under the echo-defense threshold while TTS
                    # is playing. Don't commit yet: watch it for a
                    # short confirm window instead of either accepting
                    # it outright (self-interrupt risk) or discarding
                    # it outright (missed barge-in risk).
                    pending_candidate = True
                    candidate_started_at = (
                        time.monotonic()
                    )
                    candidate_peak_level = level

                    print(
                        "[VAD BARGE-IN] "
                        f"playback=true rms={level:.4f} "
                        f"threshold={new_threshold:.4f} "
                        f"decision=CANDIDATE "
                        f"confirm_after="
                        f"{VAD_BARGE_IN_CONFIRM_SECONDS:.2f}s"
                    )
                    continue

                pending_candidate = False
                active_version = (
                    request_interrupt(
                        "speech-start"
                    )
                )
                partial_transcriber.begin(
                    active_version
                )
                next_partial_samples = (
                    partial_interval_samples
                )
                endpoint_candidate_scheduled = False

                if is_hot:
                    print(
                        "[VAD BARGE-IN] "
                        f"playback=true rms={level:.4f} "
                        f"threshold={new_threshold:.4f} "
                        f"decision=USER_SPEECH "
                        f"confirm=loud elapsed=0.00s"
                    )
                else:
                    print(
                        f"[VAD START] "
                        f"rms={level:.4f}"
                    )
                publish_event(
                    "listening",
                    turn_id=active_version,
                )

            elif (
                event == "speech"
                and pending_candidate
            ):
                candidate_peak_level = max(
                    candidate_peak_level,
                    level,
                )
                elapsed = (
                    time.monotonic()
                    - candidate_started_at
                )
                onset = classify_speech_onset(
                    echo_guard.is_hot(),
                    level,
                    new_threshold,
                    elapsed,
                    VAD_BARGE_IN_CONFIRM_SECONDS,
                )

                if onset == "confirm":
                    confirm_reason = (
                        "loud"
                        if level >= new_threshold
                        else "sustained"
                    )
                    pending_candidate = False
                    active_version = (
                        request_interrupt(
                            "speech-start"
                        )
                    )
                    partial_transcriber.begin(
                        active_version
                    )

                    active_audio = (
                        vad.get_active_audio()
                    )
                    covered_now = (
                        len(active_audio)
                        if active_audio
                        is not None
                        else 0
                    )
                    next_partial_samples = (
                        covered_now
                        + partial_interval_samples
                    )
                    endpoint_candidate_scheduled = False

                    print(
                        "[VAD BARGE-IN] "
                        f"playback=true rms={level:.4f} "
                        f"threshold={new_threshold:.4f} "
                        f"decision=USER_SPEECH "
                        f"confirm={confirm_reason} "
                        f"elapsed={elapsed:.2f}s"
                    )
                    publish_event(
                        "listening",
                        turn_id=active_version,
                    )

            elif (
                event == "speech"
                and active_version is not None
                and STT_PARTIAL_ENABLED
                and level >= vad.threshold
            ):
                active_audio = (
                    vad.get_active_audio()
                )

                if (
                    active_audio is not None
                    and len(active_audio)
                    >= next_partial_samples
                ):
                    scheduled = (
                        partial_transcriber.schedule(
                            active_version,
                            active_audio,
                        )
                    )

                    if scheduled:
                        next_partial_samples = (
                            len(active_audio)
                            + partial_interval_samples
                        )

            elif (
                event == "speech"
                and active_version is not None
            ):
                if level >= vad.threshold:
                    endpoint_candidate_scheduled = False

                elif (
                    STT_ENDPOINT_PRECOMPUTE_ENABLED
                    and not endpoint_candidate_scheduled
                    and vad.silence_run
                    >= endpoint_silence_frames
                ):
                    active_audio = (
                        vad.get_active_audio()
                    )

                    if active_audio is not None:
                        scheduled = (
                            partial_transcriber.schedule(
                                active_version,
                                active_audio,
                            )
                        )

                        if scheduled:
                            endpoint_candidate_scheduled = True

                            print(
                                "[WHISPER ENDPOINT START] "
                                f"turn_id={active_version} "
                                f"| silence="
                                f"{vad.silence_run * vad.frame_ms}ms"
                            )

            elif event == "end":
                duration = (
                    len(audio) / SAMPLE_RATE
                )

                if pending_candidate:
                    # Never confirmed: VAD's own silence timeout
                    # fired first, so this was a brief blip (echo or
                    # noise), not sustained/loud enough user speech.
                    pending_candidate = False
                    print(
                        "[ECHO GUARD] "
                        f"playback=true "
                        f"rms={candidate_peak_level:.4f} "
                        f"threshold={new_threshold:.4f} "
                        f"decision=IGNORE reason=timeout "
                        f"audio={duration:.2f}s"
                    )
                else:
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
