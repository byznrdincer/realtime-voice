import threading
import time
from typing import Literal

import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from voxcpm import VoxCPM


SIDEKICK_REFERENCE_AUDIO = (
    "/data/sidekick_female_voice_v1.wav"
)

SIDEKICK_REFERENCE_TEXT = (
    "Merhaba Beyzanur. Bugün sana açık, sakin ve anlaşılır bir şekilde "
    "yardımcı olacağım. Sorularını dikkatle dinleyerek doğru bilgileri "
    "paylaşacak, gerektiğinde önemli noktaları adım adım ve doğal bir "
    "tempoda açıklayacağım."
)

BEYZANUR_REFERENCE_AUDIO = (
    "/data/beyzanur_voice_reference_11s.wav"
)

BEYZANUR_REFERENCE_TEXT = (
    "Merhaba, ben Beyzanur. Bugün kendi sesimi kullanarak doğal ve "
    "anlaşılır bir konuşma kaydı oluşturuyorum. Bu kayıt, yapay zekâ "
    "destekli bir ses sisteminin konuşma biçimimi"
)

VOICE_PROFILES = {
    "sidekick": {
        "audio": SIDEKICK_REFERENCE_AUDIO,
        "text": SIDEKICK_REFERENCE_TEXT,
    },
    "beyzanur": {
        "audio": BEYZANUR_REFERENCE_AUDIO,
        "text": BEYZANUR_REFERENCE_TEXT,
    },
}

SAMPLE_RATE = 48000

app = FastAPI()
generation_lock = threading.Lock()


class TTSRequest(BaseModel):
    text: str
    use_reference: bool = True
    voice_profile: Literal[
        "sidekick",
        "beyzanur",
    ] = "sidekick"


print("[MODEL] VoxCPM2 yükleniyor...")

model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    device="cuda",
    load_denoiser=False,
)

print("[MODEL] Hazır")
print(f"[DEFAULT REFERENCE] {SIDEKICK_REFERENCE_AUDIO}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "VoxCPM2",
        "default_voice_profile": "sidekick",
        "voice_profiles": list(VOICE_PROFILES),
        "sample_rate": SAMPLE_RATE,
    }


@app.post("/tts/stream")
def tts_stream(request: TTSRequest):
    text = request.text.strip()

    if not text:
        return {"error": "Metin boş olamaz"}

    profile = VOICE_PROFILES[
        request.voice_profile
    ]

    reference_audio = (
        profile["audio"]
        if request.use_reference
        else None
    )

    reference_text = (
        profile["text"]
        if request.use_reference
        else None
    )

    def generate_audio():
        with generation_lock:
            start = time.perf_counter()
            first_chunk_time = None
            audio_samples = 0
            chunk_count = 0

            print(
                f"[TTS START] "
                f"profile={request.voice_profile} "
                f"| words={len(text.split())}"
            )

            for chunk in model.generate_streaming(
                text=text,
                prompt_wav_path=reference_audio,
                prompt_text=reference_text,
                reference_wav_path=reference_audio,
                retry_badcase=False,
            ):
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()

                    print(
                        f"[FIRST CHUNK] "
                        f"profile={request.voice_profile} "
                        f"| {first_chunk_time - start:.3f}s"
                    )

                chunk = np.asarray(
                    chunk,
                    dtype=np.float32,
                ).reshape(-1)

                chunk = np.clip(
                    chunk,
                    -1.0,
                    1.0,
                )

                pcm16 = (
                    chunk * 32767
                ).astype("<i2")

                audio_samples += len(chunk)
                chunk_count += 1

                yield pcm16.tobytes()

            generation_time = (
                time.perf_counter() - start
            )

            audio_duration = (
                audio_samples / SAMPLE_RATE
            )

            rtf = (
                generation_time / audio_duration
                if audio_duration
                else 0
            )

            print(
                f"[DONE] "
                f"profile={request.voice_profile} "
                f"| chunks={chunk_count} "
                f"| generation={generation_time:.3f}s "
                f"| audio={audio_duration:.3f}s "
                f"| RTF={rtf:.3f}"
            )

    return StreamingResponse(
        generate_audio(),
        media_type="application/octet-stream",
        headers={
            "X-Audio-Sample-Rate": str(
                SAMPLE_RATE
            ),
            "X-Audio-Format": "pcm_s16le",
            "X-Audio-Channels": "1",
            "X-Voice-Profile": request.voice_profile,
        },
    )
