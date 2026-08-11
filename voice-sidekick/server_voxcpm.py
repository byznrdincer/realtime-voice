import threading
import time

import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from voxcpm import VoxCPM


REFERENCE_AUDIO = "/data/sidekick_female_voice_v1.wav"

REFERENCE_TEXT = (
    "Merhaba Beyzanur. Bugün sana açık, sakin ve anlaşılır bir şekilde "
    "yardımcı olacağım. Sorularını dikkatle dinleyerek doğru bilgileri "
    "paylaşacak, gerektiğinde önemli noktaları adım adım ve doğal bir "
    "tempoda açıklayacağım."
)

SAMPLE_RATE = 48000

app = FastAPI()
generation_lock = threading.Lock()


class TTSRequest(BaseModel):
    text: str
    use_reference: bool = True


print("[MODEL] VoxCPM2 yükleniyor...")

model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    device="cuda",
    load_denoiser=False,
)

print("[MODEL] Hazır")
print(f"[REFERENCE] {REFERENCE_AUDIO}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "VoxCPM2",
        "reference": REFERENCE_AUDIO,
        "sample_rate": SAMPLE_RATE,
    }


@app.post("/tts/stream")
def tts_stream(request: TTSRequest):
    text = request.text.strip()

    if not text:
        return {"error": "Metin boş olamaz"}

    def generate_audio():
        with generation_lock:
            start = time.perf_counter()
            first_chunk_time = None
            audio_samples = 0
            chunk_count = 0

            for chunk in model.generate_streaming(
                text=text,
                prompt_wav_path=REFERENCE_AUDIO if request.use_reference else None,
                prompt_text=REFERENCE_TEXT if request.use_reference else None,
                reference_wav_path=REFERENCE_AUDIO if request.use_reference else None,
                retry_badcase=False,
            ):
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                    print(
                        f"[FIRST CHUNK] "
                        f"{first_chunk_time - start:.3f}s"
                    )

                chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
                chunk = np.clip(chunk, -1.0, 1.0)

                pcm16 = (chunk * 32767).astype("<i2")

                audio_samples += len(chunk)
                chunk_count += 1

                yield pcm16.tobytes()

            generation_time = time.perf_counter() - start
            audio_duration = audio_samples / SAMPLE_RATE
            rtf = generation_time / audio_duration if audio_duration else 0

            print(
                f"[DONE] chunks={chunk_count} "
                f"| generation={generation_time:.3f}s "
                f"| audio={audio_duration:.3f}s "
                f"| RTF={rtf:.3f}"
            )

    return StreamingResponse(
        generate_audio(),
        media_type="application/octet-stream",
        headers={
            "X-Audio-Sample-Rate": str(SAMPLE_RATE),
            "X-Audio-Format": "pcm_s16le",
            "X-Audio-Channels": "1",
        },
    )
