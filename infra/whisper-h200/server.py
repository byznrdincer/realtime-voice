import os
import shutil
import tempfile
import threading
import time

from fastapi import FastAPI, File, UploadFile
from faster_whisper import WhisperModel


MODEL_NAME = os.getenv(
    "WHISPER_MODEL",
    "turbo",
)
COMPUTE_TYPE = os.getenv(
    "WHISPER_COMPUTE_TYPE",
    "float16",
)

app = FastAPI()
model_lock = threading.Lock()

print(
    f"[WHISPER] model={MODEL_NAME} "
    f"| compute={COMPUTE_TYPE}",
    flush=True,
)

model = WhisperModel(
    MODEL_NAME,
    device="cuda",
    compute_type=COMPUTE_TYPE,
)

print("[WHISPER] Model hazır", flush=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": "cuda",
        "compute_type": COMPUTE_TYPE,
    }


@app.post("/transcribe")
def transcribe(
    file: UploadFile = File(...),
):
    suffix = (
        os.path.splitext(
            file.filename or ""
        )[1]
        or ".wav"
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temporary:
        path = temporary.name
        shutil.copyfileobj(
            file.file,
            temporary,
        )

    started = time.perf_counter()

    try:
        with model_lock:
            segments, _ = model.transcribe(
                path,
                language="tr",
                task="transcribe",
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 300,
                },
            )

            text = " ".join(
                segment.text.strip()
                for segment in segments
                if segment.text.strip()
            ).strip()

        latency = (
            time.perf_counter()
            - started
        )

        print(
            f"[TRANSCRIBE] "
            f"{latency:.3f}s "
            f"| {text!r}",
            flush=True,
        )

        return {
            "text": text,
            "latency_seconds": round(
                latency,
                3,
            ),
        }

    finally:
        try:
            os.remove(path)
        except OSError:
            pass
