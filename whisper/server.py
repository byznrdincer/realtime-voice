import os
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile
import whisper


app = FastAPI()

# Daha önce test ettiğimiz model.
MODEL_NAME = "small"

print(f"[whisper] loading model: {MODEL_NAME}")
model = whisper.load_model(MODEL_NAME)
print("[whisper] model ready")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:
        temp_path = temp_file.name
        shutil.copyfileobj(file.file, temp_file)

    try:
        result = model.transcribe(
            temp_path,
            language="tr",
            fp16=False,
        )

        return {
            "text": result.get("text", "").strip()
        }

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass