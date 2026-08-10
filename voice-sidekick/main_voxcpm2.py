import os
import subprocess
import sys

from dotenv import load_dotenv


load_dotenv()

from llm.llm_worker import chat


def start_worker(worker_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    return subprocess.Popen(
        [sys.executable, worker_path],
        env=env,
    )


def stop_process(process):
    if process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main():
    print("[MAIN] VoxCPM2 audio worker başlatılıyor...")

    audio_process = start_worker(
        "tts/voxcpm2_audio_player_worker.py"
    )

    print("[MAIN] VoxCPM2 TTS worker başlatılıyor...")

    tts_process = start_worker(
        "tts/voxcpm2_tts_worker.py"
    )

    print("[MAIN] VoxCPM2 sistemi hazır.")

    try:
        while True:
            text = input("Sen: ")
            chat(text)

    except (KeyboardInterrupt, EOFError):
        print("\n[MAIN] Sistem kapatılıyor...")

    finally:
        stop_process(tts_process)
        stop_process(audio_process)


if __name__ == "__main__":
    main()
