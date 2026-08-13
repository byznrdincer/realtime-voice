import os
import select
import subprocess
import sys

import redis
from dotenv import load_dotenv


load_dotenv()

from llm.llm_worker import chat
from queue_service.redis_client import (
    audio_redis,
    TEXT_QUEUE,
    VOXCPM2_AUDIO_QUEUE,
    PLAYBACK_BUFFER_SECONDS_KEY,
)


class WorkerStoppedError(RuntimeError):
    pass


def clear_voxcpm2_state(stage):
    try:
        text_items = audio_redis.llen(
            TEXT_QUEUE
        )
        audio_items = audio_redis.llen(
            VOXCPM2_AUDIO_QUEUE
        )

        deleted_keys = audio_redis.delete(
            TEXT_QUEUE,
            VOXCPM2_AUDIO_QUEUE,
            PLAYBACK_BUFFER_SECONDS_KEY,
        )

        print(
            f"[MAIN] Redis temizlendi "
            f"| aşama={stage} "
            f"| eski_text={text_items} "
            f"| eski_audio={audio_items} "
            f"| silinen_anahtar={deleted_keys}"
        )

        return True

    except redis.exceptions.RedisError as error:
        print(
            f"[MAIN] Redis temizlenemedi "
            f"| aşama={stage} "
            f"| hata={error}"
        )

        return False


def start_worker(worker_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    return subprocess.Popen(
        [sys.executable, worker_path],
        env=env,
    )


def stop_process(process):
    if process is None:
        return

    if process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def check_workers(workers):
    for worker_name, process in workers:
        exit_code = process.poll()

        if exit_code is not None:
            raise WorkerStoppedError(
                f"{worker_name} worker kapandı "
                f"| exit_code={exit_code}"
            )


def read_user_input(workers):
    print("Sen: ", end="", flush=True)

    while True:
        check_workers(workers)

        readable, _, _ = select.select(
            [sys.stdin],
            [],
            [],
            0.5,
        )

        if not readable:
            continue

        line = sys.stdin.readline()

        if line == "":
            raise EOFError

        return line.rstrip("\n")


def main():
    if not clear_voxcpm2_state("başlangıç"):
        print(
            "[MAIN] Redis olmadan VoxCPM2 "
            "sistemi başlatılamaz."
        )
        return

    audio_process = None
    tts_process = None

    try:
        print(
            "[MAIN] VoxCPM2 audio worker "
            "başlatılıyor..."
        )

        audio_process = start_worker(
            "tts/voxcpm2_audio_player_worker.py"
        )

        print(
            "[MAIN] VoxCPM2 TTS worker "
            "başlatılıyor..."
        )

        tts_process = start_worker(
            "tts/voxcpm2_tts_worker.py"
        )

        workers = [
            ("audio", audio_process),
            ("tts", tts_process),
        ]

        print("[MAIN] VoxCPM2 sistemi hazır.")

        while True:
            text = read_user_input(workers)

            if not text.strip():
                continue

            chat(text)
            check_workers(workers)

    except WorkerStoppedError as error:
        print(f"\n[MAIN ERROR] {error}")

    except (KeyboardInterrupt, EOFError):
        print("\n[MAIN] Sistem kapatılıyor...")

    finally:
        stop_process(tts_process)
        stop_process(audio_process)
        clear_voxcpm2_state("kapanış")


if __name__ == "__main__":
    main()
