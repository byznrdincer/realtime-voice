import os
import select
import subprocess
import sys
import threading
import time

import redis
from dotenv import load_dotenv


load_dotenv()

ENABLE_STT = (
    os.getenv("ENABLE_STT", "true")
    .strip()
    .lower()
    not in {"0", "false", "no", "off"}
)

from llm.llm_worker import chat
from queue_service.events import publish_event
from queue_service.interrupt import (
    get_interrupt_version,
    request_interrupt,
)
from ui.server import set_listen_controller, start_ui
from queue_service.messages import (
    decode_text_message,
)
from queue_service.redis_client import (
    audio_redis,
    text_redis,
    TEXT_QUEUE,
    USER_INPUT_QUEUE,
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
        voice_items = audio_redis.llen(
            USER_INPUT_QUEUE
        )

        deleted_keys = audio_redis.delete(
            TEXT_QUEUE,
            USER_INPUT_QUEUE,
            VOXCPM2_AUDIO_QUEUE,
            PLAYBACK_BUFFER_SECONDS_KEY,
        )

        print(
            f"[MAIN] Redis temizlendi "
            f"| aşama={stage} "
            f"| eski_text={text_items} "
            f"| eski_audio={audio_items} "
            f"| eski_voice={voice_items} "
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
    env["PYTHONUNBUFFERED"] = "1"

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
    remaining = []
    stt_died = False

    for worker_name, process in workers:
        exit_code = process.poll()

        if exit_code is None:
            remaining.append((worker_name, process))
            continue

        if worker_name == "stt":
            print(
                "[MAIN] STT worker kapandı "
                f"| exit_code={exit_code}"
            )
            stt_died = True
            continue

        raise WorkerStoppedError(
            f"{worker_name} worker kapandı "
            f"| exit_code={exit_code}"
        )

    if len(remaining) != len(workers):
        workers[:] = remaining

    return stt_died


def read_user_input(workers, on_stt_died=None):
    print("Sen: ", end="", flush=True)

    while True:
        if check_workers(workers) and on_stt_died:
            on_stt_died()

        try:
            voice_payload = text_redis.lpop(
                USER_INPUT_QUEUE
            )
        except redis.exceptions.RedisError as error:
            raise WorkerStoppedError(
                "Sesli giriş kuyruğu okunamadı "
                f"| hata={error}"
            ) from error

        if voice_payload:
            try:
                voice_turn_id, voice_text = (
                    decode_text_message(
                        voice_payload
                    )
                )
            except (TypeError, ValueError) as error:
                print(
                    "[VOICE INPUT ERROR]",
                    error,
                )
                continue

            if (
                voice_turn_id is not None
                and get_interrupt_version()
                != voice_turn_id
            ):
                print(
                    "[VOICE INPUT STALE] "
                    f"turn_id={voice_turn_id}"
                )
                continue

            print(
                f"\n[VOICE USER] {voice_text}"
            )
            publish_event(
                "user",
                text=voice_text,
                turn_id=voice_turn_id,
            )
            return voice_text, voice_turn_id

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

        typed = line.rstrip("\n")
        if typed.strip():
            publish_event("user", text=typed, turn_id=None)
        return typed, None


def main():
    if not clear_voxcpm2_state("başlangıç"):
        print(
            "[MAIN] Redis olmadan VoxCPM2 "
            "sistemi başlatılamaz."
        )
        return

    audio_process = None
    tts_process = None
    stt_process = None
    workers = []
    listening = False
    listen_lock = threading.Lock()

    def start_stt():
        nonlocal stt_process, listening
        if not ENABLE_STT:
            return
        if stt_process is not None and stt_process.poll() is None:
            listening = True
            return
        print("[MAIN] Whisper STT worker başlatılıyor...")
        stt_process = start_worker("stt/whisper_worker.py")
        time.sleep(0.4)
        if stt_process.poll() is not None:
            print(
                "[MAIN] STT başlatılamadı "
                f"| exit_code={stt_process.poll()}"
            )
            stt_process = None
            listening = False
            return
        workers[:] = [
            item for item in workers if item[0] != "stt"
        ]
        workers.append(("stt", stt_process))
        listening = True

    def pause_listen():
        nonlocal stt_process, listening
        process = None
        with listen_lock:
            if not listening:
                return
            process = stt_process
            stt_process = None
            workers[:] = [
                item for item in workers if item[0] != "stt"
            ]
            listening = False
        print("[MAIN] Dinleme duraklatıldı")
        request_interrupt("ui-pause")
        stop_process(process)
        time.sleep(0.3)
        publish_event("listen", active=False)

    def mark_stt_down():
        nonlocal stt_process, listening
        with listen_lock:
            if not listening and stt_process is None:
                return
            stt_process = None
            workers[:] = [
                item for item in workers if item[0] != "stt"
            ]
            listening = False
        publish_event("listen", active=False)

    def resume_listen():
        with listen_lock:
            if listening:
                return
            start_stt()
            active = listening
        if active:
            print("[MAIN] Dinleme yeniden açıldı")
            publish_event("listen", active=True)

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

        workers.append(("audio", audio_process))
        workers.append(("tts", tts_process))
        start_stt()

        set_listen_controller(
            pause_listen,
            resume_listen,
            lambda: listening,
        )
        start_ui()
        publish_event(
            "ready",
            stt=ENABLE_STT,
            listening=listening,
        )

        print(
            "[MAIN] VoxCPM2 sistemi hazır "
            f"| STT={'aktif' if listening else 'pasif'}"
        )

        while True:
            text, turn_id = read_user_input(
                workers,
                on_stt_died=mark_stt_down,
            )

            if not text.strip():
                continue

            if turn_id is None:
                turn_id = request_interrupt(
                    "terminal-input"
                )

            if (
                get_interrupt_version()
                != turn_id
            ):
                print(
                    "[MAIN] Eski kullanıcı turu "
                    f"atlandı | turn_id={turn_id}"
                )
                continue

            chat(text, turn_id)
            if check_workers(workers):
                mark_stt_down()

    except WorkerStoppedError as error:
        print(f"\n[MAIN ERROR] {error}")

    except (KeyboardInterrupt, EOFError):
        print("\n[MAIN] Sistem kapatılıyor...")

    finally:
        stop_process(stt_process)
        stop_process(tts_process)
        stop_process(audio_process)
        clear_voxcpm2_state("kapanış")


if __name__ == "__main__":
    main()
