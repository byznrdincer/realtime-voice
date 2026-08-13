import threading
import time

import redis

from queue_service.messages import (
    decode_audio_message,
)
from queue_service.redis_client import (
    audio_redis,
    VOXCPM2_AUDIO_QUEUE,
    PLAYBACK_BUFFER_SECONDS_KEY,
)
from tts.audio_player import ContinuousAudioPlayer
from queue_service.interrupt import (
    get_interrupt_version,
)
from queue_service.worker_signals import (
    install_clean_shutdown,
)



install_clean_shutdown()


player = ContinuousAudioPlayer(
    sample_rate=48000,
    prebuffer_seconds=0.2,
)

player.start()

# Ses cihazını ilk kullanıcı isteğinden önce hazırla.
player.stream.start()
player.started = True

print("[AUDIO STREAM] Önceden hazır")
print("[VOXCPM2 AUDIO] Worker hazır")


def watch_interrupts():
    last_version = None

    while True:
        try:
            current_version = (
                get_interrupt_version()
            )

            if last_version is None:
                last_version = current_version

            elif current_version != last_version:
                last_version = current_version

                player.clear()

                audio_redis.set(
                    PLAYBACK_BUFFER_SECONDS_KEY,
                    0,
                    ex=2,
                )

                print(
                    "[AUDIO INTERRUPTED] "
                    f"version={current_version}"
                )

        except redis.exceptions.RedisError as error:
            print(
                "[REDIS ERROR] interrupt watcher:",
                error,
            )

        time.sleep(0.02)



def publish_buffer_level():

    while True:
        try:
            buffered_seconds = (
                player.get_buffered_seconds()
            )

            audio_redis.set(
                PLAYBACK_BUFFER_SECONDS_KEY,
                buffered_seconds,
                ex=2,
            )

        except redis.exceptions.RedisError as error:
            print(
                "[REDIS ERROR] buffer level:",
                error,
            )

        time.sleep(0.02)


buffer_thread = threading.Thread(
    target=publish_buffer_level,
    daemon=True,
)

buffer_thread.start()

interrupt_thread = threading.Thread(
    target=watch_interrupts,
    daemon=True,
)

interrupt_thread.start()


while True:
    try:
        result = audio_redis.blpop(
            VOXCPM2_AUDIO_QUEUE,
            timeout=1,
        )
    except redis.exceptions.RedisError as error:
        print("[REDIS ERROR] audio queue:", error)
        time.sleep(1)
        continue

    if result is None:
        continue

    _, audio_payload = result

    try:
        message_turn_id, audio_bytes = (
            decode_audio_message(
                audio_payload
            )
        )
    except (TypeError, ValueError) as error:
        print(
            "[AUDIO MESSAGE ERROR]",
            error,
        )
        continue

    current_turn_id = get_interrupt_version()

    turn_id = (
        message_turn_id
        if message_turn_id is not None
        else current_turn_id
    )

    if turn_id != current_turn_id:
        print(
            "[AUDIO STALE] "
            f"turn_id={turn_id} "
            f"| current={current_turn_id}"
        )
        continue

    print(
        "[VOXCPM2 REDIS AUDIO]",
        len(audio_bytes),
        f"| turn_id={turn_id}",
    )

    player.add_audio(audio_bytes)

    if get_interrupt_version() != turn_id:
        player.clear()
        continue

    try:
        audio_redis.set(
            PLAYBACK_BUFFER_SECONDS_KEY,
            player.get_buffered_seconds(),
            ex=2,
        )
    except redis.exceptions.RedisError as error:
        print(
            "[REDIS ERROR] immediate buffer level:",
            error,
        )
