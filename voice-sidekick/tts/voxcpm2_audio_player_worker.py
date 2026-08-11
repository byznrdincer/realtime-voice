import threading
import time

import redis

from queue_service.redis_client import (
    audio_redis,
    PLAYBACK_BUFFER_SECONDS_KEY,
)
from tts.audio_player import ContinuousAudioPlayer


VOXCPM2_AUDIO_QUEUE = "voxcpm2_audio_queue"

player = ContinuousAudioPlayer(
    sample_rate=48000,
    prebuffer_seconds=0.2,
)

player.start()

print("[VOXCPM2 AUDIO] Worker hazır")


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


while True:
    try:
        result = audio_redis.blpop(
            VOXCPM2_AUDIO_QUEUE,
            timeout=5,
        )
    except redis.exceptions.RedisError as error:
        print("[REDIS ERROR] audio queue:", error)
        time.sleep(1)
        continue

    if result is None:
        continue

    _, audio_bytes = result

    print(
        "[VOXCPM2 REDIS AUDIO]",
        len(audio_bytes),
    )

    player.add_audio(audio_bytes)

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
