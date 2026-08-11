import time

import redis
import requests

from queue_service.redis_client import (
    text_redis,
    audio_redis,
    TEXT_QUEUE,
    PLAYBACK_BUFFER_SECONDS_KEY,
)

from tts.tts_normalizer import normalize_tts_text


URL = "http://127.0.0.1:18031/tts/stream"
VOXCPM2_AUDIO_QUEUE = "voxcpm2_audio_queue"
SAMPLE_RATE = 48000
NETWORK_CHUNK_BYTES = 15360
INTER_SENTENCE_PAUSE_SECONDS = 0.15

MAX_BUFFER_SECONDS = 8.0
RESUME_BUFFER_SECONDS = 4.0


def wait_for_playback_headroom():

    waiting = False

    while True:
        try:
            raw = audio_redis.get(
                PLAYBACK_BUFFER_SECONDS_KEY
            )

            player_seconds = (
                float(raw)
                if raw is not None
                else 0.0
            )

            redis_chunk_count = audio_redis.llen(
                VOXCPM2_AUDIO_QUEUE
            )

            redis_queue_seconds = (
                redis_chunk_count
                * NETWORK_CHUNK_BYTES
                / 2
                / SAMPLE_RATE
            )

            total_buffered_seconds = (
                player_seconds
                + redis_queue_seconds
            )

        except redis.exceptions.RedisError as error:
            print(
                "[REDIS ERROR] buffer check:",
                error,
            )
            time.sleep(1)
            continue

        if not waiting:

            if (
                total_buffered_seconds
                <= MAX_BUFFER_SECONDS
            ):
                return

            waiting = True

            print(
                f"[BACKPRESSURE] "
                f"player={player_seconds:.2f}s "
                f"| redis={redis_queue_seconds:.2f}s "
                f"| total={total_buffered_seconds:.2f}s"
            )

        if (
            total_buffered_seconds
            <= RESUME_BUFFER_SECONDS
        ):
            print(
                f"[BACKPRESSURE RELEASE] "
                f"total={total_buffered_seconds:.2f}s"
            )
            return

        time.sleep(0.1)


def main():
    print("[VOXCPM2 TTS] Worker hazır")

    while True:
        try:
            result = text_redis.brpop(
                TEXT_QUEUE,
                timeout=5,
            )
        except redis.exceptions.RedisError as error:
            print("[REDIS ERROR] text queue:", error)
            time.sleep(1)
            continue

        if result is None:
            continue

        _, text = result

        wait_for_playback_headroom()

        print(
            f"\n[VOXCPM2 START] "
            f"words={len(text.split())} "
            f"| {text!r}"
        )

        text = normalize_tts_text(text)
        # Ultimate Cloning referans ritmini kullanır

        start = time.perf_counter()
        first_chunk_time = None
        total_bytes = 0
        chunk_count = 0

        try:
            with requests.post(
                URL,
                json={"text": text},
                stream=True,
                timeout=120,
            ) as response:
                response.raise_for_status()

                for audio_bytes in response.iter_content(
                    chunk_size=NETWORK_CHUNK_BYTES
                ):
                    if not audio_bytes:
                        continue

                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()

                        print(
                            f"[FIRST AUDIO] "
                            f"{first_chunk_time - start:.3f}s"
                        )

                    wait_for_playback_headroom()

                    audio_redis.rpush(
                        VOXCPM2_AUDIO_QUEUE,
                        audio_bytes,
                    )

                    total_bytes += len(audio_bytes)
                    chunk_count += 1

                pause_bytes = bytes(
                    int(
                        SAMPLE_RATE
                        * INTER_SENTENCE_PAUSE_SECONDS
                        * 2
                    )
                )

                audio_redis.rpush(
                    VOXCPM2_AUDIO_QUEUE,
                    pause_bytes,
                )

                total_bytes += len(pause_bytes)

            generation_time = time.perf_counter() - start
            audio_seconds = total_bytes / 2 / SAMPLE_RATE
            rtf = (
                generation_time / audio_seconds
                if audio_seconds > 0
                else 0
            )

            print(
                f"[VOXCPM2 DONE] "
                f"chunks={chunk_count} "
                f"| generation={generation_time:.3f}s "
                f"| audio={audio_seconds:.3f}s "
                f"| RTF={rtf:.3f}"
            )

        except Exception as error:
            print("[VOXCPM2 ERROR]", error)


if __name__ == "__main__":
    main()
