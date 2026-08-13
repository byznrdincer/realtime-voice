from queue_service.redis_client import (
    audio_redis,
    TEXT_QUEUE,
    USER_INPUT_QUEUE,
    VOXCPM2_AUDIO_QUEUE,
    PLAYBACK_BUFFER_SECONDS_KEY,
    INTERRUPT_VERSION_KEY,
)


def get_interrupt_version():
    raw = audio_redis.get(
        INTERRUPT_VERSION_KEY
    )

    return int(raw or 0)


def request_interrupt(reason="barge-in"):
    with audio_redis.pipeline(
        transaction=True
    ) as pipeline:
        pipeline.incr(
            INTERRUPT_VERSION_KEY
        )
        pipeline.delete(
            TEXT_QUEUE,
            USER_INPUT_QUEUE,
            VOXCPM2_AUDIO_QUEUE,
            PLAYBACK_BUFFER_SECONDS_KEY,
        )

        version, deleted_keys = (
            pipeline.execute()
        )

    print(
        f"[BARGE-IN] reason={reason} "
        f"| version={version} "
        f"| silinen_anahtar={deleted_keys}"
    )

    return int(version)
