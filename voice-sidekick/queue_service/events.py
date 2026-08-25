import json
import time

from queue_service.redis_client import text_redis


EVENT_CHANNEL = "voice_sidekick_events"
HISTORY_KEY = "voice_sidekick_ui_history"
HISTORY_TYPES = {"user", "assistant_done"}


def publish_event(kind, **payload):
    message = {
        "type": kind,
        "ts": time.time(),
        **payload,
    }
    encoded = json.dumps(
        message,
        ensure_ascii=False,
    )

    try:
        pipeline = text_redis.pipeline()
        pipeline.publish(EVENT_CHANNEL, encoded)
        if kind in HISTORY_TYPES and payload.get("text"):
            pipeline.lpush(HISTORY_KEY, encoded)
            pipeline.ltrim(HISTORY_KEY, 0, 39)
        pipeline.execute()
    except Exception:
        return


def recent_events():
    try:
        items = text_redis.lrange(HISTORY_KEY, 0, 39)
    except Exception:
        return []

    return list(reversed(items))
