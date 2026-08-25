import json
import re
import time
from difflib import SequenceMatcher

from queue_service.redis_client import text_redis


LAST_TTS_KEY = "voice_sidekick_last_tts"


def normalize_echo_text(text):
    if not text:
        return ""

    text = text.casefold()
    text = re.sub(
        r"[^\wçğıöşü]+",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(text.split())


def remember_spoken_text(text):
    cleaned = (text or "").strip()
    if not cleaned:
        return

    payload = json.dumps(
        {
            "text": cleaned,
            "ts": time.time(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

    try:
        text_redis.lpush(LAST_TTS_KEY, payload)
        text_redis.ltrim(LAST_TTS_KEY, 0, 9)
        text_redis.expire(LAST_TTS_KEY, 30)
    except Exception:
        return


def recent_spoken_texts():
    try:
        raw_items = text_redis.lrange(
            LAST_TTS_KEY,
            0,
            9,
        )
    except Exception:
        return []

    items = []
    for raw in raw_items:
        try:
            item = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue

        text = item.get("text")
        timestamp = item.get("ts")
        if isinstance(text, str) and timestamp:
            items.append(item)

    return items


def is_echo_transcript(
    heard,
    spoken_items,
    playback_hot,
    now=None,
    ratio_threshold=0.82,
):
    heard_n = normalize_echo_text(heard)
    if not heard_n or not spoken_items:
        return False

    now = time.time() if now is None else now
    heard_words = heard_n.split()

    for item in spoken_items:
        spoken_n = normalize_echo_text(
            item.get("text")
        )
        if not spoken_n:
            continue

        age = now - float(item.get("ts") or 0)
        if age > 12:
            continue

        ratio = SequenceMatcher(
            None,
            heard_n,
            spoken_n,
        ).ratio()

        if ratio >= ratio_threshold and age <= 8:
            return True

        spoken_words = spoken_n.split()
        heard_set = set(heard_words)
        spoken_set = set(spoken_words)

        if (
            len(heard_words) >= 3
            and heard_set
            and heard_set <= spoken_set
            and age <= 8
        ):
            return True

        if (
            playback_hot
            and age <= 6
            and heard_n != spoken_n
            and (
                spoken_n.startswith(heard_n)
                or heard_n.startswith(spoken_n)
            )
        ):
            return True

    return False


def choose_vad_threshold(
    playback_seconds,
    normal_threshold=0.015,
    playback_threshold=0.10,
    enabled=True,
    active_seconds=0.0,
):
    if (
        enabled
        and playback_seconds > active_seconds
    ):
        return max(
            normal_threshold,
            playback_threshold,
        )

    return normal_threshold


class EchoGuard:
    def __init__(
        self,
        hangover_seconds=0.7,
        active_seconds=0.0,
        barge_in_threshold=0.10,
        normal_threshold=0.025,
        enabled=True,
    ):
        self.hangover_seconds = hangover_seconds
        self.active_seconds = active_seconds
        self.barge_in_threshold = (
            barge_in_threshold
        )
        self.normal_threshold = normal_threshold
        self.enabled = enabled
        self.last_hot_at = 0.0

    def observe_playback(
        self,
        playback_seconds,
        now=None,
    ):
        now = (
            time.monotonic()
            if now is None
            else now
        )

        if (
            self.enabled
            and playback_seconds
            > self.active_seconds
        ):
            self.last_hot_at = now

        return now

    def is_hot(self, now=None):
        if not self.enabled or self.last_hot_at <= 0:
            return False

        now = (
            time.monotonic()
            if now is None
            else now
        )
        return (
            now - self.last_hot_at
            <= self.hangover_seconds
        )

    def threshold(self, now=None):
        if self.is_hot(now):
            return max(
                self.normal_threshold,
                self.barge_in_threshold,
            )

        return self.normal_threshold

    def should_ignore_start(self, rms, now=None):
        return (
            self.is_hot(now)
            and rms < self.barge_in_threshold
        )
