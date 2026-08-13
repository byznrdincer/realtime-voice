import json
import struct


TEXT_MESSAGE_VERSION = 1
AUDIO_MAGIC = b"VSA1"
AUDIO_HEADER = struct.Struct(">4sQ")


def _validate_turn_id(turn_id):
    if (
        isinstance(turn_id, bool)
        or not isinstance(turn_id, int)
        or turn_id < 0
    ):
        raise ValueError(
            "turn_id negatif olmayan int olmalı"
        )

    return turn_id


def encode_text_message(turn_id, text):
    _validate_turn_id(turn_id)

    if not isinstance(text, str):
        raise TypeError("text str olmalı")

    return json.dumps(
        {
            "version": TEXT_MESSAGE_VERSION,
            "turn_id": turn_id,
            "text": text,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def decode_text_message(payload):
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")

    if not isinstance(payload, str):
        raise TypeError(
            "Metin mesajı str veya bytes olmalı"
        )

    try:
        message = json.loads(payload)
    except json.JSONDecodeError:
        return None, payload

    if (
        not isinstance(message, dict)
        or message.get("version")
        != TEXT_MESSAGE_VERSION
        or "turn_id" not in message
        or "text" not in message
    ):
        return None, payload

    turn_id = _validate_turn_id(
        message["turn_id"]
    )
    text = message["text"]

    if not isinstance(text, str):
        raise ValueError(
            "Mesajdaki text str olmalı"
        )

    return turn_id, text


def encode_audio_message(
    turn_id,
    pcm_bytes,
):
    _validate_turn_id(turn_id)

    if not isinstance(
        pcm_bytes,
        (bytes, bytearray, memoryview),
    ):
        raise TypeError(
            "pcm_bytes bytes benzeri olmalı"
        )

    return (
        AUDIO_HEADER.pack(
            AUDIO_MAGIC,
            turn_id,
        )
        + bytes(pcm_bytes)
    )


def decode_audio_message(payload):
    if not isinstance(
        payload,
        (bytes, bytearray, memoryview),
    ):
        raise TypeError(
            "Ses mesajı bytes benzeri olmalı"
        )

    payload = bytes(payload)

    if not payload.startswith(AUDIO_MAGIC):
        return None, payload

    if len(payload) < AUDIO_HEADER.size:
        raise ValueError(
            "Ses mesajı başlığı eksik"
        )

    magic, turn_id = AUDIO_HEADER.unpack(
        payload[:AUDIO_HEADER.size]
    )

    if magic != AUDIO_MAGIC:
        return None, payload

    _validate_turn_id(turn_id)

    return (
        turn_id,
        payload[AUDIO_HEADER.size:],
    )
