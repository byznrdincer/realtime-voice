import redis


text_redis = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True,
)

audio_redis = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=False,
)


TEXT_QUEUE = "text_queue"
USER_INPUT_QUEUE = (
    "voice_sidekick_user_input_queue"
)
VOXCPM2_AUDIO_QUEUE = "voxcpm2_audio_queue"

INTERRUPT_VERSION_KEY = (
    "voice_sidekick_interrupt_version"
)

PLAYBACK_BUFFER_SECONDS_KEY = (
    "voxcpm2_playback_buffer_seconds"
)

TTS_ACTIVE_KEY = (
    "voice_sidekick_tts_active"
)

