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

