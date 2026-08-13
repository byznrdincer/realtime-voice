# Realtime Voice Sidekick

Gerçek zamanlı Türkçe sesli asistan sistemi.

## Güncel mimari

Mikrofon → Energy VAD → Faster-Whisper → LLM → SentenceBuffer
→ Redis → H200 VoxCPM2 → PCM16 ses akışı → Mac hoparlörü

Sistem; tur kimlikli mesajlar, prompt cache, streaming playback,
backpressure, worker takibi ve sesli barge-in desteğine sahiptir.

## Proje bölümleri

- `voice-sidekick/`: Mac üzerinde çalışan ana uygulama
- `voice-sidekick/server_voxcpm.py`: H200 VoxCPM2 sunucusu
- `infra/whisper-h200/`: H200 Faster-Whisper Docker servisi
- `docker-compose.legacy.yml`: Eski Chatterbox mimarisi

Eski Docker Compose, güncel VoxCPM2 ve H200 Whisper mimarisini
temsil etmez.

## Hızlı çalıştırma

```bash
cd voice-sidekick
source ../venv/bin/activate
redis-cli ping
PYTHONPATH=. python3 main_voxcpm2.py
```

TTS ve Whisper SSH tünelleri uygulamadan önce açık olmalıdır.

Ayrıntılı rehber:

- `voice-sidekick/docs/voxcpm2_h200_ssh_runbook.md`
