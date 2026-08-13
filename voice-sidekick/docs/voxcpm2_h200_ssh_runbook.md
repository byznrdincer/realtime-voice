# VoxCPM2 – H200, Docker ve SSH Tüneli Çalıştırma Rehberi

## 1. Sistem mimarisi

```text
Kullanıcı
→ LLM token akışı
→ SentenceBuffer
→ Redis TEXT_QUEUE
→ VoxCPM2 TTS Worker
→ Mac 127.0.0.1:18031
→ SSH tüneli
→ H200 Docker container 172.17.0.5:8031
→ VoxCPM2 prompt cache
→ PCM16 48 kHz ses akışı
→ Redis VOXCPM2_AUDIO_QUEUE
→ ContinuousAudioPlayer
→ Mac hoparlörü
```

## 2. Temel bileşenler

- Mac: Ana uygulama, LLM, Redis ve ses oynatıcı
- H200: VoxCPM2 model sunucusu
- Docker container: `voxcpm2-test`
- TTS endpoint: `POST /tts/stream`
- Ses formatı: Mono PCM16, 48 kHz
- Yerel TTS adresi: `http://127.0.0.1:18031/tts/stream`
- H200 container portu: `8031`

## 3. H200 sunucusunu başlatma

GPU sunucusuna bağlan:

```bash
ssh bdincer@45.145.22.79
```

Container durumunu kontrol et:

```bash
docker ps -a --filter "name=voxcpm2-test"
```

VoxCPM2 sunucusunu başlat:

```bash
docker exec -d voxcpm2-test bash -lc 'python3 -m uvicorn server_voxcpm:app --app-dir /data --host 0.0.0.0 --port 8031 > /data/server_voxcpm.log 2>&1'
```

Logları izle:

```bash
docker exec voxcpm2-test tail -f /data/server_voxcpm.log
```

Aşağıdaki satırlar sunucunun hazır olduğunu gösterir:

```text
[VOICE CACHE] sidekick hazır
[VOICE CACHE] beyzanur hazır
[VOICE WARMUP] sidekick hazır
[VOICE WARMUP] beyzanur hazır
[MODEL] Hazır
Uvicorn running on http://0.0.0.0:8031
```

Container içindeki portu kontrol et:

```bash
docker exec voxcpm2-test python3 -c 'import socket; s=socket.socket(); s.settimeout(3); print(s.connect_ex(("127.0.0.1",8031))); s.close()'
```

Sonuç `0` ise servis çalışıyor, `111` ise kapalıdır.

## 4. Container IP adresini kontrol etme

```bash
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' voxcpm2-test
```

Mevcut yapıdaki beklenen IP:

```text
172.17.0.5
```

Container yeniden oluşturulursa IP değişebilir. Böyle bir durumda SSH tüneli yeni IP ile açılmalıdır.

## 5. Mac üzerinde SSH tünelini açma

Mac terminalinde:

```bash
ssh -fN -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 18031:172.17.0.5:8031 bdincer@45.145.22.79
```

Tünelin açık olduğunu kontrol et:

```bash
lsof -nP -iTCP:18031 -sTCP:LISTEN
```

Sağlık kontrolü:

```bash
curl http://127.0.0.1:18031/health
```

Başarılı örnek:

```json
{
  "status": "ok",
  "model": "VoxCPM2",
  "default_voice_profile": "sidekick",
  "voice_profiles": ["sidekick", "beyzanur"],
  "sample_rate": 48000
}
```

## 6. Mac uygulamasını çalıştırma

Proje klasörüne gir:

```bash
cd /Users/beyzanurdincer/projects/voice-sidekick/voice-sidekick
```

Sanal ortamı etkinleştir:

```bash
source ../venv/bin/activate
```

Redis’i kontrol et:

```bash
redis-cli ping
```

Beklenen cevap:

```text
PONG
```

Uygulamayı başlat:

```bash
PYTHONPATH=. python3 main_voxcpm2.py
```

## 7. Ortam değişkeni

`.env` dosyasında TTS adresi bulunur:

```env
VOXCPM2_TTS_URL=http://127.0.0.1:18031/tts/stream
```

Adres veya port değişirse worker kodunu değiştirmek yerine bu değer güncellenir.

## 8. Prompt-cache davranışı

Sunucu açılırken:

1. Sidekick referans sesi bir kez encode edilir.
2. Beyzanur referans sesi bir kez encode edilir.
3. İki sonuç GPU belleğinde tutulur.
4. Her profil için warm-up üretimi yapılır.
5. Her TTS isteğinde seçilen profilin hazır cache’i kullanılır.

Böylece referans WAV dosyaları her cümlede yeniden işlenmez.

Cache sonrasında ilk parça gecikmesi yaklaşık:

```text
0.20–0.30 saniye
```

Önceki soğuk başlangıç değeri:

```text
2.806 saniye
```

## 9. Ses profili testi

Sidekick:

```bash
curl -X POST -H 'Content-Type: application/json' -d '{"text":"Merhaba, bu Sidekick ses profilidir.","voice_profile":"sidekick"}' http://127.0.0.1:18031/tts/stream -o /tmp/sidekick.pcm
```

Beyzanur:

```bash
curl -X POST -H 'Content-Type: application/json' -d '{"text":"Merhaba, bu Beyzanur ses profilidir.","voice_profile":"beyzanur"}' http://127.0.0.1:18031/tts/stream -o /tmp/beyzanur.pcm
```

## 10. Boş metin testi

```bash
curl -i -X POST -H 'Content-Type: application/json' -d '{"text":"","voice_profile":"sidekick"}' http://127.0.0.1:18031/tts/stream
```

Beklenen sonuç:

```text
HTTP/1.1 400 Bad Request
```

## 11. Otomatik testler

```bash
cd /Users/beyzanurdincer/projects/voice-sidekick/voice-sidekick

PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Mevcut sonuç:

```text
Ran 13 tests
OK
```

Test edilen başlıca davranışlar:

- İlk uzun cümlenin hızlı gönderilmesi
- İki kısa cümlenin birlikte gönderilmesi
- Ondalık noktanın cümle sonu sayılmaması
- Uzun buffer’ın kelime sınırından bölünmesi
- Flush davranışı
- Markdown temizliği
- Teknik kelime dönüşümleri
- Sayı, yüzde, ondalık ve birim dönüşümleri

## 12. Redis ve worker güvenliği

Ana uygulama başlangıçta ve kapanışta şunları temizler:

- `TEXT_QUEUE`
- `VOXCPM2_AUDIO_QUEUE`
- Playback buffer değeri

Redis kapalıysa sistem başlatılmaz.

Audio veya TTS worker çalışma sırasında kapanırsa ana program bunu algılar, diğer worker’ı durdurur ve kuyrukları temizler.

## 13. Sık karşılaşılan hatalar

### Connection refused

```text
Failed to establish a new connection
Connection refused
```

Mac üzerindeki SSH tüneli açık değildir.

Kontrol:

```bash
lsof -nP -iTCP:18031 -sTCP:LISTEN
```

### Connection reset by peer

Tünel açıktır fakat H200 tarafındaki `8031` servisi çalışmıyordur veya container IP’si değişmiştir.

Kontrol:

```bash
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' voxcpm2-test
```

```bash
docker exec voxcpm2-test python3 -c 'import socket; s=socket.socket(); s.settimeout(3); print(s.connect_ex(("127.0.0.1",8031))); s.close()'
```

### Eski konuşmanın yeniden çalması

Ana uygulamadaki başlangıç ve kapanış Redis temizliği bunu engeller.

### Backpressure logları

```text
[BACKPRESSURE]
[BACKPRESSURE RELEASE]
```

Bunlar hata değildir. VoxCPM2 sesi oynatılandan daha hızlı ürettiğinde player kuyruğunun aşırı büyümesini engeller.

### FutureWarning

```text
torch.nn.utils.weight_norm is deprecated
```

Bu bir çalışma hatası değildir; kullanılan bağımlılığın gelecek sürüm uyarısıdır.

## 14. Sistemi kapatma

Mac uygulamasını kapatmak için:

```text
Ctrl+C
```

Ana uygulama worker’ları durdurur ve Redis kuyruklarını temizler.

SSH tünelinin PID değerini bul:

```bash
lsof -nP -iTCP:18031 -sTCP:LISTEN
```

Ardından yalnızca görünen SSH PID değerini kapat:

```bash
kill <SSH_PID>
```

H200 VoxCPM2 sunucusunu kapatmak için:

```bash
docker exec voxcpm2-test pkill -TERM -f 'uvicorn server_voxcpm:app'
```
