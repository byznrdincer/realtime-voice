import csv
import itertools
import time

import requests


URL = "http://127.0.0.1:8031/tts/stream"
SAMPLE_RATE = 48000
TEST_SECONDS = 10 * 60

TEXTS = [
    (
        "Merhaba Beyzanur, gerçek zamanlı ses sistemimizin "
        "dayanıklılığını test ediyoruz."
    ),
    (
        "Yapay zeka sistemleri doğal ve anlaşılır "
        "konuşmalar üretebilir."
    ),
    (
        "Bugün VoxCPM iki modelinin hızını ve bağlantı "
        "kararlılığını ölçüyoruz."
    ),
    (
        "Uzun cevaplarda ses parçalarının kesilmeden "
        "ulaşması oldukça önemlidir."
    ),
    (
        "Voice Sidekick, metni H iki yüz üzerindeki modele "
        "gönderir. Üretilen ses parçaları tamamlanmayı "
        "beklemeden istemciye aktarılır."
    ),
]


def run_test():
    results = []
    test_start = time.time()

    print("[TEST] 10 dakikalık VoxCPM2 stres testi başladı.\n")

    for request_number, text in enumerate(
        itertools.cycle(TEXTS),
        start=1,
    ):
        if time.time() - test_start >= TEST_SECONDS:
            break

        request_start = time.time()
        first_audio = None
        total_bytes = 0
        status = "OK"
        error = ""

        try:
            with requests.post(
                URL,
                json={"text": text},
                stream=True,
                timeout=(10, 120),
            ) as response:
                response.raise_for_status()

                for chunk in response.iter_content(
                    chunk_size=15360
                ):
                    if not chunk:
                        continue

                    if first_audio is None:
                        first_audio = (
                            time.time() - request_start
                        )

                    total_bytes += len(chunk)

            generation = time.time() - request_start
            audio_seconds = (
                total_bytes / 2 / SAMPLE_RATE
            )

            rtf = (
                generation / audio_seconds
                if audio_seconds > 0
                else 0
            )

        except Exception as exc:
            generation = time.time() - request_start
            audio_seconds = 0
            rtf = 0
            status = "ERROR"
            error = str(exc)

        result = {
            "request": request_number,
            "status": status,
            "first_audio": round(first_audio or 0, 3),
            "generation": round(generation, 3),
            "audio_seconds": round(audio_seconds, 3),
            "rtf": round(rtf, 3),
            "bytes": total_bytes,
            "error": error,
        }

        results.append(result)

        print(
            f"[{request_number:03d}] {status}"
            f" | first={result['first_audio']:.3f}s"
            f" | generation={result['generation']:.3f}s"
            f" | audio={result['audio_seconds']:.3f}s"
            f" | RTF={result['rtf']:.3f}"
        )

    save_results(results)
    print_summary(results)


def save_results(results):
    output_file = "voxcpm2_stress_results_10min.csv"

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nCSV raporu: {output_file}")


def print_summary(results):
    successful = [
        result
        for result in results
        if result["status"] == "OK"
    ]

    failed = [
        result
        for result in results
        if result["status"] != "OK"
    ]

    print("\n========== TEST SONUCU ==========")
    print("Toplam istek:", len(results))
    print("Başarılı:", len(successful))
    print("Hatalı:", len(failed))

    if not successful:
        return

    average_first = (
        sum(
            result["first_audio"]
            for result in successful
        )
        / len(successful)
    )

    average_rtf = (
        sum(
            result["rtf"]
            for result in successful
        )
        / len(successful)
    )

    print(
        "Ortalama ilk ses:",
        round(average_first, 3),
        "saniye",
    )
    print(
        "Ortalama RTF:",
        round(average_rtf, 3),
    )
    print(
        "En yüksek RTF:",
        max(
            result["rtf"]
            for result in successful
        ),
    )


if __name__ == "__main__":
    run_test()
