import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from llm.sentence_buffer import SentenceBuffer
from llm.wake_word import extract_wake_command
from queue_service.interrupt import (
    get_interrupt_version,
)


load_dotenv()


BULUTISTAN_MODEL = os.getenv(
    "BULUTISTAN_MODEL",
    "gpt-oss-120b",
)

client = OpenAI(
    api_key=os.environ["BULUTISTAN_API_KEY"],
    base_url=os.environ["BULUTISTAN_BASE_URL"],
)


def select_reasoning_effort(user_text):
    normalized = user_text.casefold()

    high_requests = (
        "derin araştır",
        "derinlemesine araştır",
        "kapsamlı araştır",
        "kaynaklarıyla araştır",
        "literatür taraması",
        "güncel kaynakları araştır",
        "web'de araştır",
        "web üzerinde araştır",
        "çok kapsamlı incele",
    )

    medium_requests = (
        "detaylı analiz et",
        "kapsamlı analiz et",
        "adım adım çöz",
        "adım adım açıkla",
        "detaylı karşılaştır",
        "derinlemesine karşılaştır",
        "hesaplayarak açıkla",
        "kodumu analiz et",
        "kodumu incele",
        "hatayı analiz et",
        "ayrıntılı plan oluştur",
        "derin düşün",
    )

    if any(
        request in normalized
        for request in high_requests
    ):
        return "high"

    if any(
        request in normalized
        for request in medium_requests
    ):
        return "medium"

    return "low"

def chat(user_text, turn_id):
    if get_interrupt_version() != turn_id:
        print(
            f"[LLM STALE] turn_id={turn_id}"
        )
        return False

    wake_command = extract_wake_command(
        user_text
    )

    if wake_command is not None:
        wake_started = time.perf_counter()

        wake_buffer = SentenceBuffer(
            turn_id=turn_id,
            started_at=wake_started,
        )

        wake_response = (
            "Oppy burada, yardıma hazır!"
        )

        print(
            f"[WAKE WORD] "
            f"turn_id={turn_id}"
        )
        print(
            f"Oppy: {wake_response}"
        )

        # Önce sabit karşılama cümlesini söyle.
        wake_buffer.add(wake_response)
        wake_buffer.flush()

        # Yalnızca “Hey Oppy” denmişse burada bitir.
        if not wake_command:
            return True

        # Devamında soru varsa LLM'e gönder.
        user_text = wake_command

    reasoning_effort = select_reasoning_effort(
        user_text
    )

    llm_started = time.perf_counter()

    print(
        f"[LATENCY] LLM START "
        f"| turn_id={turn_id} "
        f"| model={BULUTISTAN_MODEL} "
        f"| reasoning={reasoning_effort}"
    )

    buffer = SentenceBuffer(
        turn_id=turn_id,
        started_at=llm_started,
    )

    stream = client.chat.completions.create(
        model=BULUTISTAN_MODEL,
        reasoning_effort=reasoning_effort,
        messages=[
            {
                "role": "system",
                "content": (
                    "Adın Oppy. Sen kısa ve doğal cevaplar veren bir sesli asistansın. "
                    "Kullanıcı adını sorarsa yalnızca 'Oppy.' diye cevap ver. "
                    "Yanıtlarını konuşma dilinde ver. "
                    "Yanıtının ilk cümlesini kısa, doğrudan ve tamamlanmış kur. "
                    "Ardından gerektiği kadar ayrıntı verebilirsin. "
                    "Markdown veya madde işareti kullanma. "
                    "Her liste maddesini tam iki cümle şeklinde yaz. "
                    "Örnek: Birinci kullanım alanı sağlıktır. "
                    "Yapay zeka hastalıkların teşhisine yardımcı olur. "
                    "Madde numarasını veya başlığı tek başına ayrı cümle yapma."
                ),
            },
            {
                "role": "user",
                "content": user_text,
            },
        ],
        stream=True,
    )

    print(
        f"[LATENCY] LLM STREAM OPEN "
        f"| turn_id={turn_id} "
        f"| elapsed="
        f"{time.perf_counter() - llm_started:.3f}s"
    )

    first_token_seen = False
    first_reasoning_seen = False

    for chunk in stream:
        if get_interrupt_version() != turn_id:
            buffer.reset()

            close = getattr(
                stream,
                "close",
                None,
            )

            if callable(close):
                close()

            print("\n[LLM INTERRUPTED]")
            return False

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        reasoning = getattr(
            delta,
            "reasoning_content",
            None,
        )

        if reasoning and not first_reasoning_seen:
            first_reasoning_seen = True

            print(
                f"[LATENCY] LLM FIRST REASONING "
                f"| turn_id={turn_id} "
                f"| elapsed="
                f"{time.perf_counter() - llm_started:.3f}s"
            )

        token = getattr(delta, "content", None)

        if not token:
            continue

        if not first_token_seen:
            first_token_seen = True

            print(
                f"[LATENCY] LLM FIRST TOKEN "
                f"| turn_id={turn_id} "
                f"| elapsed="
                f"{time.perf_counter() - llm_started:.3f}s"
            )

            print(
                "Asistan: ",
                end="",
                flush=True,
            )

        print(
            token,
            end="",
            flush=True,
        )

        buffer.add(token)

    buffer.flush()

    print()
    return True