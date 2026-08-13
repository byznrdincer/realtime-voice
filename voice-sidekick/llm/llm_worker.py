import os
import time

from openai import OpenAI
from llm.sentence_buffer import SentenceBuffer
from queue_service.interrupt import (
    get_interrupt_version,
)


client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)



def chat(user_text, turn_id):
    if get_interrupt_version() != turn_id:
        print(
            f"[LLM STALE] turn_id={turn_id}"
        )
        return False

    llm_started = time.perf_counter()

    print(
        f"[LATENCY] LLM START "
        f"| turn_id={turn_id}"
    )

    buffer = SentenceBuffer(
        turn_id=turn_id,
        started_at=llm_started,
    )

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Sen Sidekick isimli, kısa ve doğal cevaplar veren "
                    "bir sesli asistansın. Yanıtlarını konuşma dilinde ver. "
                    "Yanıtının ilk cümlesini kısa, doğrudan ve tamamlanmış kur. "
                    "Ardından gerektiği kadar ayrıntı verebilirsin. "
                    "Markdown veya madde işareti kullanma. "
                    "Her liste maddesini tam iki cümle şeklinde yaz. "
                    "Örnek: Birinci kullanım alanı sağlıktır. "
                    "Yapay zeka hastalıkların teşhisine yardımcı olur. "
                    "Madde numarasını veya başlığı tek başına ayrı cümle yapma."
                )
            },
            {
                "role": "user",
                "content": user_text
            }
        ],
        stream=True
    )

    print(
        f"[LATENCY] LLM STREAM OPEN "
        f"| turn_id={turn_id} "
        f"| elapsed="
        f"{time.perf_counter() - llm_started:.3f}s"
    )

    first_token_seen = False

    for chunk in stream:

        if (
            get_interrupt_version()
            != turn_id
        ):
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

        token = chunk.choices[0].delta.content

        if token is None:
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

        print(token, end="", flush=True)

        buffer.add(token)

    buffer.flush()

    print()
    return True

