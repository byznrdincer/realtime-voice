import os

from openai import OpenAI
from llm.sentence_buffer import SentenceBuffer


client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

buffer = SentenceBuffer()


def chat(user_text):

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Sen Sidekick isimli, kısa ve doğal cevaplar veren "
                    "bir sesli asistansın. Yanıtlarını konuşma dilinde ver. "
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

    print("Asistan: ", end="", flush=True)

    for chunk in stream:

        token = chunk.choices[0].delta.content

        if token is None:
            continue

        print(token, end="", flush=True)

        buffer.add(token)

    buffer.flush()

    print()

