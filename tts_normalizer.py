import re
from num2words import num2words


# Sadece gerçekten problem çıkaran kelimeler
TTS_REPLACEMENTS = {
    "NVIDIA": "Nivida",
    "H200": "H iki yüz",

    "WebRTC": "web r t c",
    "PCM16": "pi ci em on altı",

    "Chatterbox": "Çatterboks",
    "Whisper": "Vispır",

    "OpenAI": "Open ay",

    "kHz": "kilohertz",
    "MHz": "megahertz",
}


def number_to_turkish(match):
    """
    125 -> yüz yirmi beş
    3 -> üç
    """

    number = int(match.group())

    try:
        return num2words(
            number,
            lang="tr"
        )

    except Exception:
        return match.group()



def normalize_numbers(text):

    # Ondalık sayı
    # 1.5 / 1,5
    text = re.sub(
        r"(\d+)[,.](\d+)",
        r"\1 virgül \2",
        text
    )


    # Normal sayılar
    # 24 -> yirmi dört
    text = re.sub(
        r"\b\d+\b",
        number_to_turkish,
        text
    )


    return text



def normalize_units(text):

    replacements = {

        "Hz": "hertz",

        "kHz": "kilohertz",

        "GB": "gigabayt",

        "MB": "megabayt",

        "ms": "milisaniye",

        "saniye": "saniye",

    }


    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )


    return text



def normalize_tts_text(text: str) -> str:
    """
    Chatterbox öncesi text normalization.

    Akış:

    LLM text
       |
       ↓
    Teknik kelime düzeltme
       |
       ↓
    Sayı -> yazı
       |
       ↓
    TTS
    """


    if not text:
        return text



    # Teknik kelimeler

    for old, new in TTS_REPLACEMENTS.items():

        text = text.replace(
            old,
            new
        )



    # Birimler

    text = normalize_units(text)



    # Sayılar

    text = normalize_numbers(text)



    # Fazla boşluk

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()



    return text