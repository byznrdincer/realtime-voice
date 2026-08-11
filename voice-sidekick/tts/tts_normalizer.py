import html
import re
from functools import lru_cache

from num2words import num2words


TTS_REPLACEMENTS = {
    "nvidia": "Nivida",
    "h200": "H iki yüz",
    "webrtc": "web r t c",
    "pcm16": "pi ci em on altı",
    "whisper": "Vispır",
    "openai": "Open ay",
}


UNIT_REPLACEMENTS = {
    "khz": "kilohertz",
    "mhz": "megahertz",
    "hz": "hertz",
    "gb": "gigabayt",
    "mb": "megabayt",
    "ms": "milisaniye",
}


# num2words bazı sürümlerde Türkçe sayıları birleşik döndürüyor.
# Bu kelimeler birleşik çıktıyı güvenli şekilde ayırmak için kullanılır.
TURKISH_NUMBER_WORDS = {
    "sıfır",
    "bir",
    "iki",
    "üç",
    "dört",
    "beş",
    "altı",
    "yedi",
    "sekiz",
    "dokuz",
    "on",
    "yirmi",
    "otuz",
    "kırk",
    "elli",
    "altmış",
    "yetmiş",
    "seksen",
    "doksan",
    "yüz",
    "bin",
    "milyon",
    "milyar",
    "trilyon",
    "katrilyon",
    "kentilyon",
    "sekstilyon",
    "septilyon",
    "oktilyon",
    "nonilyon",
    "desilyon",
    "eksi",
}


SORTED_NUMBER_WORDS = tuple(
    sorted(
        TURKISH_NUMBER_WORDS,
        key=len,
        reverse=True,
    )
)


def normalize_markdown(text: str):
    """
    Markdown ve HTML biçimlendirmelerini konuşulabilir
    düz metne dönüştürür.
    """

    # &amp; gibi HTML karakterlerini çöz.
    text = html.unescape(text)

    # Markdown kaçış karakterlerini çöz.
    text = re.sub(
        r"\\([*_`~#>\-])",
        r"\1",
        text,
    )

    # Markdown resimleri:
    # ![açıklama](adres) → açıklama
    text = re.sub(
        r"!\[([^\]]*)\]\([^)]+\)",
        r"\1",
        text,
    )

    # Markdown bağlantıları:
    # [OpenAI](adres) → OpenAI
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text,
    )

    # Otomatik bağlantıları kaldır.
    text = re.sub(
        r"<https?://[^>]+>",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Kod bloğu işaretlerini ve varsa dil adını kaldır.
    text = re.sub(
        r"```[a-zA-Z0-9_+-]*",
        " ",
        text,
    )

    text = text.replace("```", " ")

    # Satır içi kod işaretleri
    text = text.replace("`", "")

    # Markdown başlıkları
    text = re.sub(
        r"(?m)^\s{0,3}#{1,6}\s*",
        "",
        text,
    )

    # Alıntı işaretleri
    text = re.sub(
        r"(?m)^\s*>\s?",
        "",
        text,
    )

    # Sırasız liste işaretleri
    text = re.sub(
        r"(?m)^\s*[-+*]\s+",
        "",
        text,
    )

    # Sıralı listeleri konuşulabilir başlıklara dönüştür.
    # 1. Sağlık → Birinci: Sağlık
    list_ordinals = {
        "1": "Birinci",
        "2": "İkinci",
        "3": "Üçüncü",
        "4": "Dördüncü",
        "5": "Beşinci",
        "6": "Altıncı",
        "7": "Yedinci",
        "8": "Sekizinci",
        "9": "Dokuzuncu",
        "10": "Onuncu",
    }

    def ordered_list_repl(match):
        number = match.group(1)
        label = list_ordinals.get(number)

        if label is None:
            label = (
                "Madde "
                + number_to_turkish(number)
            )

        return f"{label}: "

    text = re.sub(
        r"(?m)^\s*(\d+)[.)]\s+",
        ordered_list_repl,
        text,
    )

    # Yatay çizgiler
    text = re.sub(
        r"(?m)^\s*([-*_])(?:\s*\1){2,}\s*$",
        " ",
        text,
    )

    # Markdown tablo ayırıcıları
    text = re.sub(
        r"(?m)^\s*\|?(?:\s*:?-+:?\s*\|)+\s*$",
        " ",
        text,
    )

    # Tablo sütunlarını kısa duraklara dönüştür.
    text = text.replace("|", ", ")

    # Kalın, italik ve üstü çizili metin işaretleri
    text = re.sub(
        r"(\*\*\*|\*\*|\*|___|__|~~)",
        "",
        text,
    )

    # Alt çizgileri kelime ayracı yap.
    text = text.replace("_", " ")

    # HTML etiketlerini kaldır.
    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    # Tek başına verilen URL’leri kaldır.
    text = re.sub(
        r"https?://\S+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    return text


@lru_cache(maxsize=512)
def split_compound_number_word(word: str):
    """
    Birleşik Türkçe sayı kelimesini parçalar.

    yirmibeş → yirmi beş
    bindokuzyüzotuzsekiz → bin dokuz yüz otuz sekiz
    """

    word = word.strip().lower()

    if not word:
        return ""

    if word in TURKISH_NUMBER_WORDS:
        return word

    @lru_cache(maxsize=None)
    def find_parts(index):
        if index == len(word):
            return ()

        for number_word in SORTED_NUMBER_WORDS:
            if not word.startswith(number_word, index):
                continue

            remaining = find_parts(
                index + len(number_word)
            )

            if remaining is not None:
                return (number_word,) + remaining

        return None

    parts = find_parts(0)

    if parts is None:
        return word

    return " ".join(parts)


def fix_num2words_spacing(text: str):
    """
    num2words çıktısındaki birleşik sayı kelimelerini ayırır.
    """

    text = text.replace("-", " ")

    words = re.findall(
        r"[a-zA-ZçğıöşüÇĞİÖŞÜ]+|[^\w\s]",
        text,
    )

    fixed_words = []

    for word in words:
        if re.fullmatch(
            r"[a-zA-ZçğıöşüÇĞİÖŞÜ]+",
            word,
        ):
            fixed_words.append(
                split_compound_number_word(word)
            )
        else:
            fixed_words.append(word)

    result = " ".join(fixed_words)

    # Noktalama işaretlerinden önceki gereksiz boşlukları kaldır.
    result = re.sub(
        r"\s+([.,!?;:])",
        r"\1",
        result,
    )

    return re.sub(
        r"\s+",
        " ",
        result,
    ).strip()


def number_to_turkish(number: str):
    """
    Sayıyı num2words ile Türkçeye çevirir ve
    birleşik kelimeleri ayırır.
    """

    try:
        converted = num2words(
            int(number),
            lang="tr",
        )

        return fix_num2words_spacing(converted)

    except Exception:
        return number


def decimal_to_turkish(
    integer_part: str,
    decimal_part: str,
):
    left = number_to_turkish(integer_part)

    right = " ".join(
        number_to_turkish(digit)
        for digit in decimal_part
    )

    return f"{left} virgül {right}"


def normalize_percentages(text: str):
    """
    %25 → yüzde yirmi beş
    %4,25 → yüzde dört virgül iki beş
    """

    def repl(match):
        value = match.group(1)

        decimal_match = re.fullmatch(
            r"(\d+)[,.](\d+)",
            value,
        )

        if decimal_match:
            number = decimal_to_turkish(
                decimal_match.group(1),
                decimal_match.group(2),
            )
        else:
            number = number_to_turkish(value)

        return f"yüzde {number}"

    return re.sub(
        r"%\s*(\d+(?:[,.]\d+)?)",
        repl,
        text,
    )


def normalize_decimals(text: str):
    """
    4.25 veya 4,25 → dört virgül iki beş
    """

    def repl(match):
        return decimal_to_turkish(
            match.group(1),
            match.group(2),
        )

    return re.sub(
        r"(\d+)[,.](\d+)",
        repl,
        text,
    )


def normalize_numbers(text: str):
    """
    Metinde kalan tam sayıları Türkçeye çevirir.
    """

    return re.sub(
        r"\b\d+\b",
        lambda match: number_to_turkish(
            match.group()
        ),
        text,
    )


def normalize_units(text: str):
    """
    48kHz → 48 kilohertz
    500ms → 500 milisaniye
    """

    text = re.sub(
        r"(\d)(kHz|MHz|Hz|GB|MB|ms)\b",
        r"\1 \2",
        text,
        flags=re.IGNORECASE,
    )

    for old, new in UNIT_REPLACEMENTS.items():
        text = re.sub(
            rf"\b{old}\b",
            new,
            text,
            flags=re.IGNORECASE,
        )

    return text


def normalize_words(text: str):
    """
    Teknik kelimeleri konuşulabilir biçime dönüştürür.
    """

    for old, new in TTS_REPLACEMENTS.items():
        text = re.sub(
            rf"\b{old}\b",
            new,
            text,
            flags=re.IGNORECASE,
        )

    return text


def normalize_tts_text(text: str):
    """
    LLM metnini TTS için konuşulabilir düz metne dönüştürür.
    """

    if not text:
        return text

    # Önce Markdown ve biçimlendirme karakterlerini temizle.
    text = normalize_markdown(text)

    # Teknik kelime ve birimleri dönüştür.
    text = normalize_words(text)
    text = normalize_units(text)

    # Önce özel sayı biçimleri, sonra kalan sayılar.
    text = normalize_percentages(text)
    text = normalize_decimals(text)
    text = normalize_numbers(text)

    # Satır sonlarını ve fazla boşlukları temizle.
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text
