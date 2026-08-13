import unittest

from tts.tts_normalizer import (
    normalize_tts_text,
    number_to_turkish,
    split_compound_number_word,
)


class TTSNormalizerTests(unittest.TestCase):

    def test_empty_text_is_preserved(self):
        self.assertEqual(
            normalize_tts_text(""),
            "",
        )

    def test_technical_words_and_units(self):
        text = (
            "NVIDIA H200, WebRTC ve PCM16 ile "
            "48kHz, 500ms."
        )

        expected = (
            "Nivida H iki yüz, web r t c ve "
            "pi ci em on altı ile kırk sekiz "
            "kilohertz, beş yüz milisaniye."
        )

        self.assertEqual(
            normalize_tts_text(text),
            expected,
        )

    def test_percentage_and_decimal(self):
        text = "%25 ve %4,25 tamamlandı."

        expected = (
            "yüzde yirmi beş ve yüzde dört "
            "virgül iki beş tamamlandı."
        )

        self.assertEqual(
            normalize_tts_text(text),
            expected,
        )

    def test_regular_decimal(self):
        self.assertEqual(
            normalize_tts_text("Sürüm 1.5 hazır."),
            "Sürüm bir virgül beş hazır.",
        )

    def test_markdown_is_removed(self):
        text = (
            "## Başlık\n"
            "- **OpenAI** ve "
            "[Whisper](https://example.com)."
        )

        self.assertEqual(
            normalize_tts_text(text),
            "Başlık Open ay ve Vispır.",
        )

    def test_ordered_list_is_spoken(self):
        text = "1. Sağlık\n11. Teknoloji"

        self.assertEqual(
            normalize_tts_text(text),
            "Birinci: Sağlık Madde on bir: Teknoloji",
        )

    def test_compound_number_word_is_split(self):
        self.assertEqual(
            split_compound_number_word("yirmibeş"),
            "yirmi beş",
        )

        self.assertEqual(
            number_to_turkish("1938"),
            "bin dokuz yüz otuz sekiz",
        )

    def test_replacement_does_not_change_larger_word(self):
        self.assertEqual(
            normalize_tts_text("OpenAITable"),
            "OpenAITable",
        )


if __name__ == "__main__":
    unittest.main()
