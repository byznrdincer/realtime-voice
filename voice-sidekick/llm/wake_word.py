import re


WAKE_PATTERN = re.compile(
    r"^\s*(?:"
    r"(?:hey|hay)\s+"
    r"(?:oppy|opie|opi|opy|opey|api|hopi|opp|op|o\s*p)"
    r"|(?:oppy|opie|opi|opy|opey|hopi)"
    r")\b"
    r"(?P<command>.*)$",
    flags=re.IGNORECASE,
)


def extract_wake_command(text):
    if not text:
        return None

    match = WAKE_PATTERN.match(text)

    if match is None:
        return None

    return re.sub(
        r"^[\s,;:!?.\-–—]+",
        "",
        match.group("command"),
    ).strip()
