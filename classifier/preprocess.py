"""Text normalization for Spanish legal filings."""

import re
import unicodedata

# Words that appear in almost every filing and carry no signal about its type.
LEGAL_STOPWORDS = {
    "de",
    "la",
    "el",
    "en",
    "y",
    "a",
    "los",
    "las",
    "del",
    "que",
    "por",
    "se",
    "con",
    "su",
    "al",
    "lo",
    "un",
    "una",
    "para",
    "mi",
    "me",
    "como",
    "sus",
    "o",
    "le",
    "es",
    "este",
    "esta",
    "vs",
    "v",
    "s",
    "c",
    "n",
    "dr",
    "dra",
    "calp",
    "expte",
    "autos",
    "parte",
    "digo",
    "presento",
    "respetuosamente",
    "letrado",
    "patrocinio",
    "domicilio",
    "constituyendo",
    "calle",
    "derecho",
    "propio",
    "justicia",
    "sera",
    "proveer",
    "conformidad",
}

_EXPTE = re.compile(r"expte\.?\s*n[°º]?\s*\d+/\d+", re.IGNORECASE)
_CAPTION = re.compile(r'"[^"]*"')
_NUMBERS = re.compile(r"\d[\d.,/]*")
_NON_WORD = re.compile(r"[^a-zñ\s]")
_SPACES = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    # Keep ñ: it changes meaning in Spanish (año / ano).
    text = text.replace("ñ", "\0").replace("Ñ", "\1")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return text.replace("\0", "ñ").replace("\1", "Ñ")


def normalize(text: str) -> str:
    """Lowercase, remove case numbers, the case caption, numbers, accents and stopwords.

    The caption ("Pérez c/ Gómez s/ daños") and case numbers are removed on purpose: they
    identify the case, not the type of filing, and would let the model memorize parties.
    """
    text = _EXPTE.sub(" ", text)
    text = _CAPTION.sub(" ", text)
    text = strip_accents(text.lower())
    text = _NUMBERS.sub(" ", text)
    text = _NON_WORD.sub(" ", text)
    tokens = [t for t in _SPACES.split(text) if t and t not in LEGAL_STOPWORDS and len(t) > 1]
    return " ".join(tokens)
