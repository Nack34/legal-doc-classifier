"""Rule-based extraction of key facts from a filing: case number, caption, court, dates, amounts."""

import re
from dataclasses import asdict, dataclass, field

_EXPTE = re.compile(r"expte\.?\s*n[°º]?\s*(\d+/\d{4})", re.IGNORECASE)
_CAPTION = re.compile(r'"(?P<plaintiff>[^"]+?)\s+c/\s+(?P<defendant>[^"]+?)\s+s/\s+(?P<matter>[^"]+)"')
_COURT = re.compile(r"((?:Juzgado|Tribunal)[^\n.]*?N[°º]\s*\d+(?:\s+de\s+[A-ZÁÉÍÓÚ][\wáéíóúñ ]+)?)")
_DATE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
_AMOUNT = re.compile(r"\$\s?([\d.]+(?:,\d{2})?)")


@dataclass
class KeyFacts:
    case_number: str | None = None
    plaintiff: str | None = None
    defendant: str | None = None
    matter: str | None = None
    court: str | None = None
    dates: list[str] = field(default_factory=list)
    amounts: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _to_number(amount: str) -> float:
    # Argentine format: dots for thousands, comma for decimals.
    return float(amount.replace(".", "").replace(",", "."))


def extract(text: str) -> KeyFacts:
    facts = KeyFacts()
    if m := _EXPTE.search(text):
        facts.case_number = m.group(1)
    if m := _CAPTION.search(text):
        facts.plaintiff = m.group("plaintiff").strip()
        facts.defendant = m.group("defendant").strip()
        facts.matter = m.group("matter").strip()
    if m := _COURT.search(text):
        facts.court = m.group(1).strip()
    facts.dates = list(dict.fromkeys(_DATE.findall(text)))
    facts.amounts = [_to_number(a) for a in dict.fromkeys(_AMOUNT.findall(text))]
    return facts
