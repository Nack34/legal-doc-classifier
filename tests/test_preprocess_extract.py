from classifier.extract import extract
from classifier.preprocess import normalize, strip_accents

SAMPLE = (
    'Expte. N° 12345/2024 "Pérez, Juan c/ Transportes del Sur S.A. s/ daños y perjuicios"\n'
    "Juzgado Civil y Comercial N° 14 de La Plata\n"
    "Vengo a solicitar se decrete medida cautelar de embargo preventivo por $ 1.500.000,50 "
    "conforme el contrato del 3/5/2023 y la intimación del 10/6/2023."
)


def test_strip_accents_keeps_enie():
    assert strip_accents("Apelación año Ñandú") == "Apelacion año Ñandu"


def test_normalize_removes_case_identifiers_numbers_and_stopwords():
    text = normalize(SAMPLE)
    assert "perez" not in text  # caption removed: it identifies the case, not the filing type
    assert "12345" not in text and "1" not in text.split()
    assert " de " not in f" {text} "
    assert "medida cautelar embargo preventivo" in text


def test_extract_key_facts():
    facts = extract(SAMPLE)
    assert facts.case_number == "12345/2024"
    assert facts.plaintiff == "Pérez, Juan"
    assert facts.defendant == "Transportes del Sur S.A."
    assert facts.matter == "daños y perjuicios"
    assert facts.court == "Juzgado Civil y Comercial N° 14 de La Plata"
    assert facts.dates == ["3/5/2023", "10/6/2023"]
    assert facts.amounts == [1500000.50]


def test_extract_handles_missing_fields():
    facts = extract("Escrito sin datos de carátula.")
    assert facts.case_number is None and facts.plaintiff is None
    assert facts.dates == [] and facts.amounts == []
