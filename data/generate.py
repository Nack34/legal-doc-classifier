"""
Generate a SYNTHETIC corpus of Argentine court filings ("escritos judiciales").

Real filings contain personal data and are not freely redistributable, so this repo ships with
generated examples. The generator mixes shared boilerplate, class-specific phrasing, phrases
borrowed from other classes and random sentence drops, so the task is not trivially separable.
Swap `data/escritos.csv` for a real, anonymized dataset with the same two columns
(`text`, `label`) and the training pipeline works unchanged.

Usage:  python -m data.generate --per-class 300 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

FIRST = [
    "Juan",
    "María",
    "Carlos",
    "Laura",
    "Diego",
    "Sofía",
    "Martín",
    "Lucía",
    "Pablo",
    "Ana",
    "Jorge",
    "Valeria",
    "Ricardo",
    "Camila",
    "Hernán",
    "Florencia",
    "Gustavo",
    "Paula",
]
LAST = [
    "Pérez",
    "Gómez",
    "Rodríguez",
    "Fernández",
    "López",
    "Martínez",
    "García",
    "Sánchez",
    "Romero",
    "Álvarez",
    "Torres",
    "Ruiz",
    "Ramírez",
    "Acosta",
    "Benítez",
    "Medina",
    "Herrera",
]
COMPANIES = [
    "Transportes del Sur S.A.",
    "Constructora Platense S.R.L.",
    "Seguros La Unión S.A.",
    "Banco del Litoral S.A.",
    "Inmobiliaria Centro S.R.L.",
    "Distribuidora Norte S.A.",
]
CAUSES = [
    "daños y perjuicios",
    "cobro de pesos",
    "despido",
    "desalojo",
    "accidente de tránsito",
    "cumplimiento de contrato",
    "alimentos",
    "ejecución hipotecaria",
    "nulidad de acto jurídico",
]
COURTS = [
    "Juzgado Civil y Comercial N° {n} de La Plata",
    "Juzgado de Primera Instancia en lo Civil N° {n}",
    "Tribunal de Trabajo N° {n} de La Plata",
    "Juzgado de Familia N° {n} de Quilmes",
    "Juzgado Civil y Comercial N° {n} de Lomas de Zamora",
]
CITIES = ["La Plata", "Quilmes", "Lomas de Zamora", "Berazategui", "Avellaneda"]

# Sentences that appear in any kind of filing.
SHARED = [
    "Que en tal carácter vengo en legal tiempo y forma a efectuar la presentación que se detalla.",
    "Que se tenga presente lo manifestado a sus efectos.",
    "Que conforme surge de las constancias de autos, la cuestión se encuentra debidamente acreditada.",
    "Que acompaño copia para traslado conforme lo dispuesto por el artículo 120 del CPCC.",
    "Que dejo constancia de que la presente se efectúa en el marco del expediente digital.",
    "Que fundo el derecho de mi parte en las normas del Código Civil y Comercial de la Nación.",
    "Que hago reserva del caso federal para el supuesto de que no se haga lugar a lo solicitado.",
    "Que se tenga por constituido el domicilio electrónico denunciado.",
]

FACTS = [
    "El día {date} se produjo el hecho que motiva las presentes actuaciones.",
    "La contraparte incumplió las obligaciones asumidas en el contrato suscripto el {date}.",
    "Mi mandante sufrió lesiones que le impidieron desarrollar sus tareas habituales.",
    "Se intimó fehacientemente mediante carta documento sin obtener respuesta alguna.",
    "El monto reclamado asciende a la suma de $ {amount} con más sus intereses.",
    "Las partes mantuvieron una relación comercial durante más de cinco años.",
    "El inmueble se encuentra ocupado sin título que justifique la tenencia.",
]

# Phrases characteristic of each type of filing, with several wordings each.
CLASS_PHRASES: dict[str, dict[str, list[str]]] = {
    "demanda": {
        "object": [
            "vengo a promover formal demanda de {cause} contra {defendant}",
            "vengo a iniciar acción por {cause} contra {defendant}",
            "interpongo demanda ordinaria de {cause} contra {defendant}",
        ],
        "body": [
            "Se reclama la suma de $ {amount} o lo que en más o en menos resulte de la prueba a producirse.",
            "Solicito se corra traslado de la demanda por el plazo de ley bajo apercibimiento de rebeldía.",
            "Los hechos que fundan la pretensión son los que a continuación se exponen.",
            "Ofrezco desde ya la prueba documental que se acompaña con el escrito de inicio.",
        ],
        "petition": [
            "Se tenga por promovida la demanda y se corra el traslado de ley.",
            "Oportunamente se dicte sentencia haciendo lugar a la demanda en todas sus partes, con costas.",
        ],
    },
    "contestacion_demanda": {
        "object": [
            "vengo en tiempo y forma a contestar la demanda incoada por {plaintiff}",
            "vengo a contestar el traslado de la demanda solicitando su total rechazo",
            "contesto demanda y niego todos y cada uno de los hechos invocados por la actora",
        ],
        "body": [
            "Niego todos y cada uno de los hechos expuestos en el escrito de demanda que no sean objeto de expreso reconocimiento.",
            "Desconozco la autenticidad de la documental acompañada por la contraria.",
            "Opongo excepción de falta de legitimación pasiva.",
            "La versión de los hechos que presenta la actora no se ajusta a la realidad.",
        ],
        "petition": [
            "Se tenga por contestada la demanda en legal tiempo y forma.",
            "Oportunamente se rechace la demanda en todas sus partes, con expresa imposición de costas a la actora.",
        ],
    },
    "recurso_apelacion": {
        "object": [
            "vengo a interponer recurso de apelación contra la sentencia de fecha {date}",
            "apelo la resolución dictada el {date} por causarle a mi parte un gravamen irreparable",
            "vengo a deducir recurso de apelación en relación contra el auto de fecha {date}",
        ],
        "body": [
            "La resolución recurrida incurre en una errónea valoración de la prueba producida.",
            "El a quo omitió considerar argumentos conducentes para la solución del litigio.",
            "Solicito se conceda el recurso libremente y con efecto suspensivo.",
            "Los agravios serán fundados en la oportunidad procesal correspondiente.",
        ],
        "petition": [
            "Se tenga por interpuesto en término el recurso de apelación y se lo conceda.",
            "Se eleven los autos a la Excma. Cámara de Apelaciones.",
        ],
    },
    "medida_cautelar": {
        "object": [
            "vengo a solicitar se decrete medida cautelar de embargo preventivo sobre los bienes de {defendant}",
            "solicito se ordene la medida cautelar de no innovar",
            "vengo a requerir se trabe inhibición general de bienes contra {defendant}",
        ],
        "body": [
            "La verosimilitud del derecho surge de la documentación acompañada.",
            "El peligro en la demora resulta evidente ante el riesgo de insolvencia de la demandada.",
            "Ofrezco caución juratoria en los términos del artículo 199 del CPCC.",
            "Solicito que la medida se despache inaudita parte atento la urgencia del caso.",
        ],
        "petition": [
            "Se decrete la medida cautelar solicitada y se libren los oficios correspondientes.",
            "Se tenga por ofrecida la contracautela y se ordene la traba del embargo.",
        ],
    },
    "ofrecimiento_prueba": {
        "object": [
            "vengo a ofrecer la prueba de la que intentaré valerme",
            "vengo a ofrecer prueba en los términos del artículo 484 del CPCC",
            "ofrezco los siguientes medios de prueba",
        ],
        "body": [
            "Testimonial: solicito se cite a declarar a los testigos que se indican con sus domicilios.",
            "Pericial contable: solicito se designe perito contador de la lista oficial.",
            "Informativa: se libren oficios a las entidades que se detallan a continuación.",
            "Confesional: se cite a absolver posiciones a la parte contraria bajo apercibimiento de ley.",
        ],
        "petition": [
            "Se tenga por ofrecida la prueba y se ordene su producción.",
            "Se provea la prueba ofrecida y se fije audiencia a tales efectos.",
        ],
    },
    "pronto_despacho": {
        "object": [
            "vengo a solicitar pronto despacho de las actuaciones",
            "solicito se resuelva sin más dilación la presentación efectuada el {date}",
            "vengo a requerir que se provea el escrito pendiente de despacho",
        ],
        "body": [
            "Ha transcurrido en exceso el plazo legal sin que se haya dictado resolución.",
            "La demora en resolver causa un perjuicio concreto a los derechos de mi parte.",
            "Se encuentran vencidos los plazos previstos en el artículo 34 del CPCC.",
        ],
        "petition": [
            "Se resuelva con la urgencia que el caso requiere.",
            "Se provea de inmediato lo solicitado.",
        ],
    },
}

LABELS = list(CLASS_PHRASES)


def _person(rng: random.Random) -> str:
    return f"{rng.choice(LAST)}, {rng.choice(FIRST)}"


def _fill(template: str, rng: random.Random, ctx: dict[str, str]) -> str:
    return template.format(
        date=f"{rng.randint(1, 28)}/{rng.randint(1, 12)}/{rng.randint(2019, 2025)}",
        amount=f"{rng.randint(50, 9000) * 1000:,}".replace(",", "."),
        **ctx,
    )


GENERIC_PETITIONS = [
    "Se tenga presente lo manifestado.",
    "Se provea conforme a derecho.",
    "Se agregue la documentación acompañada.",
]


def make_filing(label: str, rng: random.Random, noise: float = 0.4) -> str:
    plaintiff = _person(rng)
    defendant = rng.choice([_person(rng), rng.choice(COMPANIES)])
    ctx = {"plaintiff": plaintiff, "defendant": defendant, "cause": rng.choice(CAUSES)}
    phrases = CLASS_PHRASES[label]

    expte = f"Expte. N° {rng.randint(1000, 99999)}/{rng.randint(2019, 2025)}"
    caption = f'"{plaintiff} c/ {defendant} s/ {ctx["cause"]}"'
    court = rng.choice(COURTS).format(n=rng.randint(1, 27))
    lawyer = f"Dr{rng.choice(['.', 'a.'])} {rng.choice(FIRST)} {rng.choice(LAST)}"
    intro = (
        f"{rng.choice(FIRST)} {rng.choice(LAST)}, por derecho propio, con el patrocinio letrado de "
        f"{lawyer}, T° {rng.randint(1, 80)} F° {rng.randint(1, 500)} CALP, constituyendo domicilio "
        f"en calle {rng.randint(1, 120)} N° {rng.randint(100, 2000)} de {rng.choice(CITIES)}, "
        "a V.S. me presento y respetuosamente digo:"
    )

    sentences = [_fill(rng.choice(phrases["object"]), rng, ctx).capitalize() + "."]
    sentences += [_fill(s, rng, ctx) for s in rng.sample(phrases["body"], k=rng.randint(1, 3))]
    sentences += [_fill(s, rng, ctx) for s in rng.sample(FACTS, k=rng.randint(0, 3))]
    sentences += rng.sample(SHARED, k=rng.randint(1, 4))

    # Noise: borrow sentences from other classes (real filings often touch several things,
    # e.g. a complaint that also asks for an injunction).
    for _ in range(2):
        if rng.random() < noise:
            other = rng.choice([lab for lab in LABELS if lab != label])
            sentences.append(_fill(rng.choice(CLASS_PHRASES[other]["body"]), rng, ctx))
    # Noise: drop the opening sentence sometimes, so the "object" isn't always explicit.
    if rng.random() < noise and len(sentences) > 3:
        sentences.pop(0)
    rng.shuffle(sentences)

    # Some filings end with a generic request instead of the type-specific one.
    pool = GENERIC_PETITIONS if rng.random() < noise else phrases["petition"]
    petition = "Por todo lo expuesto, a V.S. solicito: " + " ".join(
        f"{i}) {_fill(p, rng, ctx)}" for i, p in enumerate(rng.sample(pool, k=rng.randint(1, 2)), 1)
    )
    closing = rng.choice(
        ["Proveer de conformidad, SERÁ JUSTICIA.", "Será Justicia.", "Proveer de conformidad."]
    )

    return "\n".join([f"{expte} {caption}", court, intro, " ".join(sentences), petition, closing])


def generate(per_class: int, seed: int, label_noise: float = 0.03) -> list[tuple[str, str]]:
    """`label_noise` relabels a small share of rows at random, to mimic annotator mistakes."""
    rng = random.Random(seed)
    rows = []
    for label in LABELS:
        for _ in range(per_class):
            text = make_filing(label, rng)
            if rng.random() < label_noise:
                label_out = rng.choice([lab for lab in LABELS if lab != label])
            else:
                label_out = label
            rows.append((text, label_out))
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "escritos.csv")
    args = parser.parse_args()

    rows = generate(args.per_class, args.seed)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} filings to {args.out}")


if __name__ == "__main__":
    main()
