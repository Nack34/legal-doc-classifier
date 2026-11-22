import random

import pytest
from fastapi.testclient import TestClient

from classifier.model import explain, make_pipeline, top_terms, train_and_evaluate
from classifier.preprocess import normalize
from data.generate import LABELS, generate, make_filing


@pytest.fixture(scope="module")
def small_dataset():
    rows = generate(per_class=40, seed=1, label_noise=0)
    return [normalize(t) for t, _ in rows], [label for _, label in rows]


def test_generator_is_reproducible_and_balanced():
    a = generate(per_class=5, seed=3)
    b = generate(per_class=5, seed=3)
    assert a == b
    assert len(a) == 5 * len(LABELS)


def test_training_beats_chance_and_selects_a_model(small_dataset):
    texts, labels = small_dataset
    _, result = train_and_evaluate(texts, labels)
    assert result.best_model in result.cv_scores
    assert result.test_macro_f1 > 0.8  # chance level is ~0.17 with 6 classes
    assert sorted(result.labels) == sorted(LABELS)


@pytest.mark.parametrize("name", ["logreg", "complement_nb"])
def test_explanations_point_to_relevant_terms(small_dataset, name):
    texts, labels = small_dataset
    model = make_pipeline(name).fit(texts, labels)
    assert "cautelar" in " ".join(top_terms(model)["medida_cautelar"])

    text = normalize(make_filing("recurso_apelacion", random.Random(99), noise=0))
    evidence = explain(model, text, "recurso_apelacion")
    assert evidence and any("apel" in term or "recurso" in term for term in evidence)


@pytest.fixture(scope="module")
def client(small_dataset, tmp_path_factory):
    import joblib

    from api import main

    texts, labels = small_dataset
    model_dir = tmp_path_factory.mktemp("artifacts")
    joblib.dump(make_pipeline("logreg").fit(texts, labels), model_dir / "model.joblib")
    (model_dir / "metrics.json").write_text(
        '{"best_model": "logreg", "cv_scores": {}, "test_macro_f1": 1, "labels": []}'
    )
    main.ARTIFACTS = model_dir
    main.get_model.cache_clear()
    yield TestClient(main.app)
    main.get_model.cache_clear()


def test_classify_endpoint(client):
    text = make_filing("medida_cautelar", random.Random(5), noise=0)
    response = client.post("/classify", json={"text": text})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["label"] == "medida_cautelar"
    assert body["prediction"]["name"] == "Solicitud de medida cautelar"
    assert 0 < body["prediction"]["confidence"] <= 1
    assert len(body["alternatives"]) == 2
    assert body["key_facts"]["case_number"]
    assert body["evidence"]


def test_classify_rejects_short_or_empty_text(client):
    assert client.post("/classify", json={"text": "corto"}).status_code == 422
    assert client.post("/classify", json={"text": "12345 / 678 - 90 . 12 , 34 ; 56"}).status_code == 422


def test_metadata_endpoints(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert set(client.get("/labels").json()) == set(LABELS)
    assert client.get("/model").json()["best_model"] == "logreg"
