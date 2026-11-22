"""Model definition, training and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from classifier.preprocess import normalize


def build_features() -> FeatureUnion:
    # Word n-grams capture legal formulas ("medida cautelar", "recurso de apelacion");
    # character n-grams make the model robust to typos and inflections.
    return FeatureUnion(
        [
            ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
        ]
    )


CANDIDATES = {
    "logreg": lambda: LogisticRegression(max_iter=2000, C=4.0),
    "linear_svm": lambda: CalibratedClassifierCV(LinearSVC(C=0.5), cv=3),
    "complement_nb": lambda: ComplementNB(alpha=0.3),
}


def make_pipeline(name: str = "logreg") -> Pipeline:
    return Pipeline(
        [
            ("features", build_features()),
            ("clf", CANDIDATES[name]()),
        ]
    )


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    df = pd.read_csv(path)
    return [normalize(t) for t in df["text"]], df["label"].tolist()


@dataclass
class TrainingResult:
    best_model: str
    cv_scores: dict[str, float]
    test_macro_f1: float
    report: dict
    labels: list[str]
    confusion: list[list[int]]


def train_and_evaluate(
    texts: list[str], labels: list[str], seed: int = 42
) -> tuple[Pipeline, TrainingResult]:
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=seed
    )

    # Model selection with 5-fold cross-validation on the training split only.
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = {
        name: float(
            np.mean(cross_val_score(make_pipeline(name), x_train, y_train, cv=folds, scoring="f1_macro"))
        )
        for name in CANDIDATES
    }
    best = max(cv_scores, key=cv_scores.get)

    model = make_pipeline(best).fit(x_train, y_train)
    predicted = model.predict(x_test)
    label_order = sorted(set(labels))
    result = TrainingResult(
        best_model=best,
        cv_scores={k: round(v, 4) for k, v in cv_scores.items()},
        test_macro_f1=round(float(f1_score(y_test, predicted, average="macro")), 4),
        report=classification_report(y_test, predicted, output_dict=True, zero_division=0),
        labels=label_order,
        confusion=confusion_matrix(y_test, predicted, labels=label_order).tolist(),
    )
    # Refit on all data for the model that gets served.
    return make_pipeline(best).fit(texts, labels), result


def save(model: Pipeline, result: TrainingResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(result.__dict__, indent=2, ensure_ascii=False))


def _class_weights(model: Pipeline) -> np.ndarray | None:
    """Per-class weights for linear models: coef_ (logistic regression) or
    feature_log_prob_ (Complement NB, where a higher value also means "more typical of the class")."""
    clf = model.named_steps["clf"]
    for attr in ("coef_", "feature_log_prob_"):
        weights = getattr(clf, attr, None)
        if weights is not None:
            return np.asarray(weights)
    return None


def _word_vectorizer(model: Pipeline) -> TfidfVectorizer:
    return model.named_steps["features"].transformer_list[0][1]


def top_terms(model: Pipeline, per_class: int = 8) -> dict[str, list[str]]:
    """Most characteristic word n-grams of each class."""
    weights = _class_weights(model)
    if weights is None:
        return {}
    words = _word_vectorizer(model)
    names = words.get_feature_names_out()
    n_words = len(names)
    classes = model.named_steps["clf"].classes_
    result = {}
    for label, row in zip(classes, weights, strict=True):
        word_part = row[:n_words]
        # Compare against the average class so shared legal boilerplate doesn't dominate.
        distinctive = word_part - weights[:, :n_words].mean(axis=0)
        result[label] = [names[i] for i in np.argsort(distinctive)[::-1][:per_class]]
    return result


def explain(model: Pipeline, normalized_text: str, label: str, k: int = 5) -> list[str]:
    """Word n-grams in this document that pushed the prediction toward `label`."""
    weights = _class_weights(model)
    if weights is None:
        return []
    words = _word_vectorizer(model)
    names = words.get_feature_names_out()
    n_words = len(names)
    row = words.transform([normalized_text]).toarray()[0]
    classes = list(model.named_steps["clf"].classes_)
    distinctive = weights[classes.index(label), :n_words] - weights[:, :n_words].mean(axis=0)
    contribution = row * distinctive
    present = [i for i in np.argsort(contribution)[::-1][:k] if contribution[i] > 0]
    return [names[i] for i in present]
