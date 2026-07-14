"""Self-learning tutorial: predict 30-day readmission from MIMIC-III admissions.

This is a beginner-friendly structured-data ML tutorial. It intentionally uses
only ADMISSIONS and PATIENTS, never clinical notes. The target is whether a
patient has another hospital admission within 30 days after discharge.

Do not publish MIMIC data, patient-level records, or credentials. Run this
script only in an environment authorized for MIMIC-III access.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Step 0: Set paths and a random seed. Change MIMIC_ROOT for your authorized
# local copy of MIMIC-III. The script creates only aggregate outputs.
SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent
MIMIC_ROOT = PROJECT_ROOT.parent / "mimic-iii-clinical-database-1.4" / "mimic-iii-clinical-database-1.4"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_and_label_admissions():
    """Load adult admissions and label 30-day readmission without look-ahead features.

    The label uses the *next* admission after a discharge. Features later in the
    tutorial use only information available at admission or from a patient's
    prior admission history. We remove in-hospital deaths because a subsequent
    readmission cannot occur for those admissions.
    """
    admissions_path = MIMIC_ROOT / "ADMISSIONS.csv.gz"
    patients_path = MIMIC_ROOT / "PATIENTS.csv.gz"
    if not admissions_path.exists() or not patients_path.exists():
        raise FileNotFoundError("Set MIMIC_ROOT to an authorized MIMIC-III directory.")

    admissions = pd.read_csv(
        admissions_path,
        compression="gzip",
        usecols=[
            "SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "ADMISSION_TYPE",
            "INSURANCE", "LANGUAGE", "MARITAL_STATUS", "ETHNICITY", "HOSPITAL_EXPIRE_FLAG",
        ],
        parse_dates=["ADMITTIME", "DISCHTIME"],
    )
    patients = pd.read_csv(
        patients_path,
        compression="gzip",
        usecols=["SUBJECT_ID", "DOB"],
        parse_dates=["DOB"],
    )
    data = admissions.merge(patients, on="SUBJECT_ID", how="left")
    data = data.sort_values(["SUBJECT_ID", "ADMITTIME"]).copy()

    # Use the next hospitalization only to make the label, never as a feature.
    data["next_admittime"] = data.groupby("SUBJECT_ID")["ADMITTIME"].shift(-1)
    data["days_to_next_admission"] = (data["next_admittime"] - data["DISCHTIME"]).dt.total_seconds() / 86_400
    data["readmitted_30d"] = data["days_to_next_admission"].between(0, 30, inclusive="both").astype(int)

    # MIMIC shifts dates and encodes very old ages. Clipping at 90 produces a
    # conventional adult age feature without pretending the shifted date is exact.
    data["age"] = ((data["ADMITTIME"] - data["DOB"]).dt.total_seconds() / (365.25 * 86_400)).clip(18, 90)
    data = data[(data["age"] >= 18) & data.HOSPITAL_EXPIRE_FLAG.eq(0)].copy()
    return data


def make_features(data):
    """Build admission-time and prior-history features while preventing leakage."""
    data = data.sort_values(["SUBJECT_ID", "ADMITTIME"]).copy()

    # cumcount is the number of earlier admissions for this patient, so it is
    # available at the time of the current admission.
    data["prior_admissions"] = data.groupby("SUBJECT_ID").cumcount()
    data["previous_dischtime"] = data.groupby("SUBJECT_ID")["DISCHTIME"].shift(1)
    data["days_since_previous_discharge"] = (
        (data["ADMITTIME"] - data["previous_dischtime"]).dt.total_seconds() / 86_400
    )

    # These administrative fields are easy to obtain but can encode inequity.
    # They are included for a teaching discussion, not as causal explanations.
    numeric_features = ["age", "prior_admissions", "days_since_previous_discharge"]
    categorical_features = ["ADMISSION_TYPE", "INSURANCE", "LANGUAGE", "MARITAL_STATUS", "ETHNICITY"]
    feature_frame = data[numeric_features + categorical_features].copy()
    return feature_frame, data["readmitted_30d"].copy(), data["SUBJECT_ID"].copy(), numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    """Impute missing values, scale numeric data, and one-hot encode categories."""
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", min_frequency=100)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def evaluate_models(X, y, groups, numeric_features, categorical_features):
    """Compare two models using patient-disjoint train/test data.

    GroupShuffleSplit prevents one patient's admissions from appearing in both
    splits, which would inflate performance by leaking patient history.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_index, test_index = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model_definitions = {
        "Logistic regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        "Random forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, class_weight="balanced_subsample",
            random_state=SEED, n_jobs=-1,
        ),
    }
    fitted_models, metric_rows = {}, []
    for name, estimator in model_definitions.items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor(numeric_features, categorical_features)),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        probability = pipeline.predict_proba(X_test)[:, 1]
        metric_rows.append({
            "model": name,
            "test_auroc": roc_auc_score(y_test, probability),
            "test_average_precision": average_precision_score(y_test, probability),
        })
        fitted_models[name] = (pipeline, probability)

    metrics = pd.DataFrame(metric_rows).sort_values("test_auroc", ascending=False)
    return metrics, fitted_models, y_test


def save_aggregate_figures(metrics, fitted_models, y_test):
    """Save a model-comparison table and ROC curve without patient-level output."""
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    for name, (_, probability) in fitted_models.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probability)
        auroc = roc_auc_score(y_test, probability)
        axis.plot(false_positive_rate, true_positive_rate, label=f"{name} (AUROC {auroc:.3f})")
    axis.plot([0, 1], [0, 1], "--", color="gray", label="No-skill reference")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="30-day readmission: held-out ROC curves")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curves.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    admissions = load_and_label_admissions()
    X, y, groups, numeric_features, categorical_features = make_features(admissions)
    metrics, fitted_models, y_test = evaluate_models(X, y, groups, numeric_features, categorical_features)
    save_aggregate_figures(metrics, fitted_models, y_test)

    print(f"Adult non-death admissions analyzed: {len(X):,}")
    print(f"30-day readmission prevalence: {y.mean():.1%}")
    print(metrics.to_string(index=False))
    print(f"Aggregate outputs saved to: {OUTPUT_DIR}")
