"""
25COP927 Fundamentals of AI — Lab: Iris + First Classification + Evaluation
STUDENT STARTER (with TODOs)

================================================================================
WHAT THIS SCRIPT DOES (High-level)
--------------------------------------------------------------------------------
You will follow a standard machine-learning / data-mining workflow:

    1) Load dataset
    2) Explore (prints + plots)
    3) Corrupt data (simulate missing values + outliers)   <-- TODOs
    4) Clean data (median imputation + IQR clipping)       <-- TODOs
    5) Split into train/test
    6) Scale features (important for distance-based models like kNN)
    7) Train models (Dummy baseline, kNN, Logistic Regression)
    8) Evaluate models (confusion matrix + accuracy/precision/recall/F1)
    9) Optional: uncertainty via entropy from probabilities <-- TODO optional

================================================================================
WHAT YOU (STUDENT) MUST EDIT
--------------------------------------------------------------------------------
Only complete the TODO functions:

    TODO #1  inject_missing_values()
    TODO #2  inject_outliers()
    TODO #3  impute_missing_median()
    TODO #4  clip_outliers_iqr()
    TODO #5  entropy_from_proba()            (optional)
    FINAL    train_your_extra_classifier()  (optional extra classifier)

All other code should run as-is.

================================================================================
HOW TO RUN
--------------------------------------------------------------------------------
Install (if running in your own computer):
    pip install numpy pandas matplotlib scikit-learn

Run:
    python 25COP927Lab2_dm_iris_starter.py
"""

from __future__ import annotations  # allows forward type hints in older Python versions

# -----------------------------
# Imports (what each one is for)
# -----------------------------
from dataclasses import dataclass  # dataclass: convenient container for metrics
from typing import Dict, Tuple, List, Any  # type hints: improve readability

import numpy as np  # NumPy: fast arrays + random numbers + math
import pandas as pd  # pandas: DataFrame (tables, like Excel)
import matplotlib.pyplot as plt  # matplotlib: plotting library

# scikit-learn (machine learning library)
from sklearn.datasets import load_iris  # load a built-in dataset (no CSV needed)
from sklearn.model_selection import train_test_split  # split data into train/test
from sklearn.preprocessing import StandardScaler  # standardise features for fair distances

# Models (classifiers)
from sklearn.dummy import DummyClassifier  # simple baseline model (always predicts most frequent class)
from sklearn.neighbors import KNeighborsClassifier  # kNN classifier (distance-based)
from sklearn.linear_model import LogisticRegression  # logistic regression classifier (probabilistic)

# Metrics (evaluation)
from sklearn.metrics import (
    confusion_matrix,              # counts correct/incorrect for each class
    ConfusionMatrixDisplay,        # plot confusion matrix nicely
    accuracy_score,                # accuracy = correct / total
    precision_recall_fscore_support,  # precision/recall/F1 (macro averaging)
)

# Optional pandas plotting helper (pairwise scatter matrix)
from pandas.plotting import scatter_matrix  # quick EDA plot: scatter matrix
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


# -----------------------------
# Reproducibility (important for labs)
# -----------------------------
# If we fix a random seed, results are repeatable across runs/machines.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# -----------------------------
# Small container for metrics
# -----------------------------
@dataclass
class Metrics:
    """
    Store evaluation metrics.
    We use macro-average metrics because Iris is balanced (each class has 50 samples).
    """
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float


# =============================================================================
# Step 1 — Load dataset
# =============================================================================
def load_iris_as_dataframe() -> pd.DataFrame:
    """
    Load the Iris dataset as a pandas DataFrame.

    - iris.frame contains the features + numeric target label.
    - We also add 'target_name' so students can see readable labels.
    """
    iris = load_iris(as_frame=True)          # returns a Bunch with DataFrame fields
    df = iris.frame.copy()                  # copy to safely modify

    # Map numeric targets 0/1/2 → names setosa/versicolor/virginica
    df["target_name"] = df["target"].map(lambda i: iris.target_names[i])

    return df


# =============================================================================
# Step 2 — Exploration (EDA: Exploratory Data Analysis)
# =============================================================================
def basic_exploration(df: pd.DataFrame, feature_cols: List[str]) -> None:
    """
    Print common pandas inspection commands students should know.

    This function intentionally demonstrates many commands from the lab handout:
    df.head(), df.tail(), df.shape, df.columns, df.dtypes, df.info(), df.describe(),
    df.sample(), missing values checks, duplicates, groupby stats, correlation.
    """
    print("\n" + "=" * 80)
    print("STEP 2 — BASIC EXPLORATION (PANDAS COMMANDS)")
    print("=" * 80)

    # First rows: shows columns + example values
    print("\n[df.head()] First 5 rows:")
    print(df.head())

    # Last rows: useful to check end of file / dataset tail
    print("\n[df.tail()] Last 5 rows:")
    print(df.tail())

    # Shape: number of rows and columns
    print("\n[df.shape] Dataset shape (rows, columns):")
    print(df.shape)

    # Columns: names of columns
    print("\n[df.columns] Column names:")
    print(df.columns.tolist())

    # Data types of each column
    print("\n[df.dtypes] Data types:")
    print(df.dtypes)

    # info(): memory + non-null counts
    print("\n[df.info()] Summary info:")
    df.info()

    # describe(): summary stats for numeric columns
    print("\n[df.describe()] Numeric summary statistics:")
    print(df[feature_cols + ["target"]].describe())

    # describe(include="all"): include categorical/text columns too
    print("\n[df.describe(include='all')] Full summary (includes target_name):")
    print(df.describe(include="all"))

    # sample(): random rows (reproducible)
    print("\n[df.sample(5, random_state=42)] Random sample of 5 rows:")
    print(df.sample(5, random_state=RANDOM_SEED))

    # Missing values per column
    print("\n[df.isna().sum()] Missing values count per column:")
    print(df.isna().sum())

    # Missing fraction (mean of boolean mask)
    print("\n[df.isna().mean()] Missing values fraction per column:")
    print(df.isna().mean())

    # Class balance
    print("\n[df['target_name'].value_counts()] Class counts:")
    print(df["target_name"].value_counts())

    # Groupby mean per class
    print("\n[df.groupby('target_name').mean()] Mean feature value per class:")
    print(df.groupby("target_name").mean(numeric_only=True))

    # Groupby multiple stats
    print("\n[df.groupby('target_name').agg([...])] Several stats per class:")
    print(df.groupby("target_name")[feature_cols].agg(["mean", "std", "min", "max"]))

    # Duplicates count (Iris usually has none)
    print("\n[df.duplicated().sum()] Number of duplicate rows:")
    print(df.duplicated().sum())

    # Correlation matrix
    print("\n[df.corr()] Correlation matrix (numeric only):")
    print(df[feature_cols].corr(numeric_only=True))


def plot_feature_histograms(df: pd.DataFrame, feature_cols: List[str]) -> None:
    """
    Histograms show the distribution of each feature.
    """
    print("\nPlot: Histograms of features...")
    df[feature_cols].hist(bins=20, figsize=(10, 6))
    plt.suptitle("Histograms of Iris features")
    plt.tight_layout()
    plt.show()


def plot_boxplots(df: pd.DataFrame, feature_cols: List[str]) -> None:
    """
    Boxplots help detect outliers visually.
    """
    print("Plot: Boxplots of features (outlier check)...")
    plt.figure(figsize=(10, 4))
    df[feature_cols].plot(kind="box", rot=45)
    plt.title("Boxplots of Iris features (outlier check)")
    plt.tight_layout()
    plt.show()


def scatter_petal(df: pd.DataFrame) -> None:
    """
    Scatter plot of petal length vs petal width.
    This often shows clear separation between classes.
    """
    print("Plot: Scatter (petal length vs petal width) coloured by class...")
    plt.figure(figsize=(6, 5))
    plt.scatter(df["petal length (cm)"], df["petal width (cm)"], c=df["target"])
    plt.xlabel("petal length (cm)")
    plt.ylabel("petal width (cm)")
    plt.title("Petal length vs petal width (coloured by class)")
    plt.tight_layout()
    plt.show()


def plot_scatter_matrix(df: pd.DataFrame, feature_cols: List[str]) -> None:
    """
    Scatter matrix: pairwise scatter plots for all features.
    Useful for seeing separability across multiple pairs.
    """
    print("Plot: Scatter matrix (pairwise plots)...")
    scatter_matrix(df[feature_cols], figsize=(10, 10), diagonal="hist")
    plt.suptitle("Scatter matrix of Iris features")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: List[str]) -> None:
    """
    Correlation heatmap using matplotlib only.
    """
    print("Plot: Correlation heatmap...")
    corr = df[feature_cols].corr()
    plt.figure(figsize=(6, 5))
    plt.imshow(corr.values)
    plt.xticks(range(len(feature_cols)), feature_cols, rotation=45, ha="right")
    plt.yticks(range(len(feature_cols)), feature_cols)
    plt.title("Correlation heatmap (Pearson r)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


# =============================================================================
# Step 3 — Simulate data quality issues (students implement)
# =============================================================================
def inject_missing_values(
    df: pd.DataFrame,
    feature_cols: List[str],
    missing_rate: float = 0.05
) -> pd.DataFrame:
    """
    TODO #1:
    Randomly set ~missing_rate of the feature cells to NaN.

    Why?
    - Iris is normally clean.
    - We inject missing values to practise cleaning.

    Requirements:
    - Return a COPY of df (do not modify original).
    - Only modify feature columns.
    """
    df_corrupt = df.copy()

    # -------------------------
    # HINT (commented-out on purpose):
    # Anything starting with # is a comment, so Python ignores it.
    # Students must REMOVE the # to "activate" the code.
    # -------------------------
    #
    # n_rows = len(df_corrupt)
    # n_cols = len(feature_cols)
    # total_cells = n_rows * n_cols
    #
    # # number of cells to set to NaN
    # n_missing = int(missing_rate * total_cells)
    #
    # # choose unique random cell indices (flattened)
    # flat_indices = np.random.choice(total_cells, size=n_missing, replace=False)
    #
    # # convert flat index -> (row, col)
    # for idx in flat_indices:
    #     r = idx // n_cols
    #     c = idx % n_cols
    #     # use iloc for row position and column name lookup
    #     df_corrupt.iloc[r, df_corrupt.columns.get_loc(feature_cols[c])] = np.nan

    return df_corrupt


def inject_outliers(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_outliers: int = 6
) -> pd.DataFrame:
    """
    TODO #2:
    Inject outliers by multiplying random cells in feature columns by a factor.

    Why?
    - Outliers happen in real-world data.
    - We practise handling them via clipping.

    Requirements:
    - Return a COPY of df.
    - Only modify feature columns.
    """
    df_corrupt = df.copy()

    # HINT:
    #
    # for _ in range(n_outliers):
    #     r = np.random.randint(0, len(df_corrupt))     # random row index
    #     c = np.random.choice(feature_cols)            # random feature name
    #     factor = np.random.uniform(1.8, 2.5)          # multiply by 1.8..2.5
    #     df_corrupt.loc[df_corrupt.index[r], c] *= factor

    return df_corrupt


# =============================================================================
# Step 4 — Cleaning (students implement)
# =============================================================================
def impute_missing_median(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    TODO #3:
    Replace missing values using median imputation (per column).

    Median is robust to outliers.
    """
    df_clean = df.copy()

    # HINT:
    #
    # for c in feature_cols:
    #     med = df_clean[c].median(skipna=True)
    #     df_clean[c] = df_clean[c].fillna(med)

    return df_clean


def clip_outliers_iqr(df: pd.DataFrame, feature_cols: List[str], k: float = 1.5) -> pd.DataFrame:
    """
    TODO #4:
    Clip outliers using the IQR rule.

    Q1 = 25th percentile
    Q3 = 75th percentile
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    clip to [lower, upper]
    """
    df_clean = df.copy()

    # HINT:
    #
    # for c in feature_cols:
    #     q1 = df_clean[c].quantile(0.25)
    #     q3 = df_clean[c].quantile(0.75)
    #     iqr = q3 - q1
    #     lo = q1 - k * iqr
    #     hi = q3 + k * iqr
    #     df_clean[c] = df_clean[c].clip(lo, hi)

    return df_clean


# =============================================================================
# Step 5 — Split + Scale
# =============================================================================
def make_train_test(
    df: pd.DataFrame,
    feature_cols: List[str],
    test_size: float = 0.20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepare data for machine learning.

    1) X = features (numeric measurements)
    2) y = labels (0/1/2)
    3) train_test_split to evaluate on unseen data
    4) StandardScaler to put features on same scale

    Why scaling?
    - kNN uses distances. If one feature has larger scale, it dominates distances.
    """
    # Extract features matrix X and label vector y
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=int)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y  # keeps class balance in both train and test
    )

    # Create scaler object
    scaler = StandardScaler()

    # Fit scaler on training data only (avoid leakage!) and transform
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform test data using the same scaler
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# =============================================================================
# Step 6 — Models (training)
# =============================================================================
def train_knn(X_train: np.ndarray, y_train: np.ndarray, k: int = 5) -> KNeighborsClassifier:
    """
    Train a kNN classifier.

    In scikit-learn:
    - The "model" is a Python object (here: KNeighborsClassifier).
    - model.fit(X_train, y_train) means "train the model".
    - model.predict(X_test) means "predict labels".

    k = number of neighbours used in voting.
    """
    model = KNeighborsClassifier(n_neighbors=k)  # choose the classification algorithm + its hyperparameter
    model.fit(X_train, y_train)                  # training step
    return model


def train_logreg(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    Train a Logistic Regression classifier.

    Logistic Regression outputs probabilities (predict_proba),
    and predicts the class with highest probability.

    We set solver and max_iter for stability and to avoid warnings.
    """
    model = LogisticRegression(
        max_iter=500,
        random_state=RANDOM_SEED,
        solver="lbfgs",
        multi_class="auto",
    )
    model.fit(X_train, y_train)
    return model


# =============================================================================
# Step 7 — Evaluation (metrics + confusion matrix)
# =============================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """
    Compute evaluation metrics.

    accuracy: fraction correct
    precision/recall/F1: computed using macro average (each class treated equally)
    """
    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return Metrics(
        accuracy=acc,
        precision_macro=precision,
        recall_macro=recall,
        f1_macro=f1
    )


def print_metrics_nicely(name: str, m: Metrics) -> None:
    """Pretty-print metrics to the console."""
    print(f"\n{name}")
    print(f"  Accuracy         : {m.accuracy:.3f}")
    print(f"  Precision (macro): {m.precision_macro:.3f}")
    print(f"  Recall (macro)   : {m.recall_macro:.3f}")
    print(f"  F1 (macro)       : {m.f1_macro:.3f}")


def show_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str
) -> None:
    """
    Confusion matrix plot.

    Rows = true labels
    Columns = predicted labels
    Diagonal = correct predictions
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def entropy_from_proba(proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    TODO #5 (optional):
    Compute entropy per row:

        H = -sum(p_i * log(p_i))

    Entropy is higher when the model is uncertain (probabilities spread out),
    and lower when the model is confident (one probability close to 1).

    If you do not implement this, it returns zeros and the uncertainty block won't print.
    """
    # HINT:
    # p = np.clip(proba, eps, 1.0)
    # return -np.sum(p * np.log(p), axis=1)

    return np.zeros(proba.shape[0], dtype=float)


# =============================================================================
# FINAL EXERCISE — Extra classifier (optional)
# =============================================================================
def train_your_extra_classifier(X_train: np.ndarray, y_train: np.ndarray):
    """
    FINAL EXERCISE:
    Implement ONE more classifier and return it (fitted).

    Options (choose one):
      - DecisionTreeClassifier
      - GaussianNB
      - SVC(probability=True)

    Students should:
    1) import their chosen classifier
    2) create model = Classifier(...)
    3) model.fit(X_train, y_train)
    4) return model
    """
    raise NotImplementedError("Implement your chosen extra classifier here.")


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main() -> None:
    """
    Run the entire lab pipeline in order:
    EDA plots first, then corruption/cleaning, then model training/evaluation.
    """
    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    df = load_iris_as_dataframe()

    # Identify which columns are features: in Iris, they end with "(cm)"
    feature_cols = [c for c in df.columns if c.endswith("(cm)")]

    # Define class names (in correct order)
    class_names = ["setosa", "versicolor", "virginica"]

    # -------------------------------------------------------------------------
    # EDA: prints + plots BEFORE we do any machine learning
    # -------------------------------------------------------------------------
    basic_exploration(df, feature_cols)

    plot_feature_histograms(df, feature_cols)
    plot_boxplots(df, feature_cols)
    scatter_petal(df)
    plot_scatter_matrix(df, feature_cols)
    plot_correlation_heatmap(df, feature_cols)

    # -------------------------------------------------------------------------
    # Corrupt the dataset (students implement TODO #1 and TODO #2)
    # -------------------------------------------------------------------------
    df_bad = inject_missing_values(df, feature_cols, missing_rate=0.05)
    df_bad = inject_outliers(df_bad, feature_cols, n_outliers=6)

    print("\n" + "=" * 80)
    print("STEP 3 — AFTER CORRUPTION")
    print("=" * 80)
    print("Missing values per feature column (after corruption):")
    print(df_bad[feature_cols].isna().sum())

    # -------------------------------------------------------------------------
    # Clean the dataset (students implement TODO #3 and TODO #4)
    # -------------------------------------------------------------------------
    df_clean = impute_missing_median(df_bad, feature_cols)
    df_clean = clip_outliers_iqr(df_clean, feature_cols, k=1.5)

    print("\n" + "=" * 80)
    print("STEP 4 — AFTER CLEANING")
    print("=" * 80)
    print("Missing values per feature column (after cleaning):")
    print(df_clean[feature_cols].isna().sum())

    # -------------------------------------------------------------------------
    # Split and scale
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test, _ = make_train_test(df_clean, feature_cols, test_size=0.20)

    # -------------------------------------------------------------------------
    # Train models
    # -------------------------------------------------------------------------
    # Dummy baseline: simplest benchmark
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
    dummy.fit(X_train, y_train)

    # kNN model
    knn = train_knn(X_train, y_train, k=5)

    # Logistic Regression model
    logreg = train_logreg(X_train, y_train)

    # -------------------------------------------------------------------------
    # Evaluate models + show plots
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 7 — MODEL EVALUATION")
    print("=" * 80)

    results: Dict[str, Metrics] = {}

    # We store models in a list so we can loop without repeating code
    models: List[Tuple[str, Any]] = [
        ("Dummy (most frequent)", dummy),
        ("kNN (k=5)", knn),
        ("Logistic Regression", logreg),
    ]

    for name, model in models:
        # Predict labels for test set
        y_pred = model.predict(X_test)

        # Compute metrics and store them
        m = compute_metrics(y_test, y_pred)
        results[name] = m

        # Print metrics nicely
        print_metrics_nicely(name, m)

        # Confusion matrix plot
        show_confusion_matrix(y_test, y_pred, class_names, title=f"Confusion Matrix — {name}")

        # Optional uncertainty block (only if model supports predict_proba)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            ent = entropy_from_proba(proba)

            # Until TODO #5 is implemented, ent will be all zeros and this won't print.
            if np.any(ent > 0):
                print("  Entropy (uncertainty) — top 5 most uncertain test samples:")
                top_idx = np.argsort(-ent)[:5]
                for i in top_idx:
                    print(f"    test_idx={i:3d} entropy={ent[i]:.4f} proba={proba[i]} true={class_names[y_test[i]]}")

    # -------------------------------------------------------------------------
    # Final summary table for quick comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (macro metrics)")
    print("=" * 80)
    for name, m in results.items():
        print(
            f"{name:24s} | "
            f"acc={m.accuracy:.3f}  "
            f"prec={m.precision_macro:.3f}  "
            f"rec={m.recall_macro:.3f}  "
            f"f1={m.f1_macro:.3f}"
        )

    # -------------------------------------------------------------------------
    # Optional: show one example prediction
    # -------------------------------------------------------------------------
    idx = 0  # first test sample
    x_example = X_test[idx].reshape(1, -1)

    print("\n" + "=" * 80)
    print("EXAMPLE PREDICTION (first test sample)")
    print("=" * 80)
    print("True class:", class_names[y_test[idx]])
    print("Dummy predicts:", class_names[dummy.predict(x_example)[0]])
    print("kNN predicts  :", class_names[knn.predict(x_example)[0]])
    print("LogReg predicts:", class_names[logreg.predict(x_example)[0]])

    # Show probabilities for Logistic Regression (confidence)
    if hasattr(logreg, "predict_proba"):
        p = logreg.predict_proba(x_example)[0]
        print("LogReg probabilities:", {class_names[i]: float(p[i]) for i in range(len(class_names))})

    # -------------------------------------------------------------------------
    # Final exercise: extra classifier (students can uncomment after implementing)
    # -------------------------------------------------------------------------
    # extra = train_your_extra_classifier(X_train, y_train)
    # y_pred_extra = extra.predict(X_test)
    # m_extra = compute_metrics(y_test, y_pred_extra)
    # print_metrics_nicely("Your Extra Classifier", m_extra)
    # show_confusion_matrix(y_test, y_pred_extra, class_names, "Confusion Matrix — Extra Classifier")


# Standard Python pattern: only run main() if this file is executed directly.
if __name__ == "__main__":
    main()
