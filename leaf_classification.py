# ============================================================
# IMPORTS
# ============================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from scipy.stats import f_oneway, ttest_rel

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_val_predict
)

from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ============================================================
# SETTINGS
# ============================================================

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("leaf_data.csv")

X = df.iloc[:, :-1]

# Encode target labels for XGBoost compatibility
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(df.iloc[:, -1])

# ============================================================
# DATASET SUMMARY
# ============================================================

dataset_info = {
    "Number of Samples": [df.shape[0]],
    "Number of Features": [X.shape[1]],
    "Number of Classes": [len(np.unique(y))]
}

dataset_df = pd.DataFrame(dataset_info)

dataset_df.to_csv(
    os.path.join(output_dir, "dataset_summary.csv"),
    index=False
)

print("\n=== DATASET SUMMARY ===")
print(dataset_df)

# ============================================================
# CLASS DISTRIBUTION PLOT
# ============================================================

plt.figure(figsize=(8, 5))

# Convert encoded labels to Pandas Series
pd.Series(y).value_counts().sort_index().plot(
    kind='bar'
)

plt.title("Class Distribution")
plt.xlabel("Encoded Class Label")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "class_distribution.png"
    ),
    dpi=300,
    bbox_inches='tight'
)

plt.close()

# ============================================================
# FEATURE SELECTION
# ============================================================

feature_sets = {
    "All": "passthrough",
    "RFE": "dynamic"
}

# ============================================================
# MODELS
# ============================================================

models = {

    "MLP": (
        MLPClassifier(
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=15,
            random_state=42
        ),
        True
    ),

    "RF": (
        RandomForestClassifier(
            random_state=42
        ),
        False
    ),

    "KNN": (
        KNeighborsClassifier(),
        True
    ),

    "SVM": (
        SVC(
            probability=True,
            random_state=42
        ),
        True
    ),

    "XGB": (
        XGBClassifier(
            eval_metric='mlogloss',
            random_state=42,
            tree_method="hist",
            n_jobs=1
        ),
        False
    )
}

# ============================================================
# DYNAMIC RFE GENERATOR
# ============================================================
def get_rfe_estimator(model_name):

    if model_name == "SVM":

        return SVC(
            kernel="linear",
            random_state=42
        )

    elif model_name == "RF":

        return RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=1
        )

    elif model_name == "XGB":

        return XGBClassifier(
            eval_metric="mlogloss",
            random_state=42,
            n_estimators=100
        )

    elif model_name == "KNN":

        return RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=1
        )

    elif model_name == "MLP":

        return RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=1
        )

# ============================================================
# PARAMETER GRID
# ============================================================

def get_param_grid(model_name, fs_name):

    param_grid = {}

    # --------------------------------------------------------
    # Feature Selection Parameters
    # --------------------------------------------------------

    if fs_name == "RFE":

        param_grid["fs__n_features_to_select"] = [5, 6, 8]

    # --------------------------------------------------------
    # MLP PARAMETERS
    # --------------------------------------------------------

    if model_name == "MLP":

        param_grid.update({


            "model__hidden_layer_sizes": [
                (8,),
                (12,),
                (16,),
                (8, 8)
            ],

            "model__activation": [
                "relu",
                "tanh"
            ],

            "model__alpha": np.logspace(-3, -1, 3),

            "model__learning_rate_init": [
                0.001,
                0.01
            ],

            "model__learning_rate": [
                "adaptive"
            ]
        })

    # --------------------------------------------------------
    # RANDOM FOREST PARAMETERS
    # --------------------------------------------------------

    elif model_name == "RF":

        param_grid.update({

            "model__n_estimators": [
                50,
                100,
                200
            ],

            "model__max_depth": [
                5,
                10,
                None
            ],

            "model__min_samples_split": [
                2,
                5
            ],

            "model__min_samples_leaf": [
                2,
                4
            ],

            "model__max_features": [
                "sqrt"
            ]
        })

    # --------------------------------------------------------
    # KNN PARAMETERS
    # --------------------------------------------------------

    elif model_name == "KNN":

        param_grid.update({

            "model__n_neighbors": [
                3,
                5,
                7,
                9
            ],

            "model__weights": [
                "uniform","distance"
            ],

            "model__p": [
                1,
                2
            ],
            "model__metric": [
                "euclidean", "manhattan"
            ]
        })

    # --------------------------------------------------------
    # SVM PARAMETERS
    # --------------------------------------------------------

    elif model_name == "SVM":

        param_grid.update({

            "model__C": [
                0.1,
                1,
                10,
                100
            ],

            "model__kernel": [
                "linear",
                "rbf"
            ],

            "model__gamma": [
                "scale",
                "auto"
            ]
        })

    # --------------------------------------------------------
    # XGBOOST PARAMETERS
    # --------------------------------------------------------

    elif model_name == "XGB":

        param_grid.update({

            "model__n_estimators": [
                50,
                100
            ],

            "model__max_depth": [
                3,
                6,
                9
            ],

            "model__learning_rate": [
                0.01,
                0.1,
                0.3
            ],

            "model__subsample": [
                0.8,
                1.0
            ]
        })

    return param_grid

# ============================================================
# PIPELINE FUNCTION
# ============================================================

def build_pipeline(fs, model, scale):

    steps = []

    if scale:
        steps.append(
            ("scaler", StandardScaler())
        )

    steps.append(
        ("fs", fs)
    )

    steps.append(
        ("model", model)
    )

    return Pipeline(steps)

# ============================================================
# CROSS VALIDATION
# ============================================================

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

inner_cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

# ============================================================
# TRAINING LOOP
# ============================================================

results = {}

print("\n=== TRAINING MODELS ===")

for model_name, (model, scale) in models.items():

    for fs_name, fs in feature_sets.items():

        if fs_name == "RFE":

            fs = RFE(
                estimator=get_rfe_estimator(
                    model_name
                )
            )
        print(f"\nRunning: {model_name} - {fs_name}")

        pipe = build_pipeline(fs, model, scale)

        param_grid = get_param_grid(
            model_name,
            fs_name
        )

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="f1_weighted",
            n_jobs=1
        )

        scores = cross_val_score(
            estimator=grid,
            X=X,
            y=y,
            cv=outer_cv,
            scoring="f1_weighted"
        )

        results[f"{model_name}-{fs_name}"] = scores

# ============================================================
# SAVE RAW CV SCORES
# ============================================================

scores_df = pd.DataFrame(results)

scores_df.to_csv(
    os.path.join(output_dir, "cv_scores.csv"),
    index=False
)

# ============================================================
# RESULTS SUMMARY
# ============================================================

summary = []

print("\n=== NESTED CV RESULTS ===")

for key, scores in results.items():

    mean = np.mean(scores)
    std = np.std(scores)

    ci = 1.96 * std / np.sqrt(len(scores))

    summary.append([
        key,
        mean,
        std,
        ci
    ])

    print(
        f"{key}: "
        f"Mean={mean:.4f}, "
        f"Std={std:.4f}, "
        f"CI=±{ci:.4f}"
    )

results_df = pd.DataFrame(
    summary,
    columns=[
        "Model",
        "Mean",
        "Std",
        "CI"
    ]
)

results_df = results_df.sort_values(
    by="Mean",
    ascending=False
)

results_df.to_csv(
    os.path.join(output_dir, "results_summary.csv"),
    index=False
)

# ============================================================
# ANOVA + T-TESTS
# ============================================================

with open(
    os.path.join(output_dir, "stat_tests.txt"),
    "w"
) as f:

    f.write("=== ANOVA TEST ===\n")

    f_stat, p_val = f_oneway(*results.values())

    f.write(f"F-stat: {f_stat}\n")
    f.write(f"p-value: {p_val}\n\n")

    f.write("=== PAIRED T-TEST ===\n")

    best_key = results_df.iloc[0]["Model"]
    best_scores = results[best_key]

    for key, scores in results.items():

        if key != best_key:

            t_stat, p = ttest_rel(
                best_scores,
                scores
            )

            f.write(
                f"{best_key} vs {key} "
                f"-> p={p:.4f}\n"
            )

# ============================================================
# BAR PLOT
# ============================================================

plt.figure(figsize=(10, 5))

plt.bar(
    results_df["Model"],
    results_df["Mean"],
    yerr=results_df["CI"],
    capsize=5
)

plt.title("Model Performance (Nested CV)")
plt.ylabel("Weighted F1 Score")
plt.xlabel("Models")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "bar_plot.png"),
    dpi=300,
    bbox_inches='tight'
)

plt.close()

# ============================================================
# BOX PLOT
# ============================================================

plt.figure(figsize=(10, 5))

plt.boxplot(
    results.values(),
    labels=results.keys()
)

plt.title("Model Stability Across Folds")
plt.ylabel("Weighted F1 Score")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "box_plot.png"),
    dpi=300,
    bbox_inches='tight'
)

plt.close()

# ============================================================
# BEST MODEL
# ============================================================

best_key = results_df.iloc[0]["Model"]

print(f"\nBest Model: {best_key}")

best_model_name, best_fs_name = best_key.split("-")

model, scale = models[best_model_name]

if best_fs_name == "RFE":

    fs = RFE(
        estimator=get_rfe_estimator(
            best_model_name
        )
    )

else:

    fs = "passthrough"

best_pipe = build_pipeline(
    fs,
    model,
    scale
)

param_grid = get_param_grid(
    best_model_name,
    best_fs_name
)

grid = GridSearchCV(
    estimator=best_pipe,
    param_grid=param_grid,
    cv=inner_cv,
    scoring="f1_weighted",
    n_jobs=1
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

y_pred = cross_val_predict(
    estimator=grid,
    X=X,
    y=y,
    cv=outer_cv
)

cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(cmap="Blues")

plt.title(
    f"Confusion Matrix - {best_key}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "confusion_matrix.png"),
    dpi=300,
    bbox_inches='tight'
)

plt.close()

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y,
    y_pred
)

with open(
    os.path.join(output_dir, "classification_report.txt"),
    "w"
) as f:

    f.write(report)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

if best_model_name == "RF":

    grid.fit(X, y)

    best_estimator = grid.best_estimator_

    rf_model = best_estimator.named_steps["model"]

importances = rf_model.feature_importances_

if best_fs_name == "RFE":

    selector = best_estimator.named_steps["fs"]

    feature_names = X.columns[
        selector.support_
    ]

else:

    feature_names = X.columns

    importance_df = pd.DataFrame({

        "Feature": feature_names,
        "Importance": importances

    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    importance_df.to_csv(
        os.path.join(
            output_dir,
            "feature_importance.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE PLOT
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.gca().invert_yaxis()

    plt.title(
        "Random Forest Feature Importance"
    )

    plt.xlabel("Importance Score")
    plt.ylabel("Features")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "feature_importance.png"
        ),
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


# ============================================================
# SAVE REQUIREMENTS FILE
# ============================================================

os.system("pip freeze > requirements.txt")

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n===================================")
print("ALL OUTPUTS SAVED SUCCESSFULLY")
print("===================================")

print(f"\nOutputs folder: {output_dir}/")

print("\nGenerated Files:")
print("- dataset_summary.csv")
print("- cv_scores.csv")
print("- results_summary.csv")
print("- stat_tests.txt")
print("- classification_report.txt")
print("- class_distribution.png")
print("- bar_plot.png")
print("- box_plot.png")
print("- confusion_matrix.png")
print("- feature_importance.png")
print("- feature_importance.csv")
print("- requirements.txt")