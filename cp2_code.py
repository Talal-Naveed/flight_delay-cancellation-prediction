# ===============================================
# PROJECT CODE ORGANIZATION
# ===============================================
# 1. Imports
# 2. Output Configuration
# 3. Data Loading
# 4. Data Cleaning and Feature Engineering
# 5. Missing-Value Handling
# 6. Target Definition and Leakage Exclusion
# 7. Train-Test Partitioning
# 8. Frequency Encoding and Imputation
# 9. Descriptive Tables
# 10. Exploratory Data Analysis
# 11. Preprocessing for Classification
# 12. Evaluation Helper Functions
# 13. Arrival Delay Regression
# 14. Departure Delay Regression
# 15. Arrival and Departure Delay Classification
# 16. Cancellation Classification
# 17. Visualisations and Output Export
# ================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier

from xgboost import XGBRegressor, XGBClassifier

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, roc_auc_score
)


# =====================
# Output folders
# =====================

OUT_DIR = r"C:\Users\tnave\OneDrive\Desktop\trial_cp2_outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")

# =====================
# Load Data
# =====================

df = pd.read_csv(r"C:\Users\tnave\OneDrive\Desktop\flights_sample_3m.csv")


# ===============================
# Cleaning + Feature Engineering
# ===============================

df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])

df['FL_YEAR'] = df['FL_DATE'].dt.year
df['FL_MONTH'] = df['FL_DATE'].dt.month
df['FL_DAY'] = df['FL_DATE'].dt.day

df = df.drop(columns=['FL_DATE'])

df = df.drop(columns=[
    'AIRLINE_DOT',
    'AIRLINE_CODE',
    'DOT_CODE',
    'FL_NUMBER',
    'ORIGIN_CITY',
    'DEST_CITY',
])


def hhmm_to_minutes(x):
    return (x // 100) * 60 + (x % 100)


df['SCHEDULE_DEP_TIME(MINS)'] = hhmm_to_minutes(df['CRS_DEP_TIME'])
df['SCHEDULE_ARR_TIME(MINS)'] = hhmm_to_minutes(df['CRS_ARR_TIME'])
df = df.drop(columns=['CRS_DEP_TIME', 'CRS_ARR_TIME'])

df['DEP_TIME(MINS)'] = hhmm_to_minutes(df['DEP_TIME'])
df = df.drop(columns=['DEP_TIME'])

df['ARR_TIME(MINS)'] = hhmm_to_minutes(df['ARR_TIME'])
df = df.drop(columns=['ARR_TIME'])

df['WHEELS_OFF(MINS)'] = hhmm_to_minutes(df['WHEELS_OFF'])
df = df.drop(columns=['WHEELS_OFF'])

df['WHEELS_ON(MINS)'] = hhmm_to_minutes(df['WHEELS_ON'])
df = df.drop(columns=['WHEELS_ON'])

df = df.rename(columns={
    'CRS_ELAPSED_TIME': 'SCHEDULED_DURATION(MINS)',
    'ELAPSED_TIME': 'FLIGHT_DURATION(MINS)',
    'DISTANCE': 'DISTANCE_(MILES)'
})


# ========================
# Missing Values
# ========================

df['CANCELLED'] = df['CANCELLED'].fillna(0)

delay_cols = [
    'DIVERTED',
    'DELAY_DUE_CARRIER',
    'DELAY_DUE_WEATHER',
    'DELAY_DUE_NAS',
    'DELAY_DUE_SECURITY',
    'DELAY_DUE_LATE_AIRCRAFT'
]

df[delay_cols] = df[delay_cols].fillna(0)


# ====================
# Target Sets
# ====================
df_delay = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)].copy()

DROP_ARR = ['ARR_DELAY', 'DEP_DELAY', 'ARR_TIME(MINS)', 'DEP_TIME(MINS)',
            'WHEELS_OFF(MINS)', 'WHEELS_ON(MINS)', 'TAXI_IN', 'TAXI_OUT',
            'FLIGHT_DURATION(MINS)', 'AIR_TIME', 'CANCELLED', 'DIVERTED',
            'CANCELLATION_CODE'] + delay_cols
X_arr = df_delay.drop(columns=DROP_ARR)
Y_arr = df_delay['ARR_DELAY']

DROP_DEP = ['DEP_DELAY', 'ARR_DELAY', 'ARR_TIME(MINS)', 'DEP_TIME(MINS)',
            'WHEELS_OFF(MINS)', 'WHEELS_ON(MINS)', 'TAXI_IN', 'TAXI_OUT',
            'FLIGHT_DURATION(MINS)', 'AIR_TIME', 'CANCELLED', 'DIVERTED',
            'CANCELLATION_CODE'] + delay_cols
X_dep = df_delay.drop(columns=DROP_DEP)
Y_dep = df_delay['DEP_DELAY']

arr_mask = Y_arr.notna()
X_arr, Y_arr = X_arr[arr_mask], Y_arr[arr_mask]
dep_mask = Y_dep.notna()
X_dep, Y_dep = X_dep[dep_mask], Y_dep[dep_mask]

DROP_CANCEL = ['CANCELLED', 'CANCELLATION_CODE', 'ARR_DELAY', 'DEP_DELAY',
               'ARR_TIME(MINS)', 'AIR_TIME', 'DEP_TIME(MINS)',
               'WHEELS_OFF(MINS)', 'WHEELS_ON(MINS)', 'FLIGHT_DURATION(MINS)',
               'TAXI_OUT', 'TAXI_IN', 'DIVERTED'] + delay_cols
X_cancel = df.drop(columns=DROP_CANCEL)
Y_cancel = df['CANCELLED']

# ==========================
# Data Partition (80%/20%)
# ==========================

X_train_arr, X_test_arr, Y_train_arr, Y_test_arr = train_test_split(
    X_arr, Y_arr, test_size=0.2, random_state=66
)

X_train_dep, X_test_dep, Y_train_dep, Y_test_dep = train_test_split(
    X_dep, Y_dep, test_size=0.2, random_state=66
)

X_train_cancel, X_test_cancel, Y_train_cancel, Y_test_cancel = train_test_split(
    X_cancel, Y_cancel, test_size=0.2, random_state=66, stratify=Y_cancel
)


# ====================
# Frequency Encoding
# ====================

def frequency_encode_train_test(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()
    cat_cols = X_train.select_dtypes(include=['object']).columns
    for col in cat_cols:
        freq = X_train[col].value_counts()
        X_train[col] = X_train[col].map(freq).fillna(0)
        X_test[col] = X_test[col].map(freq).fillna(0)
    return X_train, X_test


X_train_arr, X_test_arr = frequency_encode_train_test(X_train_arr, X_test_arr)
X_train_dep, X_test_dep = frequency_encode_train_test(X_train_dep, X_test_dep)
X_train_cancel, X_test_cancel = frequency_encode_train_test(X_train_cancel, X_test_cancel)

# ==================
# Imputation
# ==================

def apply_imputation(X_train, X_test):
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    imputer = SimpleImputer(strategy="median")
    X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])
    return X_train, X_test


X_train_arr, X_test_arr = apply_imputation(X_train_arr, X_test_arr)
X_train_dep, X_test_dep = apply_imputation(X_train_dep, X_test_dep)
X_train_cancel, X_test_cancel = apply_imputation(X_train_cancel, X_test_cancel)


# Keep an unscaled copy of cancel features (for feature importance / column names)
cancel_feature_names = X_train_cancel.columns.tolist()



# ================================
# Scaling for Logistic Regression
# ================================

scaler = StandardScaler()
X_train_cancel_scaled = scaler.fit_transform(X_train_cancel)
X_test_cancel_scaled = scaler.transform(X_test_cancel)


# ======================================================
# HELPER: full metrics for a fitted regressor (train+test)
# ======================================================
def regression_metrics(model, X_train, Y_train, X_test, Y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    return {
        "Train_RMSE": np.sqrt(mean_squared_error(Y_train, train_pred)),
        "Train_MAE": mean_absolute_error(Y_train, train_pred),
        "Train_R2": r2_score(Y_train, train_pred),
        "Test_RMSE": np.sqrt(mean_squared_error(Y_test, test_pred)),
        "Test_MAE": mean_absolute_error(Y_test, test_pred),
        "Test_R2": r2_score(Y_test, test_pred),
    }, test_pred


def classification_metrics(model, X_train, Y_train, X_test, Y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    metrics = {
        "Train_Accuracy": accuracy_score(Y_train, train_pred),
        "Test_Accuracy": accuracy_score(Y_test, test_pred),
        "Test_Precision": precision_score(Y_test, test_pred),
        "Test_Recall": recall_score(Y_test, test_pred),
        "Test_F1": f1_score(Y_test, test_pred),
    }
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        metrics["Test_ROC_AUC"] = roc_auc_score(Y_test, proba)
    return metrics, test_pred


def print_metrics_block(model_name, metrics):
    """Print one model's metrics as clean, separately-labeled lines
    instead of a single-line raw dict dump."""
    print(f"\n  {model_name}")
    for metric_name, value in metrics.items():
        print(f"    {metric_name:<15} {value:.6f}")


# ============================================================
# ARRIVAL MODELS
# (static hold-out only: fit on train, evaluate once on test)
# ============================================================

arr_models = {
    "Linear": LinearRegression(),

    "Tree": DecisionTreeRegressor(
                random_state=66
            ),

    "GBR": GradientBoostingRegressor(
                random_state=66
            ),

    "Random Forest": RandomForestRegressor(
                        n_estimators=50,
                        max_depth=15,
                        n_jobs=-1,
                        random_state=66
                     ),

    "XGBoost": XGBRegressor(
                   random_state=66,
                   objective="reg:squarederror"
               )
}

arr_results = {}
arr_preds = {}
arr_fitted = {}

print("\n=== ARRIVAL DELAY — full metrics ===")
for name, model in arr_models.items():
    model.fit(X_train_arr, Y_train_arr)
    metrics, test_pred = regression_metrics(model, X_train_arr, Y_train_arr, X_test_arr, Y_test_arr)
    arr_results[name] = metrics
    arr_preds[name] = test_pred
    arr_fitted[name] = model
    print_metrics_block(name, metrics)

arr_results_df = pd.DataFrame(arr_results).T
arr_results_df.to_csv(os.path.join(OUT_DIR, "arrival_model_metrics.csv"))
best_arr_name = arr_results_df["Test_R2"].idxmax()
best_arr_model = arr_fitted[best_arr_name]
print(f"\nBest arrival model by Test R2: {best_arr_name}")

# =========================
# DEPARTURE MODELS
# =========================

dep_models = {
    "Linear": LinearRegression(),

    "Tree": DecisionTreeRegressor(
                random_state=66
            ),

    "GBR": GradientBoostingRegressor(
               random_state=66
           ),

    "Random Forest": RandomForestRegressor(
                         n_estimators=50,
                         max_depth=15,
                         n_jobs=-1,
                         random_state=66
                     ),

    "XGBoost": XGBRegressor(
                   random_state=66,
                   objective="reg:squarederror"
               )
}

dep_results = {}
dep_preds = {}
dep_fitted = {}

print("\n=== DEPARTURE DELAY — full metrics ===")
for name, model in dep_models.items():
    model.fit(X_train_dep, Y_train_dep)
    metrics, test_pred = regression_metrics(model, X_train_dep, Y_train_dep, X_test_dep, Y_test_dep)
    dep_results[name] = metrics
    dep_preds[name] = test_pred
    dep_fitted[name] = model
    print_metrics_block(name, metrics)

dep_results_df = pd.DataFrame(dep_results).T
dep_results_df.to_csv(os.path.join(OUT_DIR, "departure_model_metrics.csv"))
best_dep_name = dep_results_df["Test_R2"].idxmax()
best_dep_model = dep_fitted[best_dep_name]
print(f"\nBest departure model by Test R2: {best_dep_name}")

# ============================================================
# ARRIVAL & DEPARTURE DELAY — CLASSIFICATION (>15 min = delayed)
# Same standard threshold used across most published flight-delay
# classification literature, derived from the same regression targets
# already computed above (no re-cleaning/re-encoding needed).
# ============================================================

DELAY_THRESHOLD = 15  # minutes; matches the industry/literature standard

# --- Binary labels derived from existing regression targets ---
Y_train_arr_bin = (Y_train_arr > DELAY_THRESHOLD).astype(int)
Y_test_arr_bin  = (Y_test_arr  > DELAY_THRESHOLD).astype(int)
Y_train_dep_bin = (Y_train_dep > DELAY_THRESHOLD).astype(int)
Y_test_dep_bin  = (Y_test_dep  > DELAY_THRESHOLD).astype(int)

print("\nArrival delay classification balance (Train):")
print(Y_train_arr_bin.value_counts(normalize=True).round(4) * 100, "%")
print("\nDeparture delay classification balance (Train):")
print(Y_train_dep_bin.value_counts(normalize=True).round(4) * 100, "%")

# --- Scale features (same X_train_arr/X_train_dep used for regression,
#     scaled separately here since Logistic Regression needs it) ---
scaler_arr_cls = StandardScaler()
X_train_arr_scaled = scaler_arr_cls.fit_transform(X_train_arr)
X_test_arr_scaled  = scaler_arr_cls.transform(X_test_arr)

scaler_dep_cls = StandardScaler()
X_train_dep_scaled = scaler_dep_cls.fit_transform(X_train_dep)
X_test_dep_scaled  = scaler_dep_cls.transform(X_test_dep)

# --- Sample weights for GB (same balanced logic as cancellation) ---
gb_sample_weights_arr = compute_sample_weight(class_weight="balanced", y=Y_train_arr_bin)
gb_sample_weights_dep = compute_sample_weight(class_weight="balanced", y=Y_train_dep_bin)


def build_delay_classifiers(y_train_bin):
    """Fresh model instances per task so fitted state isn't shared."""
    pos_weight = (y_train_bin.value_counts().loc[0] / y_train_bin.value_counts().loc[1])
    return {
        "Logistic": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Tree": DecisionTreeClassifier(class_weight="balanced", random_state=66),
        "GB": GradientBoostingClassifier(random_state=66),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, max_depth=15, class_weight="balanced",
            n_jobs=-1, random_state=66
        ),
        "XGBoost": XGBClassifier(
            random_state=66, eval_metric="logloss", scale_pos_weight=pos_weight
        ),
    }


def run_delay_classification(task_name, X_train_scaled, Y_train_bin,
                              X_test_scaled, Y_test_bin, gb_weights, out_dir):
    models = build_delay_classifiers(Y_train_bin)
    results, preds, fitted = {}, {}, {}

    print(f"\n=== {task_name} DELAY CLASSIFICATION (>{DELAY_THRESHOLD} min) — full metrics ===")
    for name, model in models.items():
        if name == "GB":
            model.fit(X_train_scaled, Y_train_bin, sample_weight=gb_weights)
        else:
            model.fit(X_train_scaled, Y_train_bin)
        metrics, test_pred = classification_metrics(
            model, X_train_scaled, Y_train_bin, X_test_scaled, Y_test_bin
        )
        results[name] = metrics
        preds[name] = test_pred
        fitted[name] = model
        print_metrics_block(name, metrics)

    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(out_dir, f"{task_name.lower()}_classification_metrics.csv"))
    best_name = results_df["Test_F1"].idxmax()
    best_model = fitted[best_name]
    print(f"\nBest {task_name} classification model by Test F1: {best_name}")
    print(f"\nClassification Report ({task_name}, {best_name}):")
    print(classification_report(Y_test_bin, preds[best_name]))

    return results_df, preds, fitted, best_name, best_model


arr_cls_results_df, arr_cls_preds, arr_cls_fitted, best_arr_cls_name, best_arr_cls_model = \
    run_delay_classification("ARRIVAL", X_train_arr_scaled, Y_train_arr_bin,
                              X_test_arr_scaled, Y_test_arr_bin,
                              gb_sample_weights_arr, OUT_DIR)

dep_cls_results_df, dep_cls_preds, dep_cls_fitted, best_dep_cls_name, best_dep_cls_model = \
    run_delay_classification("DEPARTURE", X_train_dep_scaled, Y_train_dep_bin,
                              X_test_dep_scaled, Y_test_dep_bin,
                              gb_sample_weights_dep, OUT_DIR)


# =========================
# CANCELLATION MODELS
# =========================

cancel_models = {
    "Logistic": LogisticRegression(max_iter=3000, class_weight="balanced"),
    "Tree": DecisionTreeClassifier(class_weight="balanced", random_state=66),
    "GB": GradientBoostingClassifier(random_state=66),  # no class_weight param -> uses sample_weight at fit time instead
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=15, class_weight="balanced", n_jobs=-1, random_state=66),
    "XGBoost": XGBClassifier(
        random_state=66, eval_metric="logloss",
        scale_pos_weight=(Y_train_cancel.value_counts().loc[0] / Y_train_cancel.value_counts().loc[1])
    ),
}

# Sample weights for GB: same 'balanced' logic sklearn applies internally
# for class_weight="balanced" on the other models, computed manually since
# GradientBoostingClassifier's constructor has no class_weight argument.
gb_sample_weights = compute_sample_weight(class_weight="balanced", y=Y_train_cancel)

cancel_results = {}
cancel_preds = {}
cancel_fitted = {}

print("\n=== CANCELLATION — full metrics ===")
for name, model in cancel_models.items():
    if name == "GB":
        model.fit(X_train_cancel_scaled, Y_train_cancel, sample_weight=gb_sample_weights)
    else:
        model.fit(X_train_cancel_scaled, Y_train_cancel)
    metrics, test_pred = classification_metrics(model, X_train_cancel_scaled, Y_train_cancel, X_test_cancel_scaled, Y_test_cancel)
    cancel_results[name] = metrics
    cancel_preds[name] = test_pred
    cancel_fitted[name] = model
    print_metrics_block(name, metrics)

cancel_results_df = pd.DataFrame(cancel_results).T
cancel_results_df.to_csv(os.path.join(OUT_DIR, "cancellation_model_metrics.csv"))
best_cancel_name = cancel_results_df["Test_F1"].idxmax()
best_cancel_model = cancel_fitted[best_cancel_name]
print(f"\nBest cancellation model by Test F1: {best_cancel_name}")

print("\nClassification Report (best cancellation model):")
print(classification_report(Y_test_cancel, cancel_preds[best_cancel_name]))
print("Confusion Matrix:")
print(confusion_matrix(Y_test_cancel, cancel_preds[best_cancel_name]))


# ============================================================
# TABLE 1 — Dataset Descriptive Statistics
# ============================================================
raw_col_count = 32  
clean_col_count = len(df.columns)

desc_stats = {
    "Metric": [
        "Total flights (post date filter)",
        "Date range",
        "Raw columns (source)",
        "Columns after cleaning/engineering",
        "Cancelled flights",
        "Diverted flights",
        "Delay-eligible flights (not cancelled/diverted)",
        "Arrival Delay — train rows",
        "Arrival Delay — test rows",
        "Departure Delay — train rows",
        "Departure Delay — test rows",
        "Cancellation — train rows",
        "Cancellation — test rows",
        "Cancellation rate (full dataset)",
        "Arrival Delay skewness",
        "Departure Delay skewness",
    ],
    "Value": [
        f"{len(df):,}",
        "2019-01-01 to 2023-08-31",
        raw_col_count,
        clean_col_count,
        f"{int(df['CANCELLED'].sum()):,}",
        f"{int(df['DIVERTED'].sum()):,}",
        f"{len(df_delay):,}",
        f"{len(X_train_arr):,}",
        f"{len(X_test_arr):,}",
        f"{len(X_train_dep):,}",
        f"{len(X_test_dep):,}",
        f"{len(X_train_cancel):,}",
        f"{len(X_test_cancel):,}",
        f"{df['CANCELLED'].mean()*100:.2f}%",
        f"{skew(df_delay['ARR_DELAY'].dropna()):.2f}",
        f"{skew(df_delay['DEP_DELAY'].dropna()):.2f}",
    ],
}
desc_stats_df = pd.DataFrame(desc_stats)
desc_stats_df.to_csv(os.path.join(OUT_DIR, "table1_dataset_descriptive_statistics.csv"), index=False)
print("\n=== TABLE 1: Dataset Descriptive Statistics ===")
for metric, value in zip(desc_stats_df["Metric"], desc_stats_df["Value"]):
    print(f"{metric:<50} {value}")

# Render as a clean table image for direct figure use in the report
fig, ax = plt.subplots(figsize=(9, 7))
ax.axis('off')
tbl = ax.table(cellText=desc_stats_df.values, colLabels=desc_stats_df.columns,
                cellLoc='left', colLoc='left', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.5)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#E6F1FB')
ax.set_title("Table 1: Dataset Descriptive Statistics", fontsize=12, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "table1_dataset_descriptive_statistics.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)


# ============================================================
# TABLE 2 — Leakage Exclusion Table
# ============================================================
leakage_table = pd.DataFrame({
    "Excluded column": [
        "DEP_DELAY", "ARR_TIME(MINS)", "DEP_TIME(MINS)", "WHEELS_OFF(MINS)",
        "WHEELS_ON(MINS)", "TAXI_IN", "TAXI_OUT", "FLIGHT_DURATION(MINS)",
        "AIR_TIME", "DELAY_DUE_CARRIER / WEATHER / NAS / SECURITY / LATE_AIRCRAFT",
        "CANCELLATION_CODE",
    ],
    "Reason for exclusion": [
        "Only known after the flight departs; near-deterministic predictor of ARR_DELAY (target for arrival task)",
        "Actual arrival time — only known after the flight lands",
        "Actual departure time — only known after departure occurs",
        "Recorded once the aircraft leaves the gate",
        "Recorded once the aircraft lands",
        "Ground time after landing — post-arrival only",
        "Ground time before takeoff — only known once departure has begun",
        "Total flight duration — only fully known after landing",
        "Time airborne — only known after landing",
        "Delay-cause breakdown is only assigned after a delay has occurred and been categorized",
        "Only populated for already-cancelled flights; would leak the cancellation outcome itself",
    ],
    "Excluded from task(s)": [
        "Arrival Delay, Departure Delay (target)",
        "Arrival Delay, Departure Delay",
        "Arrival Delay, Departure Delay",
        "Arrival Delay, Departure Delay",
        "Arrival Delay, Departure Delay",
        "Arrival Delay, Departure Delay",
        "Arrival Delay, Departure Delay",
        "Arrival Delay, Departure Delay",
        "Arrival Delay, Departure Delay",
        "Arrival Delay, Departure Delay, Cancellation",
        "Cancellation",
    ],
})
leakage_table.to_csv(os.path.join(OUT_DIR, "table2_leakage_exclusion.csv"), index=False)
print("\n=== TABLE 2: Leakage Exclusion Table ===")
for _, row in leakage_table.iterrows():
    print(f"\nColumn:            {row['Excluded column']}")
    print(f"Reason:            {row['Reason for exclusion']}")
    print(f"Excluded from:     {row['Excluded from task(s)']}")

fig, ax = plt.subplots(figsize=(13, 6))
ax.axis('off')
tbl2 = ax.table(cellText=leakage_table.values, colLabels=leakage_table.columns,
                 cellLoc='left', colLoc='left', loc='center',
                 colWidths=[0.22, 0.55, 0.23])
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(8)
tbl2.scale(1, 2.2)
for (row, col), cell in tbl2.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#FAECE7')
ax.set_title("Table 2: Leakage Exclusion Table", fontsize=12, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "table2_leakage_exclusion.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)


# ============================================================
# EDA SECTION 
#   1. Cancellation class balance
#   2. Delay distribution histograms (skewness noted)
#   3. Correlation heatmap of RETAINED features vs target (leakage-fix evidence)
# ============================================================

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), dpi=300, bbox_inches="tight")
    plt.close(fig)

# --- EDA 1: Cancellation class balance ---
counts = df['CANCELLED'].value_counts().sort_index()
pcts = (counts / counts.sum() * 100).round(2)

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(['Not Cancelled', 'Cancelled'], counts.values,
               color=['#4C72B0', '#C44E52'])
for bar, pct, cnt in zip(bars, pcts.values, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f"{cnt:,}\n({pct}%)", ha='center', va='bottom', fontsize=11)
ax.set_ylabel("Number of Flights")
ax.set_title("Class Balance: Flight Cancellations")
save(fig, "eda_01_cancellation_class_balance.png")

print(f"\nCancellation class balance:\n{counts}\n{pcts}%")
print("-> Justifies class_weight='balanced' / scale_pos_weight, and using "
      "F1/ROC-AUC over raw accuracy as the cancellation selection metric.")

# --- EDA 2: Delay distribution histograms (arrival + departure) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

arr_delay_vals = df_delay['ARR_DELAY'].dropna()
arr_skew = skew(arr_delay_vals)
axes[0].hist(arr_delay_vals.clip(-60, 300), bins=100, color="#4C72B0", edgecolor='none')
axes[0].axvline(arr_delay_vals.median(), color='r', linestyle='--',
                 label=f"Median = {arr_delay_vals.median():.1f} min")
axes[0].set_title(f"Arrival Delay Distribution (skewness = {arr_skew:.2f})")
axes[0].set_xlabel("Arrival Delay (min, clipped at 300 for display)")
axes[0].set_ylabel("Flight Count")
axes[0].legend()

dep_delay_vals = df_delay['DEP_DELAY'].dropna()
dep_skew = skew(dep_delay_vals)
axes[1].hist(dep_delay_vals.clip(-60, 300), bins=100, color="#DD8452", edgecolor='none')
axes[1].axvline(dep_delay_vals.median(), color='r', linestyle='--',
                 label=f"Median = {dep_delay_vals.median():.1f} min")
axes[1].set_title(f"Departure Delay Distribution (skewness = {dep_skew:.2f})")
axes[1].set_xlabel("Departure Delay (min, clipped at 300 for display)")
axes[1].set_ylabel("Flight Count")
axes[1].legend()

save(fig, "eda_02_delay_distributions.png")

print(f"\nArrival delay skewness: {arr_skew:.3f}")
print(f"Departure delay skewness: {dep_skew:.3f}")
print("-> Heavy right skew: most flights on time, a long tail of severe "
      "delays. This is why RMSE >> MAE (RMSE penalizes the tail harder), "
      "and why R2 is structurally hard to raise from pre-departure "
      "features alone -- nearly all the variance to explain lives in that "
      "thin, hard-to-predict tail.")

# --- EDA 3: Correlation heatmap of RETAINED features vs target ---
def plot_corr_heatmap(X_train, Y_train, target_name, filename, top_n=20):
    combined = X_train.copy()
    combined[target_name] = Y_train.values
    corr = combined.corr(numeric_only=True)[target_name].drop(target_name)
    corr_sorted = corr.reindex(corr.abs().sort_values(ascending=False).index).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(corr_sorted.to_frame(), annot=True, fmt=".3f", cmap="coolwarm",
                center=0, cbar_kws={"label": "Pearson correlation"}, ax=ax)
    ax.set_title(f"Retained Feature Correlation with {target_name}\n"
                 f"(post-leakage-removal, top {top_n} by |r|)")
    save(fig, filename)
    return corr_sorted

arr_corr = plot_corr_heatmap(X_train_arr, Y_train_arr, "ARR_DELAY",
                               "eda_03_correlation_heatmap_arrival.png")
dep_corr = plot_corr_heatmap(X_train_dep, Y_train_dep, "DEP_DELAY",
                               "eda_04_correlation_heatmap_departure.png")

print("\nTop correlations with ARR_DELAY (retained features only):")
print(arr_corr)
print("\nTop correlations with DEP_DELAY (retained features only):")
print(dep_corr)
print("-> No feature should show |r| anywhere near 1.0 here. If one does, "
      "trace it back -- that's a sign a leaky column slipped through the "
      "exclusion list. Weak, spread-out correlations (typically |r| < 0.15) "
      "are the expected, honest signature of a properly leakage-free "
      "pre-departure feature set, and directly explain your low R2.")

print(f"\nEDA figures saved to '{FIG_DIR}/' (4 files, prefixed eda_)\n")

# ============================================================
# VISUALIZATIONS — Arrival & Departure Delay Classification
# ============================================================

# --- 11. Model comparison: Accuracy vs F1 (arrival + departure) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(arr_cls_results_df))
w = 0.35
axes[0].bar(x - w/2, arr_cls_results_df["Test_Accuracy"], width=w, label="Test Accuracy")
axes[0].bar(x + w/2, arr_cls_results_df["Test_F1"], width=w, label="Test F1")
axes[0].set_xticks(x); axes[0].set_xticklabels(arr_cls_results_df.index, rotation=30, ha="right")
axes[0].set_title("Arrival Delay Classification — Model Comparison")
axes[0].legend()

x2 = np.arange(len(dep_cls_results_df))
axes[1].bar(x2 - w/2, dep_cls_results_df["Test_Accuracy"], width=w, label="Test Accuracy", color="#F0997B")
axes[1].bar(x2 + w/2, dep_cls_results_df["Test_F1"], width=w, label="Test F1", color="#D85A30")
axes[1].set_xticks(x2); axes[1].set_xticklabels(dep_cls_results_df.index, rotation=30, ha="right")
axes[1].set_title("Departure Delay Classification — Model Comparison")
axes[1].legend()
save(fig, "11_delay_classification_comparison.png")

# --- 12. ROC curves (arrival + departure, all models) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for name, model in arr_cls_fitted.items():
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test_arr_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(Y_test_arr_bin, proba)
        auc = roc_auc_score(Y_test_arr_bin, proba)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curves — Arrival Delay Classifiers")
axes[0].legend()

for name, model in dep_cls_fitted.items():
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test_dep_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(Y_test_dep_bin, proba)
        auc = roc_auc_score(Y_test_dep_bin, proba)
        axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curves — Departure Delay Classifiers")
axes[1].legend()
save(fig, "12_delay_classification_roc_curves.png")

# --- 13. Confusion matrices (best arrival + best departure model) ---
cm_arr = confusion_matrix(Y_test_arr_bin, arr_cls_preds[best_arr_cls_name])
cm_dep = confusion_matrix(Y_test_dep_bin, dep_cls_preds[best_dep_cls_name])
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Not Delayed", "Delayed"], yticklabels=["Not Delayed", "Delayed"])
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
axes[0].set_title(f"Confusion Matrix — Arrival ({best_arr_cls_name})")

sns.heatmap(cm_dep, annot=True, fmt="d", cmap="Oranges", ax=axes[1],
            xticklabels=["Not Delayed", "Delayed"], yticklabels=["Not Delayed", "Delayed"])
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
axes[1].set_title(f"Confusion Matrix — Departure ({best_dep_cls_name})")
save(fig, "13_delay_classification_confusion_matrices.png")


# ============================================================
# 14. FIT DIAGNOSIS — is the selected best model over/under/good fit?
#     Compares Train vs Test Accuracy AND F1 for the chosen best model
#     only, with an automatic diagnosis label based on the train-test gap.
# ============================================================

def diagnose_fit(train_score, test_score, gap_high=0.05, low_perf=0.60):
    gap = train_score - test_score
    if train_score < low_perf and test_score < low_perf:
        return "Underfitting"
    elif gap > gap_high:
        return "Overfitting"
    else:
        return "Good Fit"


arr_diag = diagnose_fit(
    arr_cls_results_df.loc[best_arr_cls_name, "Train_Accuracy"],
    arr_cls_results_df.loc[best_arr_cls_name, "Test_Accuracy"]
)
dep_diag = diagnose_fit(
    dep_cls_results_df.loc[best_dep_cls_name, "Train_Accuracy"],
    dep_cls_results_df.loc[best_dep_cls_name, "Test_Accuracy"]
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

metrics_arr = [arr_cls_results_df.loc[best_arr_cls_name, "Train_Accuracy"],
               arr_cls_results_df.loc[best_arr_cls_name, "Test_Accuracy"]]
axes[0].bar(["Train", "Test"], metrics_arr, color=["#4C72B0", "#55A868"])
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel("Accuracy")
axes[0].set_title(f"Arrival — Best Model: {best_arr_cls_name}\nDiagnosis: {arr_diag}")
for i, v in enumerate(metrics_arr):
    axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

metrics_dep = [dep_cls_results_df.loc[best_dep_cls_name, "Train_Accuracy"],
               dep_cls_results_df.loc[best_dep_cls_name, "Test_Accuracy"]]
axes[1].bar(["Train", "Test"], metrics_dep, color=["#DD8452", "#C44E52"])
axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel("Accuracy")
axes[1].set_title(f"Departure — Best Model: {best_dep_cls_name}\nDiagnosis: {dep_diag}")
for i, v in enumerate(metrics_dep):
    axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

save(fig, "14_best_model_fit_diagnosis.png")

print(f"\nArrival delay classification — best model ({best_arr_cls_name}) fit diagnosis: {arr_diag}")
print(f"Departure delay classification — best model ({best_dep_cls_name}) fit diagnosis: {dep_diag}")
print("\nArrival classification results:\n", arr_cls_results_df)
print("\nDeparture classification results:\n", dep_cls_results_df)


# ============================================================
# VISUALIZATIONS 
# ============================================================

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), dpi=300, bbox_inches="tight")
    plt.close(fig)

# --- 1. Model comparison: Test R2 across models (arrival + departure) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
arr_results_df["Test_R2"].sort_values().plot(kind="barh", ax=axes[0], color="#4C72B0")
axes[0].set_title("Arrival Delay — Test R² by Model")
axes[0].set_xlabel("R²")
dep_results_df["Test_R2"].sort_values().plot(kind="barh", ax=axes[1], color="#DD8452")
axes[1].set_title("Departure Delay — Test R² by Model")
axes[1].set_xlabel("R²")
save(fig, "01_model_comparison_r2.png")

# --- 2. Model comparison: Test RMSE across models ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
arr_results_df["Test_RMSE"].sort_values().plot(kind="barh", ax=axes[0], color="#55A868")
axes[0].set_title("Arrival Delay — Test RMSE by Model (min)")
dep_results_df["Test_RMSE"].sort_values().plot(kind="barh", ax=axes[1], color="#C44E52")
axes[1].set_title("Departure Delay — Test RMSE by Model (min)")
save(fig, "02_model_comparison_rmse.png")

# --- 3. Train vs Test R2 (overfitting check) — Arrival ---
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(arr_results_df))
w = 0.35
ax.bar(x - w/2, arr_results_df["Train_R2"], width=w, label="Train R²")
ax.bar(x + w/2, arr_results_df["Test_R2"], width=w, label="Test R²")
ax.set_xticks(x)
ax.set_xticklabels(arr_results_df.index, rotation=30, ha="right")
ax.set_title("Arrival Delay — Train vs Test R² (Overfitting Check)")
ax.legend()
save(fig, "03_arrival_train_vs_test_r2.png")

# --- 3b. Train vs Test R2 (overfitting check) — Departure ---
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(dep_results_df))
w = 0.35
ax.bar(x - w/2, dep_results_df["Train_R2"], width=w, label="Train R²", color="#F0997B")
ax.bar(x + w/2, dep_results_df["Test_R2"], width=w, label="Test R²", color="#D85A30")
ax.set_xticks(x)
ax.set_xticklabels(dep_results_df.index, rotation=30, ha="right")
ax.set_title("Departure Delay — Train vs Test R² (Overfitting Check)")
ax.legend()
save(fig, "03b_departure_train_vs_test_r2.png")

# --- 4. Actual vs Predicted scatter — best arrival + departure model ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
sample_idx = np.random.choice(len(Y_test_arr), size=min(5000, len(Y_test_arr)), replace=False)
axes[0].scatter(Y_test_arr.values[sample_idx], arr_preds[best_arr_name][sample_idx], alpha=0.15, s=10)
lims = [min(Y_test_arr.min(), arr_preds[best_arr_name].min()), max(Y_test_arr.max(), arr_preds[best_arr_name].max())]
axes[0].plot(lims, lims, 'r--', linewidth=1)
axes[0].set_xlabel("Actual Arrival Delay (min)")
axes[0].set_ylabel("Predicted Arrival Delay (min)")
axes[0].set_title(f"Arrival Delay: Actual vs Predicted ({best_arr_name})")

sample_idx2 = np.random.choice(len(Y_test_dep), size=min(5000, len(Y_test_dep)), replace=False)
axes[1].scatter(Y_test_dep.values[sample_idx2], dep_preds[best_dep_name][sample_idx2], alpha=0.15, s=10, color="#DD8452")
lims2 = [min(Y_test_dep.min(), dep_preds[best_dep_name].min()), max(Y_test_dep.max(), dep_preds[best_dep_name].max())]
axes[1].plot(lims2, lims2, 'r--', linewidth=1)
axes[1].set_xlabel("Actual Departure Delay (min)")
axes[1].set_ylabel("Predicted Departure Delay (min)")
axes[1].set_title(f"Departure Delay: Actual vs Predicted ({best_dep_name})")
save(fig, "04_actual_vs_predicted.png")

# --- 5. Residual plots ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
resid_arr = Y_test_arr.values[sample_idx] - arr_preds[best_arr_name][sample_idx]
axes[0].scatter(arr_preds[best_arr_name][sample_idx], resid_arr, alpha=0.15, s=10)
axes[0].axhline(0, color='r', linestyle='--', linewidth=1)
axes[0].set_xlabel("Predicted Arrival Delay (min)")
axes[0].set_ylabel("Residual (Actual - Predicted)")
axes[0].set_title(f"Arrival Delay Residuals ({best_arr_name})")

resid_dep = Y_test_dep.values[sample_idx2] - dep_preds[best_dep_name][sample_idx2]
axes[1].scatter(dep_preds[best_dep_name][sample_idx2], resid_dep, alpha=0.15, s=10, color="#DD8452")
axes[1].axhline(0, color='r', linestyle='--', linewidth=1)
axes[1].set_xlabel("Predicted Departure Delay (min)")
axes[1].set_ylabel("Residual (Actual - Predicted)")
axes[1].set_title(f"Departure Delay Residuals ({best_dep_name})")
save(fig, "05_residuals.png")

# --- 6. Feature importance (only if best model is tree-based) ---
def plot_feature_importance(model, feature_names, title, filename, top_n=15):
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(9, 7))
    importances.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    save(fig, filename)

plot_feature_importance(best_arr_model, X_train_arr.columns, f"Top Feature Importances — Arrival Delay ({best_arr_name})", "06_feature_importance_arrival.png")
plot_feature_importance(best_dep_model, X_train_dep.columns, f"Top Feature Importances — Departure Delay ({best_dep_name})", "07_feature_importance_departure.png")

# --- 7. Cancellation: confusion matrix heatmap ---
cm = confusion_matrix(Y_test_cancel, cancel_preds[best_cancel_name])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Not Cancelled", "Cancelled"],
            yticklabels=["Not Cancelled", "Cancelled"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {best_cancel_name}")
save(fig, "08_confusion_matrix.png")

# --- 8. Cancellation: ROC curve (all models with predict_proba) ---
fig, ax = plt.subplots(figsize=(7, 6))
for name, model in cancel_fitted.items():
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test_cancel_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(Y_test_cancel, proba)
        auc = roc_auc_score(Y_test_cancel, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Cancellation Classifiers")
ax.legend()
save(fig, "09_roc_curves.png")

# --- 9. Cancellation: model comparison (F1 / Accuracy) ---
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(cancel_results_df))
w = 0.35
ax.bar(x - w/2, cancel_results_df["Test_Accuracy"], width=w, label="Test Accuracy")
ax.bar(x + w/2, cancel_results_df["Test_F1"], width=w, label="Test F1")
ax.set_xticks(x)
ax.set_xticklabels(cancel_results_df.index, rotation=30, ha="right")
ax.set_title("Cancellation — Model Comparison")
ax.legend()
save(fig, "10_cancellation_comparison.png")

print(f"\nAll metrics CSVs and {len(os.listdir(FIG_DIR))} figures saved to '{OUT_DIR}/'")
print("Arrival results:\n", arr_results_df)
print("\nDeparture results:\n", dep_results_df)
print("\nCancellation results:\n", cancel_results_df)
