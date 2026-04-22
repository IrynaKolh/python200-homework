# for the project exercise
# --- Part 2: Mini-Project -- Predicting Student Math Performance ---
#
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so plots save without a display
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Task 1: Load and Explore
# ---------------------------------------------------------------------------
print("=" * 60)
print("TASK 1: Load and Explore")
print("=" * 60)
# Dataset uses ';' as separator → need sep=';' in pd.read_csv
df = pd.read_csv(os.path.join(BASE_DIR, "student_performance_math.csv"), sep=";")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)

# Histogram of G3 (one bin per grade value 0-20 = 21 bins)
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df["G3"], bins=range(0, 22), edgecolor="black", color="steelblue", align="left")
ax.set_title("Distribution of Final Math Grades")
ax.set_xlabel("G3 (Final Grade)")
ax.set_ylabel("Count")
ax.set_xticks(range(0, 21))
path = os.path.join(OUTPUT_DIR, "g3_distribution.png")
fig.savefig(path, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved to {path}")
# The histogram shows a cluster of zeros that sits clearly apart from the main
# distribution (which peaks around 10-12). These zeros are students who were
# absent for the final exam, not students who genuinely scored zero.


# ---------------------------------------------------------------------------
# Task 2: Preprocess the Data
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 2: Preprocess the Data")
print("=" * 60)

# --- Handle G3=0 rows ---
print(f"Shape before filtering G3=0: {df.shape}")
df_clean = df[df["G3"] > 0].copy()
print(f"Shape after  filtering G3=0: {df_clean.shape}")
print(f"Rows removed: {len(df) - len(df_clean)}")
# Keeping G3=0 rows would distort the model because those students did not sit
# the final exam -- their zero is a missing value, not a real score. Including
# them would artificially lower predicted grades for students who share their
# characteristics (older, more failures, more absences), pulling the regression
# line down and making every coefficient biased toward zero.

# --- Convert yes/no → 1/0 and sex F/M → 0/1 ---
binary_yn_cols = ["schoolsup", "internet", "higher", "activities"]
for col in binary_yn_cols:
    df_clean[col] = (df_clean[col] == "yes").astype(int)
    df[col]       = (df[col]       == "yes").astype(int)

df_clean["sex"] = (df_clean["sex"] == "M").astype(int)   # F=0, M=1
df["sex"]       = (df["sex"]       == "M").astype(int)

# --- Absences correlation: original vs filtered ---
r_orig, _ = pearsonr(df["absences"],       df["G3"])
r_filt, _ = pearsonr(df_clean["absences"], df_clean["G3"])
print(f"\nPearson r(absences, G3)  -- original dataset : {r_orig:.4f}")
print(f"Pearson r(absences, G3)  -- filtered dataset : {r_filt:.4f}")
# The correlation is much stronger (more negative) in the filtered dataset.
# Students with G3=0 typically had VERY high absence counts AND a zero final
# grade. Those extreme (high-absence, zero-grade) points added artificial
# variance that made high absences *look* associated with low (zero) grades,
# but for a different reason than academic performance: they simply skipped
# the exam. Once those rows are removed, the true negative relationship between
# absences and earned grades becomes clearer.


# ---------------------------------------------------------------------------
# Task 3: Exploratory Data Analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 3: Exploratory Data Analysis")
print("=" * 60)

numeric_cols = ["age", "Medu", "Fedu", "traveltime", "studytime",
                "failures", "absences", "freetime", "goout", "Walc",
                "schoolsup", "internet", "higher", "activities", "sex"]

correlations = {}
for col in numeric_cols:
    r, _ = pearsonr(df_clean[col], df_clean["G3"])
    correlations[col] = r

sorted_corr = sorted(correlations.items(), key=lambda x: x[1])
print("\nPearson correlations with G3 (sorted most negative → most positive):")
for col, r in sorted_corr:
    print(f"  {col:12s}: {r:+.4f}")
# failures has the strongest negative relationship with G3.
# 'higher' (wanting to pursue higher education) has the strongest positive one.
# Surprisingly, absences shows only a modest negative correlation even after
# filtering -- suggesting that within exam-takers, attendance is less decisive
# than we might expect.

# --- Plot 1: G3 by number of past failures (boxplot) ---
fig, ax = plt.subplots(figsize=(8, 5))
failure_groups = [df_clean[df_clean["failures"] == f]["G3"].values
                  for f in sorted(df_clean["failures"].unique())]
failure_labels = [str(f) for f in sorted(df_clean["failures"].unique())]
ax.boxplot(failure_groups, tick_labels=failure_labels)
ax.set_title("G3 Distribution by Number of Past Failures")
ax.set_xlabel("Past Failures")
ax.set_ylabel("G3 (Final Grade)")
path = os.path.join(OUTPUT_DIR, "g3_by_failures.png")
fig.savefig(path, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved to {path}")
# Students with 0 past failures have a much higher and tighter grade distribution.
# Each additional failure step shifts the median notably downward and the spread
# becomes wider, showing that past failure is both a strong and reliable predictor.

# --- Plot 2: G3 by 'higher' (wants higher education) ---
fig, ax = plt.subplots(figsize=(7, 5))
groups  = [df_clean[df_clean["higher"] == v]["G3"].values for v in [0, 1]]
ax.boxplot(groups, tick_labels=["No (0)", "Yes (1)"])
ax.set_title("G3 Distribution by Higher-Education Aspiration")
ax.set_xlabel("Wants Higher Education")
ax.set_ylabel("G3 (Final Grade)")
path = os.path.join(OUTPUT_DIR, "g3_by_higher.png")
fig.savefig(path, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved to {path}")
# Students who want to pursue higher education score noticeably higher on average.
# The "No" group has a lower median and a long lower tail. This is likely a
# motivational effect: students with academic ambitions study more consistently.


# ---------------------------------------------------------------------------
# Task 4: Baseline Model (failures only)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 4: Baseline Model (failures only)")
print("=" * 60)

X_base = df_clean[["failures"]].values
y      = df_clean["G3"].values

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_base, y, test_size=0.2, random_state=42)

model_base = LinearRegression()
model_base.fit(X_train_b, y_train_b)
y_pred_b = model_base.predict(X_test_b)

rmse_b = np.sqrt(np.mean((y_pred_b - y_test_b) ** 2))
r2_b   = model_base.score(X_test_b, y_test_b)

print(f"Slope     : {model_base.coef_[0]:+.4f}")
print(f"Intercept : {model_base.intercept_:.4f}")
print(f"RMSE      : {rmse_b:.4f}")
print(f"R²        : {r2_b:.4f}")
# On a 0-20 scale, the slope of roughly -2 means each additional past failure
# is associated with about 2 fewer points in the final grade.
# An RMSE of ~3.3 means a typical prediction is off by about 3-4 grade points --
# enough to misclassify a student as passing vs. failing in many grading systems.
# R² around 0.13 is low but not surprising: failures alone captures the
# broad trend but leaves most variance unexplained. The EDA suggested failures
# is the strongest single predictor, so this is roughly what we expected.


# ---------------------------------------------------------------------------
# Task 5: Full Model
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 5: Full Model")
print("=" * 60)

feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]

X = df_clean[feature_cols].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model_full = LinearRegression()
model_full.fit(X_train, y_train)
y_pred = model_full.predict(X_test)

rmse_full   = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2_train    = model_full.score(X_train, y_train)
r2_test     = model_full.score(X_test,  y_test)

print(f"Train R²  : {r2_train:.4f}")
print(f"Test  R²  : {r2_test:.4f}")
print(f"RMSE      : {rmse_full:.4f}")
print(f"\nBaseline R² (failures only): {r2_b:.4f}")
print(f"Full model R²              : {r2_test:.4f}")
print(f"Improvement                : +{r2_test - r2_b:.4f}")

print("\nFeature coefficients:")
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"  {name:12s}: {coef:+.3f}")

# Coefficient notes:
# - failures is the largest negative coefficient (~-2): the more past failures,
#   the lower the predicted grade. This is the expected direction and magnitude.
# - higher (wants higher education) is the largest positive coefficient (~+1.5):
#   academic ambition correlates with better performance.
# - schoolsup is NEGATIVE (~-1.5), which looks surprising at first. Students
#   receiving extra school support likely already have lower grades -- the support
#   is remedial. The coefficient reflects who gets it, not whether it helps.
# - sex (M=1) is modestly positive (~+0.5): male students score slightly higher
#   on average in this Portuguese 2005 dataset. As the feature guide notes, PISA
#   research links this gap to social context, not inherent ability.
# - Train and test R² are close (no large gap), suggesting the model is not
#   overfitting -- it generalizes about as well as it fits the training data.
#
# Production feature selection:
#   KEEP: failures, higher, Medu, studytime, schoolsup, sex
#         -- these have the largest absolute coefficients and clear interpretations.
#   CONSIDER DROPPING: activities, internet, freetime, traveltime, Fedu
#         -- small coefficients (near zero), adding noise without much signal.


# ---------------------------------------------------------------------------
# Task 6: Evaluate and Summarize
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 6: Evaluate and Summarize")
print("=" * 60)

# Predicted vs Actual plot
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_pred, y_test, alpha=0.6, color="steelblue", edgecolors="white", linewidths=0.4)
diag_min = min(y_pred.min(), y_test.min())
diag_max = max(y_pred.max(), y_test.max())
ax.plot([diag_min, diag_max], [diag_min, diag_max], "k--", lw=1.5, label="Perfect prediction")
ax.set_title("Predicted vs Actual (Full Model)")
ax.set_xlabel("Predicted G3")
ax.set_ylabel("Actual G3")
ax.legend()
path = os.path.join(OUTPUT_DIR, "predicted_vs_actual.png")
fig.savefig(path, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved to {path}")

# A point ABOVE the diagonal means the model underpredicted -- the student did
# better than expected. A point BELOW means the model overpredicted -- the
# student did worse. The scatter appears roughly uniform across grade levels,
# with no strong fan shape, suggesting errors are similar regardless of grade.
# The spread is widest in the middle range (8-14), where most students cluster.

# --- Plain-language summary ---
print(f"\nDataset size (after filtering): {len(df_clean)} students")
print(f"Test set size                  : {len(y_test)} students")
print(f"Best model RMSE                : {rmse_full:.2f}  "
      f"(typical error ~{rmse_full:.1f} grade points on a 0-20 scale)")
print(f"Best model R²                  : {r2_test:.4f}  "
      f"(explains ~{r2_test*100:.0f}% of variance in final grades)")

coef_dict = dict(zip(feature_cols, model_full.coef_))
top2_pos = sorted(coef_dict, key=lambda k: coef_dict[k], reverse=True)[:2]
top2_neg = sorted(coef_dict, key=lambda k: coef_dict[k])[:2]
print("\nTwo largest POSITIVE coefficients:")
for f in top2_pos:
    print(f"  {f}: {coef_dict[f]:+.3f}")
print("Two largest NEGATIVE coefficients:")
for f in top2_neg:
    print(f"  {f}: {coef_dict[f]:+.3f}")

# Surprise: schoolsup having a negative coefficient is counterintuitive --
# extra school support hurts predictions? It is actually a selection effect:
# the weakest students receive remedial support, so 'schoolsup=1' is a proxy
# for 'struggling student', not a measure of how effective the support is.


# ---------------------------------------------------------------------------
# Bonus: Add G1 to the full model
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("BONUS: Full Model + G1")
print("=" * 60)

feature_cols_g1 = feature_cols + ["G1"]
X_g1 = df_clean[feature_cols_g1].values
y_g1 = df_clean["G3"].values

X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(
    X_g1, y_g1, test_size=0.2, random_state=42)

model_g1 = LinearRegression()
model_g1.fit(X_train_g1, y_train_g1)
r2_test_g1 = model_g1.score(X_test_g1, y_test_g1)

print(f"Test R² without G1: {r2_test:.4f}")
print(f"Test R² with    G1: {r2_test_g1:.4f}")
print(f"Jump in R²        : +{r2_test_g1 - r2_test:.4f}")

# Does a high R² here mean G1 is CAUSING G3? No -- correlation is not
# causation. G1 and G3 are highly correlated because they measure the same
# underlying thing: how well this particular student understands math.
# A student who scores 16 in period 1 likely scores 16 in period 3 because
# they are a strong math student, not because G1 produced G3.
#
# Is this a useful model for identifying students who might struggle?
# Only partially. If a student has already received G1, we know who is
# struggling -- we don't need a model to tell us. The real value would be
# in early intervention BEFORE G1 is available, using background features
# (failures, mother's education, study time) to flag at-risk students on
# day one of the school year. That is the Task 5 model's purpose:
# it can act before any grade information is collected.
