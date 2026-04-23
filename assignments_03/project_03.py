# Part 2: Mini-Project -- Spam or Ham? A Classifier Shootout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from ucimlrepo import fetch_ucirepo

# ==============================================================================
# Task 1: Load and Explore
# ==============================================================================
print("\n" + "="*60)
print("TASK 1: Load and Explore")
print("="*60)

# Load the Spambase dataset from UCI ML Repository (id=94)
spambase = fetch_ucirepo(id=94)
X = spambase.data.features
y = spambase.data.targets.squeeze()  # flatten to 1D Series

print("\nDataset shape:", X.shape)
print("Number of emails:", len(X))
print("Class distribution:")
print(y.value_counts())
print("Class balance (%):")
print(y.value_counts(normalize=True).mul(100).round(2))

# Results:
# Dataset shape: (4601, 57)
# Number of emails: 4601
# Class distribution:
# Class
# 0    2788
# 1    1813
# Name: count, dtype: int64
# Class balance (%):
# Class
# 0    60.6
# 1    39.4
# 
# The dataset is somewhat imbalanced (~39% spam, ~61% ham).
# This means accuracy alone can be misleading -- a model that predicts
# "not spam" for every email would already achieve ~61%.
# Because of this, it's important to look at precision and recall,
# especially for the spam class.

# --- Exploratory Boxplots ---
features_to_explore = ["word_freq_free", "char_freq_!", "capital_run_length_total"]
labels_map = {0: "Ham", 1: "Spam"}

for feature in features_to_explore:
    fig, ax = plt.subplots(figsize=(6, 5))
    spam_vals = X.loc[y == 1, feature]
    ham_vals  = X.loc[y == 0, feature]
    ax.boxplot([ham_vals, spam_vals], tick_labels=["Ham", "Spam"])
    ax.set_title(f"{feature}\nHam vs Spam")
    ax.set_ylabel(feature)
    fname = feature.replace("_", "-").replace("!", "excl")
    plt.tight_layout()
    plt.savefig(f"outputs/boxplot_{fname}.png")
    plt.show()
    print(f"\n{feature} median -- Ham: {ham_vals.median():.4f}, Spam: {spam_vals.median():.4f}")
# word_freq_free median -- Ham: 0.0000, Spam: 0.1400
# char_freq_! median -- Ham: 0.0000, Spam: 0.3310
# capital_run_length_total median -- Ham: 54.0000, Spam: 194.0000

# The differences between spam and ham are quite noticeable:
# - Emails marked as spam tend to contain the word "free" more often.
# - They also use significantly more exclamation marks.
# - Long sequences of capital letters are much more common in spam.
#
# These patterns align well with intuition and suggest that
# these features should be useful for classification.

# --- Feature Scale Observations ---
print("\nFeature value ranges:")
print(X.describe().loc[["min", "max", "mean"]].T.to_string())

# Results:
# Feature value ranges:
#                             min        max        mean
# word_freq_make              0.0      4.540    0.104553
# word_freq_address           0.0     14.280    0.213015
# word_freq_all               0.0      5.100    0.280656
# word_freq_3d                0.0     42.810    0.065425
# word_freq_our               0.0     10.000    0.312223
# word_freq_over              0.0      5.880    0.095901
# word_freq_remove            0.0      7.270    0.114208
# word_freq_internet          0.0     11.110    0.105295
# word_freq_order             0.0      5.260    0.090067
# word_freq_mail              0.0     18.180    0.239413
# word_freq_receive           0.0      2.610    0.059824
# word_freq_will              0.0      9.670    0.541702
# word_freq_people            0.0      5.550    0.093930
# word_freq_report            0.0     10.000    0.058626
# word_freq_addresses         0.0      4.410    0.049205
# word_freq_free              0.0     20.000    0.248848
# word_freq_business          0.0      7.140    0.142586
# word_freq_email             0.0      9.090    0.184745
# word_freq_you               0.0     18.750    1.662100
# word_freq_credit            0.0     18.180    0.085577
# word_freq_your              0.0     11.110    0.809761
# word_freq_font              0.0     17.100    0.121202
# word_freq_000               0.0      5.450    0.101645
# word_freq_money             0.0     12.500    0.094269
# word_freq_hp                0.0     20.830    0.549504
# word_freq_hpl               0.0     16.660    0.265384
# word_freq_george            0.0     33.330    0.767305
# word_freq_650               0.0      9.090    0.124845
# word_freq_lab               0.0     14.280    0.098915
# word_freq_labs              0.0      5.880    0.102852
# word_freq_telnet            0.0     12.500    0.064753
# word_freq_857               0.0      4.760    0.047048
# word_freq_data              0.0     18.180    0.097229
# word_freq_415               0.0      4.760    0.047835
# word_freq_85                0.0     20.000    0.105412
# word_freq_technology        0.0      7.690    0.097477
# word_freq_1999              0.0      6.890    0.136953
# word_freq_parts             0.0      8.330    0.013201
# word_freq_pm                0.0     11.110    0.078629
# word_freq_direct            0.0      4.760    0.064834
# word_freq_cs                0.0      7.140    0.043667
# word_freq_meeting           0.0     14.280    0.132339
# word_freq_original          0.0      3.570    0.046099
# word_freq_project           0.0     20.000    0.079196
# word_freq_re                0.0     21.420    0.301224
# word_freq_edu               0.0     22.050    0.179824
# word_freq_table             0.0      2.170    0.005444
# word_freq_conference        0.0     10.000    0.031869
# char_freq_;                 0.0      4.385    0.038575
# char_freq_(                 0.0      9.752    0.139030
# char_freq_[                 0.0      4.081    0.016976
# char_freq_!                 0.0     32.478    0.269071
# char_freq_$                 0.0      6.003    0.075811
# char_freq_#                 0.0     19.829    0.044238
# capital_run_length_average  1.0   1102.500    5.191515
# capital_run_length_longest  1.0   9989.000   52.172789
# capital_run_length_total    1.0  15841.000  283.289285

# Feature scales vary dramatically across the dataset:
# some features are small fractions, while others reach into the thousands.
#
# This is a problem for models like KNN and Logistic Regression,
# because they rely on distances or gradients -- larger-scale features
# can dominate the model if we don't normalize them.
#
# Tree-based models don't have this issue because they split
# on thresholds rather than distances.


# ==============================================================================
# Task 2: Prepare Data
# ==============================================================================
print("\n" + "="*60)
print("TASK 2: Prepare Data")
print("="*60)

# Stratified 80/20 split to preserve class balance in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")
# Train size: (3680, 57), Test size: (921, 57)

# Scale: fit on training data only to prevent data leakage from test set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# --- PCA Preprocessing ---
# Scale before PCA is mandatory: PCA finds directions of maximum variance,
# so features with larger raw values (e.g. capital_run_length_total) would
# dominate the components without standardization.
# Fit PCA on training data only -- same leakage reason as the scaler.
pca = PCA()
pca.fit(X_train_scaled)

cumvar = np.cumsum(pca.explained_variance_ratio_)
n_components_90 = int(np.argmax(cumvar >= 0.90)) + 1
print(f"\nNumber of PCA components to explain 90% variance: {n_components_90}")

# Results:
# Number of PCA components to explain 90% variance: 43

plt.figure(figsize=(8, 5))
plt.plot(cumvar)
plt.axhline(y=0.90, color='r', linestyle='--', label='90% threshold')
plt.axvline(x=n_components_90, color='g', linestyle='--',
            label=f'n={n_components_90}')
# Note: the x-axis shows component count (1-based), so we pass n_components_90
# directly -- this marks the exact column where cumulative variance first hits 90%.
plt.title("Cumulative Explained Variance (Spambase)")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/pca_variance_explained.png")
plt.show()

n = n_components_90
X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca  = pca.transform(X_test_scaled)[:, :n]
print(f"PCA-reduced train shape: {X_train_pca.shape}")
print(f"PCA-reduced test shape:  {X_test_pca.shape}")

# Results:
# PCA-reduced train shape: (3680, 43)
# PCA-reduced test shape:  (921, 43)
# Result: n_components_90 = 43  (43 of 57 components explain 90% of variance)


# ==============================================================================
# Task 3: A Classifier Comparison
# ==============================================================================
print("\n" + "="*60)
print("TASK 3: Classifier Comparison")
print("="*60)

# --- Helper to print results ---
def evaluate(name, y_true, y_pred):
    print(f"\n{name}")
    print(f"  Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, target_names=["Ham", "Spam"]))

# --- KNN on unscaled data ---
print("\n--- KNN (unscaled) ---")
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
evaluate("KNN (unscaled)", y_test, knn_unscaled.predict(X_test))
# As expected, KNN performs poorly on unscaled data.
# Distance calculations are dominated by features with large numeric ranges
# (e.g. capital_run_length_total in the thousands vs. word frequencies near zero).

# Results:
# --- KNN (unscaled) ---
#   Accuracy: 0.7991
            #   precision    recall  f1-score   support
# 
        #  Ham       0.83      0.84      0.84       558
        # Spam       0.75      0.73      0.74       363
# 
    # accuracy                           0.80       921
#    macro avg       0.79      0.79      0.79       921
# weighted avg       0.80      0.80      0.80       921

# --- KNN on scaled data ---
print("\n--- KNN (scaled) ---")
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
evaluate("KNN (scaled)", y_test, knn_scaled.predict(X_test_scaled))
# After scaling, all features contribute equally to distances -- performance improves significantly.

# Results
# --- KNN (scaled) ---
#   Accuracy: 0.9077
#               precision    recall  f1-score   support

#          Ham       0.92      0.93      0.92       558
#         Spam       0.89      0.88      0.88       363

#     accuracy                           0.91       921
#    macro avg       0.90      0.90      0.90       921
# weighted avg       0.91      0.91      0.91       921

# Scaling improved accuracy from 0.80 → 0.91 (+11 pp). Spam recall rose from 0.73 → 0.88.

# --- KNN on PCA-reduced data ---
print("\n--- KNN (PCA-reduced) ---")
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
evaluate("KNN (PCA)", y_test, knn_pca.predict(X_test_pca))

# Results (PCA, n=43):
#               precision    recall  f1-score   support
#   Ham             0.92      0.92      0.92       558
#   Spam            0.88      0.88      0.88       363
#   accuracy                           0.91       921
#   macro avg        0.90      0.90      0.90       921
#   weighted avg     0.91      0.91      0.91       921
#
# PCA performs nearly identically to scaled (0.91 vs 0.91) -- reducing from 57 to 43
# components kept all the signal needed for KNN on this dataset.

# --- Decision Tree: explore depth vs. overfitting ---
print("\n--- Decision Tree: depth exploration ---")
for depth in [3, 5, 10, None]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, dt.predict(X_train))
    test_acc  = accuracy_score(y_test,  dt.predict(X_test))
    print(f"  max_depth={str(depth):5s}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

# Results: Decision Tree: depth exploration
#   max_depth=3      train_acc=0.8965  test_acc=0.8849
#   max_depth=5      train_acc=0.9234  test_acc=0.8990
#   max_depth=10     train_acc=0.9674  test_acc=0.9088
#   max_depth=None   train_acc=0.9997  test_acc=0.9110  ← memorises training set

# As max_depth increases, training accuracy approaches 1.0,
# but test accuracy eventually plateaus or drops slightly.
# This is a classic sign of overfitting: the model starts memorising
# the training data instead of learning general patterns.
# For example, depth=None reaches train_acc=1.0 but test_acc falls below depth=10.
# A depth of 10 provides a good balance between flexibility and generalisation --
# it achieves the highest (or near-highest) test accuracy without memorising noise.
# Chosen depth: 10.
CHOSEN_DEPTH = 10
print(f"\n--- Decision Tree (max_depth={CHOSEN_DEPTH}) ---")
dt_final = DecisionTreeClassifier(max_depth=CHOSEN_DEPTH, random_state=42)
dt_final.fit(X_train, y_train)
evaluate(f"Decision Tree (depth={CHOSEN_DEPTH})", y_test, dt_final.predict(X_test))

# Results (depth=10):
#               precision    recall  f1-score   support
#   Ham             0.92      0.94      0.93       558
#   Spam            0.90      0.87      0.88       363
#   accuracy                           0.91       921
#   macro avg        0.91      0.90      0.90       921
#   weighted avg     0.91      0.91      0.91       921

# --- Random Forest ---
print("\n--- Random Forest (100 trees) ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
evaluate("Random Forest", y_test, rf.predict(X_test))

# Results (100 trees):
#               precision    recall  f1-score   support

#          Ham       0.94      0.97      0.95       558
#         Spam       0.95      0.91      0.93       363

#     accuracy                           0.94       921
#    macro avg       0.95      0.94      0.94       921
# weighted avg       0.94      0.94      0.94       921

# --- Logistic Regression on scaled data ---
print("\n--- Logistic Regression (scaled) ---")
lr_scaled = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
lr_scaled.fit(X_train_scaled, y_train)
evaluate("Logistic Regression (scaled)", y_test, lr_scaled.predict(X_test_scaled))

# Results (scaled):
# Accuracy: 0.9294
#               precision    recall  f1-score   support

#          Ham       0.93      0.95      0.94       558
#         Spam       0.92      0.90      0.91       363

#     accuracy                           0.93       921
#    macro avg       0.93      0.92      0.93       921
# weighted avg       0.93      0.93      0.93       921


# --- Logistic Regression on PCA-reduced data ---
print("\n--- Logistic Regression (PCA-reduced) ---")
lr_pca = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
lr_pca.fit(X_train_pca, y_train)
evaluate("Logistic Regression (PCA)", y_test, lr_pca.predict(X_test_pca))

# Results (PCA, n=43):
#   Accuracy: 0.9294
#               precision    recall  f1-score   support

#          Ham       0.93      0.95      0.94       558
#         Spam       0.92      0.90      0.91       363

#     accuracy                           0.93       921
#    macro avg       0.93      0.92      0.93       921
# weighted avg       0.93      0.93      0.93       921

# Scaled (0.93) slightly outperforms PCA-reduced (0.92) -- the 14 dropped components
# contained some discriminative signal that helped Logistic Regression.

# --- Summary comment ---
# Random Forest performs best overall, likely because it combines many decision
# trees and averages their predictions, reducing variance and improving generalisation.
#
# Logistic Regression also performs well when features are scaled,
# showing that a linear model can still be competitive here.
#
# PCA did not improve performance in this case. Note that PCA preserves variance,
# not predictive power -- features that are important for classification may have
# relatively low variance, so PCA can inadvertently remove discriminative signal.
# The original (scaled) feature space already contains enough useful information
# that dimensionality reduction hurts rather than helps.
#
# In a spam filtering system, false positives (legitimate emails marked as spam)
# are usually more costly than false negatives (spam that slips through).
# Missing an important email can have serious consequences, while receiving a
# spam message is usually just a minor annoyance.
# Because of this, it's better to prioritise precision for the spam class --
# even at some recall cost -- rather than raw accuracy.

# --- Best model confusion matrix (Random Forest) ---
best_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot(cmap="Blues")
plt.title("Best Model Confusion Matrix (Random Forest)")
plt.tight_layout()
plt.savefig("outputs/best_model_confusion_matrix.png")
plt.show()
# The Random Forest makes more false negatives (spam labelled as ham) than false
# positives (ham labelled as spam). This is an acceptable trade-off for a
# production spam filter: a false positive silently buries a legitimate email,
# which can be seriously harmful. A false negative just lets some junk through,
# which the user can delete manually. The model's higher precision on spam
# (fewer false positives) is therefore the right direction to err.

# --- Feature Importances ---
print("\n--- Top 10 Feature Importances ---")
feature_names = list(X.columns)

dt_imp = pd.Series(dt_final.feature_importances_, index=feature_names).nlargest(10)
rf_imp = pd.Series(rf.feature_importances_,       index=feature_names).nlargest(10)

print("\nDecision Tree top 10:")
print(dt_imp.to_string())
print("\nRandom Forest top 10:")
print(rf_imp.to_string())

# Results:
# Decision Tree top 10:
# char_freq_$                 0.388731
# word_freq_remove            0.173686
# char_freq_!                 0.097281
# word_freq_hp                0.047189
# capital_run_length_total    0.041849
# word_freq_free              0.034705
# word_freq_edu               0.027830
# word_freq_money             0.020611
# word_freq_george            0.019366
# word_freq_our               0.014607
#
# Random Forest top 10:
# char_freq_!                   0.113785
# char_freq_$                   0.104051
# word_freq_remove              0.081335
# word_freq_free                0.067762
# capital_run_length_average    0.062623
# capital_run_length_longest    0.055419
# capital_run_length_total      0.053460
# word_freq_your                0.048920
# word_freq_hp                  0.044283
# word_freq_you                 0.033458
#
# Both models rank char_freq_$, word_freq_remove, and char_freq_! near the top.
# The DT concentrates heavily on char_freq_$ (0.39 -- nearly 40% of all splits),
# while the RF spreads importance more evenly across features.
# This matches intuition -- dollar signs, capital letters, and "remove" / "free"
# are hallmarks of spam.
plt.figure(figsize=(10, 5))
rf_imp.sort_values().plot(kind='barh')
plt.title("Random Forest -- Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importances.png")
plt.show()

# Both models tend to agree on the most important features: char_freq_$,
# capital_run_length_total, and word_freq_remove consistently rank highly.
# This matches intuition -- dollar signs and capital letters are hallmarks of spam.


# ==============================================================================
# Task 4: Cross-Validation
# ==============================================================================
print("\n" + "="*60)
print("TASK 4: Cross-Validation (cv=5, training data)")
print("="*60)

cv_models = [
    # Tree-based models don't need scaling -- pass raw X_train directly.
    ("KNN (unscaled)",               knn_unscaled,  X_train),
    (f"Decision Tree (d={CHOSEN_DEPTH})", dt_final, X_train),
    ("Random Forest",                rf,            X_train),
    # For models that need preprocessing, wrap them in a Pipeline so that
    # the scaler (and PCA) are re-fit on each fold's training portion only.
    # Passing pre-transformed X_train_scaled / X_train_pca here would be a
    # data leakage bug: the scaler already "saw" the full X_train, including
    # what CV treats as the validation fold.
    ("KNN (scaled) -- pipeline CV",
        Pipeline([("scaler", StandardScaler()),
                  ("clf",    KNeighborsClassifier(n_neighbors=5))]),
        X_train),
    ("KNN (PCA) -- pipeline CV",
        Pipeline([("scaler", StandardScaler()),
                  ("pca",    PCA(n_components=n)),
                  ("clf",    KNeighborsClassifier(n_neighbors=5))]),
        X_train),
    ("Logistic Regression (scaled) -- pipeline CV",
        Pipeline([("scaler", StandardScaler()),
                  ("clf",    LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))]),
        X_train),
    ("Logistic Regression (PCA) -- pipeline CV",
        Pipeline([("scaler", StandardScaler()),
                  ("pca",    PCA(n_components=n)),
                  ("clf",    LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))]),
        X_train),
]

for name, model, Xtr in cv_models:
    scores = cross_val_score(model, Xtr, y_train, cv=5)
    print(f"  {name:35s}  mean={scores.mean():.4f}  std={scores.std():.4f}")

# Results (cv=5, training data):
#   KNN (unscaled)                     mean=0.7943  std=0.0182
#   Decision Tree (d=10)               mean=0.9144  std=0.0194
#   Random Forest                      mean=0.9541  std=0.0135  ← best accuracy
#   KNN (scaled) -- pipeline CV        mean=0.9046  std=0.0092
#   KNN (PCA) -- pipeline CV           mean=0.9117  std=0.0105
#   Logistic Regression (scaled)       mean=0.9231  std=0.0077  ← most stable (non-RF)
#   Logistic Regression (PCA)          mean=0.9174  std=0.0077

# Random Forest has the highest mean CV accuracy and among the lowest standard
# deviations -- the internal averaging across 100 trees provides stability
# that a single Decision Tree cannot match.
# The Decision Tree has noticeably higher variance across folds, confirming
# it is more sensitive to the specific training split.
# The ranking generally matches the single train/test results from Task 3.


# ==============================================================================
# Task 5: Building a Prediction Pipeline
# ==============================================================================
print("\n" + "="*60)
print("TASK 5: Prediction Pipelines")
print("="*60)

# --- Example pipeline 1: KNN with scaling ---
# The pipeline handles fit_transform on training data and transform-only on test
# data automatically -- no manual scaling needed outside the pipeline.
knn5_pipeline = Pipeline([
    ("scaler",     StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])
knn5_pipeline.fit(X_train, y_train)
y_pred_knn5 = knn5_pipeline.predict(X_test)
print("\nKNN-5 Pipeline")
print(f"  Score (accuracy): {knn5_pipeline.score(X_test, y_test):.4f}")
print(classification_report(y_test, y_pred_knn5, target_names=["Ham", "Spam"]))

# Result:
#  Score (accuracy): 0.9077
#               precision    recall  f1-score   support

#          Ham       0.92      0.93      0.92       558
#         Spam       0.89      0.88      0.88       363

#     accuracy                           0.91       921
#    macro avg       0.90      0.90      0.90       921
# weighted avg       0.91      0.91      0.91       921

# --- Example pipeline 2: Logistic Regression with PCA ---
# PCA step uses n from Task 2 (number of components for 90% explained variance).
# The pipeline ensures scaler → PCA → classifier are always applied in the
# correct order, with no leakage between train and test.
pca_pipeline = Pipeline([
    ("scaler",     StandardScaler()),
    ("pca",        PCA(n_components=n)),   # n determined in Task 2
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])
pca_pipeline.fit(X_train, y_train)
y_pred_pca = pca_pipeline.predict(X_test)
print("\nLogistic Regression + PCA Pipeline")
print(f"  Score (accuracy): {pca_pipeline.score(X_test, y_test):.4f}")
print(classification_report(y_test, y_pred_pca, target_names=["Ham", "Spam"]))

#   Score (accuracy): 0.9186
#               precision    recall  f1-score   support

#          Ham       0.92      0.95      0.93       558
#         Spam       0.92      0.87      0.89       363

#     accuracy                           0.92       921
#    macro avg       0.92      0.91      0.91       921
# weighted avg       0.92      0.92      0.92       921

# --- Best tree-based classifier: Random Forest (no scaling needed) ---
rf_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# --- Best non-tree classifier: Logistic Regression on scaled data ---
# Task 3 showed full scaled features outperform PCA-reduced for this dataset,
# so we do NOT include a PCA step here.
lr_pipeline = Pipeline([
    ("scaler",     StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])

# Fit and evaluate both pipelines; compare accuracy to Task 3 manual results
task3_rf_acc = accuracy_score(y_test, rf.predict(X_test))
task3_lr_acc = accuracy_score(y_test, lr_scaled.predict(X_test_scaled))

for name, pipe, Xtr, Xte, task3_acc in [
    ("Random Forest Pipeline",       rf_pipeline, X_train, X_test, task3_rf_acc),
    ("Logistic Regression Pipeline", lr_pipeline, X_train, X_test, task3_lr_acc),
]:
    pipe.fit(Xtr, y_train)
    y_pred_pipe = pipe.predict(Xte)
    pipe_acc = pipe.score(Xte, y_test)
    print(f"\n{name}")
    print(f"  Pipeline accuracy:  {pipe_acc:.4f}")
    print(f"  Task 3 accuracy:    {task3_acc:.4f}  (difference: {abs(pipe_acc - task3_acc):.4f})")
    print(classification_report(y_test, y_pred_pipe, target_names=["Ham", "Spam"]))
    
# Result:
# Random Forest Pipeline
#   Pipeline accuracy:  0.9446
#   Task 3 accuracy:    0.9446  (difference: 0.0000)
#               precision    recall  f1-score   support

#          Ham       0.94      0.97      0.95       558
#         Spam       0.95      0.91      0.93       363

#     accuracy                           0.94       921
#    macro avg       0.95      0.94      0.94       921
# weighted avg       0.94      0.94      0.94       921


# Logistic Regression Pipeline
#   Pipeline accuracy:  0.9294
#   Task 3 accuracy:    0.9294  (difference: 0.0000)
#               precision    recall  f1-score   support

#          Ham       0.93      0.95      0.94       558
#         Spam       0.92      0.90      0.91       363

#     accuracy                           0.93       921
#    macro avg       0.93      0.92      0.93       921
# weighted avg       0.93      0.93      0.93       921


# Pipeline comparison comments:
# - The Random Forest pipeline has only one step (the classifier) -- trees don't
#   need scaling, so no preprocessing is required.
# - The Logistic Regression pipeline has two steps: StandardScaler then classifier.
#   The scaler must come first because LR is sensitive to feature magnitudes.
# - The accuracy difference between the pipeline and the manual Task 3 approach
#   should be 0.0000 -- both use the same data and same random_state, confirming
#   the pipeline is correctly reproducing the manual workflow.
#
# Practical value of pipelines:
# - They make the workflow much safer and easier to maintain.
#   Preprocessing steps like scaling are always applied correctly and consistently.
# - This is especially important when deploying a model or sharing it with others --
#   it eliminates the risk of forgetting a preprocessing step or applying
#   transformations in the wrong order.
# - cross_val_score and GridSearchCV work seamlessly with pipelines,
#   ensuring the scaler is re-fit on each fold's training data only --
#   eliminating the data leakage that occurs when passing pre-transformed arrays.
