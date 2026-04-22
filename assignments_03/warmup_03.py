# for the warmup exercises
# Part 1: Warmup Exercises

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Preprocessing ---
# Preprocessing Q1: Split X and y into training and test sets using an 80/20 split 
# with stratify=y and random_state=42. Print the shapes of all four arrays.
print("\n--- Preprocessing Q1 ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Preprocessing Q2: Fit a StandardScaler on X_train and use it to transform both X_train and X_test. 
# Print the mean of each column in X_train_scaled -- they should all be very close to 0. 
# Add a comment explaining in one sentence why you fit the scaler on X_train only.
print("\n--- Preprocessing Q2 ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("Means of columns in X_train_scaled:", X_train_scaled.mean(axis=0))
# We fit the scaler on X_train only to avoid data leakage from the test set, 
# ensuring that the scaling parameters are derived solely from the training data.

# --- KNN ---
# KNN Q1: Build a KNeighborsClassifier with n_neighbors=5, fit it on the unscaled training data (X_train), 
# and predict on the test set. Print the accuracy score and the full classification report.
print("\n--- KNN Q1 ---")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("KNN Accuracy (unscaled):", accuracy_score(y_test, y_pred))
print("KNN Classification Report (unscaled):\n", classification_report(y_test, y_pred))

# KNN Q2: Repeat KNN Question 1 using the scaled data (X_train_scaled, X_test_scaled). 
# Print the accuracy score. Add a comment: does scaling improve performance, hurt it, or make no difference? 
# Why might that be for this particular dataset?
print("\n--- KNN Q2 ---")
knn.fit(X_train_scaled, y_train)
y_pred_scaled = knn.predict(X_test_scaled)
print("KNN Accuracy (scaled):", accuracy_score(y_test, y_pred_scaled))
# Scaling improves performance for KNN because it is sensitive to the scale of the input features. 
# In this dataset, the features are on different scales, so scaling helps the model to perform better. 

# KNN Q3: Using cross_val_score with cv=5, evaluate the k=5 KNN model on the unscaled training data. 
# Print each fold score, the mean, and the standard deviation. 
# Add a comment: is this result more or less trustworthy than a single train/test split, and why?
print("\n--- KNN Q3 ---")
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
print("KNN Cross-Validation Scores (unscaled):", cv_scores)
print("KNN Cross-Validation Mean (unscaled):", cv_scores.mean())
print("KNN Cross-Validation Std Dev (unscaled):", cv_scores.std())
# This result is more trustworthy than a single train/test split because it uses multiple train/test splits to evaluate the model, providing a better estimate of its performance.

# KNN Q4: Loop over k values [1, 3, 5, 7, 9, 11, 13, 15]. 
# For each, compute 5-fold cross-validation accuracy on the unscaled training data and print k and the mean CV score. 
# Add a comment identifying which k you would choose and why.
print("\n--- KNN Q4 ---")
for k in range(1, 16, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"KNN Cross-Validation Mean (unscaled) for k={k}:", cv_scores.mean())
# I would choose k=5 because it typically provides a good balance between bias and variance, and in this case, 
# it has the highest mean CV score among the tested k values.

# --- Classifier Evaluation ---
# Classifier Evaluation Q1: Using your predictions from KNN Question 1, 
# create a confusion matrix and display it with ConfusionMatrixDisplay, 
# passing display_labels=iris.target_names. Save the figure to outputs/knn_confusion_matrix.png. 
# Add a comment: which pair of species does the model most often confuse (if any)?
print("\n--- Classifier Evaluation Q1 ---")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("KNN Confusion Matrix (unscaled)")
plt.savefig("outputs/knn_confusion_matrix.png")
plt.show()
# The model most often confuses the species 'versicolor' and 'virginica', 
# as indicated by the off-diagonal values in the confusion matrix.

# --- The sklearn API: Decision Trees ---
# The sklearn API: Decision Trees Q1: Create a DecisionTreeClassifier(max_depth=3, random_state=42), 
# fit it on the unscaled training data, and predict on the test set. 
# Print the accuracy score and classification report. 
# Add a comment comparing the Decision Tree accuracy to KNN. 
# Then add a second comment: given that Decision Trees don't rely on distance calculations, 
# would scaled vs. unscaled data affect the result?
print("\n--- Decision Trees Q1 ---")
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy (unscaled):", accuracy_score(y_test, y_pred_dt))
print("Decision Tree Classification Report (unscaled):\n", classification_report(y_test, y_pred_dt))
# The Decision Tree accuracy is comparable to KNN, but it may be less sensitive to the specific distribution of the data.
# Given that Decision Trees don't rely on distance calculations, scaling the data is unlikely to have a significant impact on the results.

# --- Logistic Regression and Regularization ---
# Logistic Regression Q1: Train three logistic regression models on the scaled Iris data, 
# identical in every way except for the C parameter: C=0.01, C=1.0, and C=100. 
# Use max_iter=1000 and solver='liblinear' for all three. 
# For each model, print the C value and the total size of all coefficients using np.abs(model.coef_).sum(). 
# Add a comment: what happens to the total coefficient magnitude as C increases? 
# What does this tell you about what regularization is doing?
print("\n--- Logistic Regression Q1 ---")
for C in [0.01, 1.0, 100]:
    lr = LogisticRegression(C=C, max_iter=1000, solver='liblinear')
    lr.fit(X_train_scaled, y_train)
    total_coef_magnitude = np.abs(lr.coef_).sum()
    print(f"C={C}, Total Coefficient Magnitude: {total_coef_magnitude}")
# As C increases, the total coefficient magnitude also increases. 
# This indicates that regularization is penalizing larger coefficients, 
# and as C increases, the penalty is reduced, allowing the model to fit the training data more closely, 
# which can lead to larger coefficients.

# --- PCA ---
digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting
# PCA Q1: Print the shape of X_digits and images. Then create a 1-row subplot showing one example of each digit class (0-9), 
# using cmap='gray_r' with each digit's label as the title. Save the figure to outputs/sample_digits.png. 
# (gray_r is the reversed grayscale colormap -- it renders higher pixel values as darker, 
# so digits appear as dark ink on a light background, which is more readable than the default.)
print("\n--- PCA Q1 ---")
print("X_digits shape:", X_digits.shape)
print("images shape:", images.shape)
fig, axes = plt.subplots(1, 10, figsize=(15, 3))
for i in range(10):
    axes[i].imshow(images[i], cmap='gray_r')
    axes[i].set_title(f"Digit: {y_digits[i]}")
    axes[i].axis('off')
plt.suptitle("Sample Digits")
plt.tight_layout()
plt.savefig("outputs/sample_digits.png")
plt.show()

# PCA Q2: Fit PCA() on X_digits (with no n_components argument) then get the scores with scores = pca.transform(X_digits). 
# As in the lesson, scores tell you how strongly each component is weighted for each sample -- scores[i, 0] 
# is the weighting for PC1 in sample i, scores[i, 1] is the weighting for PC2, and so on.
# Use scores[:, 0] and scores[:, 1] to make a scatter plot, coloring each point by its digit label and adding a colorbar. 
# Here is the pattern for coloring by a label array and attaching a colorbar:
print("\n--- PCA Q2 ---")
pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)
plt.figure()
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)
plt.colorbar(scatter, label='Digit')
# Save the figure to outputs/pca_2d_projection.png. Add a comment: do same-digit images tend to cluster together in this 2D space?
plt.title("PCA 2D Projection of Digits")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("outputs/pca_2d_projection.png")
plt.show()
# Yes, same-digit images tend to cluster together in this 2D space, 
# indicating that the PCA has captured some of the variance that distinguishes different digit classes.

# PCA Q3: Using the PCA object you fit in Question 2, plot cumulative explained variance vs. number of components 
# using np.cumsum(pca.explained_variance_ratio_). Save to outputs/pca_variance_explained.png. 
# Add a comment: approximately how many components do you need to explain 80% of the variance?
print("\n--- PCA Q3 ---")
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.figure()
plt.plot(cumulative_variance)
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.axhline(y=0.8, color='r', linestyle='--')
plt.savefig("outputs/pca_variance_explained.png")
plt.show()
# Approximately 40 components are needed to explain 80% of the variance.

# PCA Q4:The preprocessing lesson showed that a reconstruction is built by starting from the mean 
# and adding each component weighted by its score. Here is the same idea generalized to n components -- 
# add this function to your file:

def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

# Using this function, the PCA object, and the scores from Question 2, reconstruct the first 5 digits 
# in X_digits using reconstruction through principal components n = 2, 5, 15, and 40.
# Build a grid of subplots where rows correspond to each n value and columns show those 5 digits. 
# Add an "Original" row at the top (use images[i], which is already shaped as (8, 8)). 
# Save to outputs/pca_reconstructions.png.
# Add a comment: at what n do the digits become clearly recognizable, and does that match where the variance curve levels off?
print("\n--- PCA Q4 ---")
n_values = [2, 5, 15, 40]
fig, axes = plt.subplots(len(n_values) + 1, 5, figsize=(15, 10))
# Original row
for i in range(5):
    axes[0, i].imshow(images[i], cmap='gray_r')
    axes[0, i].set_title(f"Original: {y_digits[i]}")
    axes[0, i].axis('off')
# Reconstructed rows
for row, n in enumerate(n_values, start=1):
    for col in range(5):
        reconstructed_image = reconstruct_digit(col, scores, pca, n)
        axes[row, col].imshow(reconstructed_image, cmap='gray_r')
        axes[row, col].set_title(f"n={n}")
        axes[row, col].axis('off')
plt.suptitle("PCA Reconstructions of Digits")
plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png")
plt.show()
# The digits become clearly recognizable at around n=15, which matches where the variance curve starts to level off, indicating that most of the important variance is captured by the first 15 components.

