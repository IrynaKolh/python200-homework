# for the warmup exercises
# --- scikit-learn API ---
# scikit-learn Q1: The core pattern in scikit-learn is create → fit → predict. Practice it here with a simple dataset: years of work experience versus annual salary.
import os
os.environ["OMP_NUM_THREADS"] = "1"  # suppress KMeans memory-leak warning on Windows with MKL
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so plots save without a display
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])
#  Create a LinearRegression model, fit it to this data, and then predict the salary for someone with 4 years of experience and someone with 8 years. Print the slope (model.coef_[0]), the intercept (model.intercept_), and the two predictions. Label each printed value.

model = LinearRegression()
model.fit(years, salary)

pred_4_years = model.predict(np.array([[4]]))
pred_8_years = model.predict(np.array([[8]]))

print("--- scikit-learn Q1 ---")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted salary for 4 years of experience:", pred_4_years[0])
print("Predicted salary for 8 years of experience:", pred_8_years[0])

# scikit-learn Q2: scikit-learn requires the feature array X to be 2D even when you only have one feature. Start with this 1D array:
x = np.array([10, 20, 30, 40, 50])
# Print its shape. Use .reshape() to convert it to a 2D array and print the new shape. Add a comment explaining, in your own words, why scikit-learn needs X to be 2D.
print("\n--- scikit-learn Q2 ---")
print("Original shape:", x.shape)
x_reshaped = x.reshape(-1, 1)
print("Reshaped shape:", x_reshaped.shape)
# scikit-learn needs X to be 2D because it is designed to handle multiple features (columns) and multiple samples (rows). Even if you only have one feature, it still requires a 2D array to maintain this structure.

# scikit-learn Q3: K-Means is an unsupervised algorithm that follows the same create → fit → predict pattern as everything else in scikit-learn. Use the code below to generate a synthetic dataset with three natural clusters:
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
# Create a KMeans model with n_clusters=3 and random_state=42, fit it to X_clusters, and predict a cluster label for each point. Print the cluster centers (kmeans.cluster_centers_) and how many points fell into each cluster using np.bincount(labels).
# Then create a scatter plot coloring each point by its cluster label, plot the cluster centers as black X's, add a title and axis labels. Save the figure to outputs/kmeans_clusters.png.
print("\n--- scikit-learn Q3 ---")
print(X_clusters.shape)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters) 
print("Cluster centers:", kmeans.cluster_centers_)
print("Points in each cluster:", np.bincount(labels))

plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker='X')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/kmeans_clusters.png")
plt.close()
print("Plot saved to outputs/kmeans_clusters.png")


# --- Linear Regression ---
# Linear Regression Q1: Before fitting anything, look at the data. Create a scatter plot of age on the x-axis and cost on the y-axis. Color the points by smoker status by passing c=smoker and cmap="coolwarm" to plt.scatter(). Add a title "Medical Cost vs Age", label both axes, and save to outputs/cost_vs_age.png.
# Add a comment describing what you see. Are there two distinct groups visible? What does that suggest about the smoker variable?

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

print("\n--- Linear Regression Q1 ---")
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost")
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/cost_vs_age.png")
plt.close()
print("Plot saved to outputs/cost_vs_age.png")
# The scatter plot shows two distinct groups of points. The points colored in red (smok

# Linear Regression Q2: Split the data into training and test sets using age as the only feature, an 80/20 split, and random_state=42. Reshape age to a 2D array before using it as X. Print the shapes of all four arrays.

print("\n--- Linear Regression Q2 ---")
X_train, X_test, y_train, y_test = train_test_split(age.reshape(-1, 1), cost, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Linear Regression Q3: Fit a LinearRegression model to your training data from Question 2. Print the slope and intercept. Then predict on the test set and print:
# RMSE: np.sqrt(np.mean((y_pred - y_test) ** 2))
# R² on the test set: model.score(X_test, y_test)
# Add a comment interpreting the slope in plain English -- what does it mean for medical costs?

print("\n--- Linear Regression Q3 ---")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("Slope:", lr_model.coef_[0])
print("Intercept:", lr_model.intercept_)

y_pred = lr_model.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2   = lr_model.score(X_test, y_test)
print("RMSE:", rmse)
print("R²:", r2)

# Answer: The slope means that for each additional year of age, the predicted medical cost
# increases by approximately that many dollars. For example, a slope of ~200 means
# a patient who is 1 year older is expected to cost about $200 more per year.
# However, since smoker status is not included in this model, the R² will be
# relatively low -- the model is missing a major predictor of medical costs.

# Linear Regression Q4: Now add smoker as a second feature and fit a new model.
print("\n--- Linear Regression Q4 ---")
X_full = np.column_stack([age, smoker])
# Split, fit, and print the test R². Compare it to the R² from Question 3 -- does adding the smoker flag help? Print both coefficients:
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, cost, test_size=0.2, random_state=42)
model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)
print("Test R²:", model_full.score(X_test_full, y_test_full))
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])
# Add a comment interpreting the smoker coefficient: what does it represent in practical terms?
# Answer: The smoker coefficient represents the additional cost associated with being a smoker, holding age constant. 
# For example, if the smoker coefficient is around 15000, it means that being a smoker is associated with 
# an increase of about $15,000 in medical costs compared to a non-smoker of the same age. 
# This shows that smoking has a significant impact on medical costs, which is 
# why including it as a feature greatly improves the model's performance (as seen in the higher R²).

# Linear Regression Q5: A predicted vs actual plot is a standard tool for evaluating regression models. Each test observation becomes a dot: the model's prediction goes on the x-axis, the true value goes on the y-axis. A perfect model would place every point on the diagonal line where predicted equals actual.
# Using the two-feature model from Linear Regression Question 4, create this plot for the test set. Add a diagonal reference line, a title "Predicted vs Actual", labeled axes, and save to outputs/predicted_vs_actual.png.
# Add a comment: what does it mean when a point falls above the diagonal? What about below?
print("\n--- Linear Regression Q5 ---")
y_pred_full = model_full.predict(X_test_full)
plt.scatter(y_pred_full, y_test_full)
plt.plot([y_test_full.min(), y_test_full.max()], [y_test_full.min(), y_test_full.max()], 'k--', lw=2)
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/predicted_vs_actual.png")
plt.close()
print("Plot saved to outputs/predicted_vs_actual.png")
# A point above the diagonal means the model underpredicted the cost (the actual cost is higher than the predicted cost). A point below the diagonal means the model overpredicted the cost (
# the actual cost is lower than the predicted cost). The closer the points are to the diagonal, the better the model's predictions match the actual values.
