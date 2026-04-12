# for the warmup exercises
# --- Pandas ---
# Pandas Q1: Create the following DataFrame and print the first three rows, the shape, and the data types of each column.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)
print("--- Pandas Q1 ---")
print(f"Num Rows: {len(df)}")

print(f"First 3 Rows: {df.head(3)}")
print(f"Shape: {df.shape}")
print(f"Data Types: {df.dtypes}")

# Pandas Q2: Using the DataFrame from Q1, filter the rows to show only students who passed and have a grade above 80. Print the result.
filtered = df[(df["passed"] == True) & (df["grade"] > 80)]
print("--- Pandas Q2 ---")
print(f"Filtered Rows: {filtered}")

# Pandas Q3: Add a new column called "grade_curved" that adds 5 points to each student's grade. Print the updated DataFrame (all columns, all rows).
df["grade_curved"] = df["grade"] + 5
print("--- Pandas Q3 ---")
print(f"Updated DataFrame: {df}")   

# Pandas Q4: Add a new column called "name_upper" that contains each student's name in uppercase, using the .str accessor. Print the "name" and "name_upper" columns together.
df["name_upper"] = df["name"].str.upper()
print("--- Pandas Q4 ---")
print(f"Name and Name Upper: {df[['name', 'name_upper']]}")

# Pandas Q5: Group the DataFrame by "city" and compute the mean grade for each city. Print the result.
mean_grades = df.groupby("city")["grade"].mean()
print("--- Pandas Q5 ---")
print(f"Mean Grades by City: {mean_grades}")

# Pandas Q6: Replace the value "Austin" in the "city" column with "Houston". Print the "name" and "city" columns to confirm the change.
df["city"] = df["city"].replace("Austin", "Houston")
print("--- Pandas Q6 ---")
print(f"Name and City After Replacement: {df[['name', 'city']]}")

# Pandas Q7: Sort the DataFrame by "grade" in descending order and print the top 3 rows.
sorted_df = df.sort_values(by="grade", ascending=False)
print("--- Pandas Q7 ---")
print(f"Top 3 Rows by Grade: {sorted_df.head(3)}")

# --- NumPy Review ---
# NumPy Q1: Create a 1D NumPy array from the list [10, 20, 30, 40, 50]. Print its shape, dtype, and ndim.

arr = np.array([10, 20, 30, 40, 50])
print("--- NumPy Q1 ---")
print(f"Shape: {arr.shape}")
print(f"Dtype: {arr.dtype}")
print(f"Ndim: {arr.ndim}")   

# NumPy Q2: Create the following 2D array and print its shape and size (total number of elements).
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print("--- NumPy Q2 ---")
print(f"Shape: {arr.shape}")
print(f"Size: {arr.size}")

# NumPy Q3: Using the 2D array from Q2, slice out the top-left 2x2 block and print it. The expected result is [[1, 2], [4, 5]].
top_left_2x2 = arr[:2, :2]
print("--- NumPy Q3 ---")
print(f"Top-left 2x2 Block: {top_left_2x2}")

# NumPy Q4:Create a 3x4 array of zeros using a built-in command. Then create a 2x5 array of ones using a built-in command. Print both.
zeros_array = np.zeros((3, 4))
ones_array = np.ones((2, 5))
print("--- NumPy Q4 ---")
print(f"3x4 Zeros Array:\n{zeros_array}")
print(f"2x5 Ones Array:\n{ones_array}")

# NumPy Q5: Create an array using np.arange(0, 50, 5). First, think about what you expect it to look like. Then, print the array, its shape, mean, sum, and standard deviation.
arr = np.arange(0, 50, 5)
print("--- NumPy Q5 ---")
print(f"Array:\n{arr}")
print(f"Shape: {arr.shape}")
print(f"Mean: {arr.mean()}")
print(f"Sum: {arr.sum()}")
print(f"Standard Deviation: {arr.std()}")

# NumPy Q6: Generate an array of 200 random values drawn from a normal distribution with mean 0 and standard deviation 1 (use np.random.normal()). Print the mean and standard deviation of the result.
random_values = np.random.normal(loc=0, scale=1, size=200)
print("--- NumPy Q6 ---")
print(f"Random Values Mean: {random_values.mean()}")
print(f"Random Values Standard Deviation: {random_values.std()}")

# --- Matplotlib Review ---
# Matplotlib Q1: Plot the following data as a line plot. Add a title "Squares", x-axis label "x", and y-axis label "y".
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

print("--- Matplotlib Q1 ---")
plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Matplotlib Q2: Create a bar plot for the following subject scores. Add a title "Subject Scores" and label both axes.
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

print("--- Matplotlib Q2 ---")
plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()

# Matplotlib Q3: Plot the two datasets below as a scatter plot on the same figure. Use different colors for each, add a legend, and label both axes.
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

print("--- Matplotlib Q3 ---")
plt.scatter(x1, y1, color='blue', label='Dataset 1')
plt.scatter(x2, y2, color='red', label='Dataset 2')
plt.title("Scatter Plot of Two Datasets")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Matplotlib Q4: Use plt.subplots() to create a figure with 1 row and 2 subplots side by side. In the left subplot, plot x vs y from Q1 as a line. In the right subplot, plot the subjects and scores from Q2 as a bar plot. Add a title to each subplot and call plt.tight_layout() before showing.
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Left subplot
axs[0].plot(x, y)
axs[0].set_title("Squares")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")

# Right subplot
axs[1].bar(subjects, scores)
axs[1].set_title("Subject Scores")
axs[1].set_xlabel("Subjects")
axs[1].set_ylabel("Scores")

print("--- Matplotlib Q4 ---")
plt.tight_layout()
plt.show()

#  --- Descriptive Statistics Review ---
# Descriptive Statistics Q1: Given the list below, use NumPy to compute and print the mean, median, variance, and standard deviation. Label each printed value.

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
data_array = np.array(data)
print("--- Descriptive Statistics Q1 ---")
print(f"Mean: {data_array.mean()}")
print(f"Median: {np.median(data_array)}")
print(f"Variance: {data_array.var()}")
print(f"Standard Deviation: {data_array.std()}")

# Descriptive Statistics Q2: Generate 500 random values from a normal distribution with mean 65 and standard deviation 10 (use np.random.normal(65, 10, 500)). Plot a histogram with 20 bins. Add a title "Distribution of Scores" and label both axes.
random_scores = np.random.normal(65, 10, 500)
plt.hist(random_scores, bins=20)
print("--- Descriptive Statistics Q2 ---")
plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")
plt.show()

# Descriptive Statistics Q3: reate a boxplot comparing the two groups below. Label each box ("Group A" and "Group B") and add a title "Score Comparison".
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
# Hint: pass labels=["Group A", "Group B"] to plt.boxplot().
# MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
print("--- Descriptive Statistics Q3 ---")
plt.boxplot([group_a, group_b], tick_labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Scores")
plt.show()

# Descriptive Statistics Q4: You are given two datasets: one normally distributed and one 'exponential' distribution.
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)
# Create side-by-side boxplots comparing the two distributions. Label each boxplot appropriately ("Normal" and "Exponential") and add a title "Distribution Comparison".
# Then, add a comment in your code briefly noting which distribution is more skewed, and which descriptive statistic (mean or median) would provide a more appropriate measure of central tendency for each distribution.
print("--- Descriptive Statistics Q4 ---")
plt.boxplot([normal_data, skewed_data], tick_labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.ylabel("Values")
plt.show()

# Descriptive Statistics Q5: Print the mean, median, and mode of the following:
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

print("--- Descriptive Statistics Q5 ---")
print(f"Data1 Mean: {np.mean(data1)}")
print(f"Data1 Median: {np.median(data1)}")
mode1 = stats.mode(data1)
print(f"Data1 Mode: {mode1}")

print(f"Data2 Mean: {np.mean(data2)}")
print(f"Data2 Median: {np.median(data2)}")
mode2 = stats.mode(data2)
print(f"Data2 Mode: {mode2}")
# Why are the median and mean so different for data2? Add your answer as a comment in the code.
# Answer: The mean is much higher than the median for data2 because of the presence of the outlier value 150, 
# which skews the distribution and pulls the mean upwards, while the median remains unchanged.

#  --- Hypothesis Testing Review ---
# Hypothesis Testing Q1: Run an independent samples t-test on the two groups below. Print the t-statistic and p-value.

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_val = stats.ttest_ind(group_a, group_b)

print("--- Hypothesis Testing Q1 ---")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_val}")

# Hypothesis Testing Q2: Using the p-value from Q1, write an if/else statement that prints whether the result is statistically significant at alpha = 0.05.
print("--- Hypothesis Testing Q2 ---")
if p_val < 0.05:
    print("The result is statistically significant.")
else:
    print("The result is not statistically significant.")

# Hypothesis Testing Q3: Run a paired t-test on the before/after scores below (the same students measured twice). Print the t-statistic and p-value.

before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat, p_val = stats.ttest_rel(before, after)

print("--- Hypothesis Testing Q3 ---")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_val}")

# Hypothesis Testing Q4: Run a one-sample t-test to check whether the mean of scores is significantly different from a national benchmark of 70. Print the t-statistic and p-value.
scores = [72, 68, 75, 70, 69, 74, 71, 73]
t_stat, p_val = stats.ttest_1samp(scores, 70)

print("--- Hypothesis Testing Q4 ---")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_val}")

# Hypothesis Testing Q5: Re-run the test from Q1 as a one-tailed test to check whether group_a scores are less than group_b scores. Print the resulting p-value. Use the alternative parameter.
t_stat, p_val = stats.ttest_ind(group_a, group_b, alternative='less')

print("--- Hypothesis Testing Q5 ---")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_val}")

# Hypothesis Testing Q6: Write a plain-language conclusion for the result of Q1 (do not just say "reject the null hypothesis"). Format it as a print() statement. Your conclusion should mention the direction of the difference and whether it is likely due to chance.
print("--- Hypothesis Testing Q6 ---")
if p_val < 0.05:
    print("There is a statistically significant difference between group_a and group_b scores, with group_a having lower scores than group_b. This difference is unlikely to be due to chance.")
else:
    print("There is no statistically significant difference between group_a and group_b scores, suggesting that any observed difference could be due to chance.")

#  --- Correlation Review ---
# Correlation Q1: Compute the Pearson correlation between x and y below using np.corrcoef(). Print the full correlation matrix, then print just the correlation coefficient (the value at position [0, 1]).
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
correlation_matrix = np.corrcoef(x, y)
print("--- Correlation Q1 ---")
print(f"Correlation Matrix:\n{correlation_matrix}")
print(f"Correlation Coefficient: {correlation_matrix[0, 1]}")   
# What do you expect the correlation to be, and why? Add your answer as a comment in the code.
# Answer: I expect the correlation to be 1 because y is a perfect linear transformation of x (y = 2x).

# Correlation Q2: Use pearsonr() from scipy.stats to compute the correlation between x and y below. Print both the correlation coefficient and the p-value.
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]
corr_coef, p_val = pearsonr(x, y)
print("--- Correlation Q2 ---")
print(f"Correlation Coefficient: {corr_coef}")
print(f"P-value: {p_val}")

# Correlation Q3: Create the following DataFrame and use df.corr() to compute the correlation matrix. Print the result.
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
correlation_matrix = df.corr()
print("--- Correlation Q3 ---")
print(f"Correlation Matrix:\n{correlation_matrix}")

# Correlation Q4: Create a scatter plot of x and y below, which have a negative relationship. Add a title "Negative Correlation" and label both axes.
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]
print("--- Correlation Q4 ---")
plt.scatter(x, y)
plt.title("Negative Correlation")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# Correlation Q5: Using the correlation matrix from Q3, create a heatmap with sns.heatmap(). Pass annot=True so the correlation values appear in each cell, and add a title "Correlation Heatmap".
print("--- Correlation Q5 ---")
sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation Heatmap")
plt.show()

# --- Pipelines ---
# Pipelines Q1: A data pipeline is a sequence of processing steps where each step takes in data, transforms it, and passes the result to the next. You don't need a special framework to build one -- chaining plain functions together is often enough.
# Given the array below, which contains some missing values scattered throughout:
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
# Implement the following three functions and then connect them in a data_pipeline() function.
    # create_series(arr) : takes a NumPy array and returns a pandas Series with the name "values".
    # clean_data(series) : takes the Series, removes any NaN values using .dropna(), and returns the cleaned Series.
    # summarize_data(series) -- takes the cleaned Series and returns a dictionary with four keys: "mean", "median", "std", and "mode". For mode, use series.mode()[0] to get a single value.
    # data_pipeline(arr) -- calls the three functions above in sequence and returns the summary dictionary.
# Call data_pipeline(arr) and print each key and its value from the result.
# This is the last answer to put in warmups_01.py. Congrats!!!
# The next question will be in prefect_warmup.py, but will implement the same functionality using Prefect instead of plain Python.

def create_series(arr):
    return pd.Series(arr, name="values")

def clean_data(series):
    return series.dropna()

def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

def data_pipeline(arr):
    series = create_series(arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)
    return summary

result = data_pipeline(arr)
for key, value in result.items():
    print(f"{key}: {value}")