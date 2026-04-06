# for the prefect pipeline warmup exercise

# --- Pipelines ---
# Pipelines Q2: Rebuild the pipeline from Q1 using Prefect. Copy your three functions from Pipeline Question 1 (create_series, clean_data, summarize_data) into this file and turn them into Prefect tasks using @task.
# Turn data_pipeline() into a Prefect flow using @flow. Inside the flow, call the three tasks in order and return the summary dictionary.
# Add this block at the bottom of the file so the flow runs when you execute the script directly:
from prefect import task, flow
import pandas as pd
import numpy as np

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task
def create_series(arr):
    return pd.Series(arr, name="values")

@task
def clean_data(series):
    return series.dropna()

@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

@flow
def pipeline_flow(arr):
    series = create_series(arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)
    return summary

if __name__ == "__main__":
    result = pipeline_flow(arr)
    for key, value in result.items():
        print(f"{key}: {value}")
# Run your workflow from the terminal: python prefect_warmup.py
# The summary values should match what you got in Question 1.

# Finally, add a comment block at the bottom of prefect_warmup.py answering these two questions:
# This pipeline is simple -- just three small functions on a handful of numbers. Why might Prefect be more overhead than it is worth here?
# Prefect is designed for more complex workflows that involve multiple tasks, dependencies, and potentially long-running processes. In this simple pipeline, the overhead of setting up Prefect tasks and flows may not provide significant benefits, as the operations are straightforward and can be easily executed without the need for orchestration or monitoring features that Prefect offers.

# Describe some realistic scenarios where a framework like Prefect could still be useful, even if the pipeline logic itself stays simple like in this case.
# Some scenarios where Prefect could be useful include:
# - When the pipeline needs to be scaled to handle larger datasets or more complex transformations.
# - If the pipeline requires integration with other systems or services, such as databases or APIs.
# - When there is a need for monitoring, logging, or error handling that goes beyond simple print statements.
# - If the pipeline needs to be scheduled or triggered based on specific events or conditions.