# for the project exercise
# ---Part 2: Mini-Project: World Happiness Pipeline---
#
# Pre-preprocessing observations (from inspecting raw files in a text editor):
#   - Columns are separated by semicolons (;), not commas
#   - Decimal values use a comma (,) as the separator (European format), e.g. 7,59
#   - pd.read_csv() needs: sep=";", decimal=","
#   - Each file has no "year" column -- we must add it after loading

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so plots save without a display
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from prefect import task, flow, get_run_logger

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "happiness_input")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
YEARS = list(range(2015, 2025))

# Mapping of known alternative column names → canonical names used throughout the pipeline.
# Add entries here if future dataset files use different column names.
COLUMN_ALIASES = {
    "ladder_score":       "happiness_score",
    "score":              "happiness_score",
    "happiness.score":    "happiness_score",
    "region":             "regional_indicator",
    "country_name":       "country",
    "country_or_region":  "country",
    "gdp_per_capita_ppp": "gdp_per_capita",
    "economy_(gdp_per_capita)": "gdp_per_capita",
    "health_(life_expectancy)": "healthy_life_expectancy",
    "trust_(government_corruption)": "perceptions_of_corruption",
    "family":             "social_support",
}


# ---------------------------------------------------------------------------
# Task 1: Load Multiple Years of Data
# ---------------------------------------------------------------------------
@task(retries=3, retry_delay_seconds=2)
def load_data(years, data_dir, output_dir):
    logger = get_run_logger()
    frames = []
    for year in years:
        path = os.path.join(data_dir, f"world_happiness_{year}.csv")

        # --- error handling: skip missing or unreadable files ---
        if not os.path.exists(path):
            logger.warning(f"File not found, skipping: {path}")
            continue
        try:
            df = pd.read_csv(path, sep=";", decimal=",")
        except Exception as e:
            logger.warning(f"Could not read {path}: {e} — skipping")
            continue

        # Normalise column names to snake_case
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Apply alias mapping so every year uses the same canonical column names
        df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns},
                  inplace=True)

        df["year"] = year
        frames.append(df)
        logger.info(f"Loaded {year}: {len(df)} rows, columns: {df.columns.tolist()}")

    if not frames:
        raise ValueError(f"No data files could be loaded from {data_dir}. "
                         "Check that the happiness_input folder exists and contains CSV files.")

    merged = pd.concat(frames, ignore_index=True)

    # Drop any accidental duplicate columns (safety net)
    merged = merged.loc[:, ~merged.columns.duplicated()]

    out_path = os.path.join(output_dir, "merged_happiness.csv")
    merged.to_csv(out_path, index=False)
    logger.info(f"Saved merged dataset to {out_path}  ({len(merged)} rows total)")
    logger.info(f"Final columns: {merged.columns.tolist()}")
    return merged


# ---------------------------------------------------------------------------
# Task 2: Descriptive Statistics
# ---------------------------------------------------------------------------
@task
def descriptive_stats(df):
    logger = get_run_logger()

    score = df["happiness_score"]
    logger.info(f"Overall happiness_score -- mean: {score.mean():.4f}, "
                f"median: {score.median():.4f}, std: {score.std():.4f}")

    logger.info("--- Mean happiness score by YEAR ---")
    for year, mean_val in df.groupby("year")["happiness_score"].mean().items():
        logger.info(f"  {year}: {mean_val:.4f}")

    logger.info("--- Mean happiness score by REGION ---")
    region_col = "regional_indicator"
    for region, mean_val in (
        df.groupby(region_col)["happiness_score"].mean()
          .sort_values(ascending=False).items()
    ):
        logger.info(f"  {region}: {mean_val:.4f}")

    return df


# ---------------------------------------------------------------------------
# Task 3: Visual Exploration
# ---------------------------------------------------------------------------
@task
def visual_exploration(df, output_dir):
    logger = get_run_logger()

    # 1. Histogram of all happiness scores
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["happiness_score"].dropna(), bins=30, edgecolor="black", color="steelblue")
    ax.set_title("Distribution of Happiness Scores (2015–2024)")
    ax.set_xlabel("Happiness Score")
    ax.set_ylabel("Count")
    path = os.path.join(output_dir, "happiness_histogram.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    # 2. Boxplot by year
    fig, ax = plt.subplots(figsize=(12, 6))
    years_sorted = sorted(df["year"].unique())
    data_by_year = [df[df["year"] == y]["happiness_score"].dropna().values for y in years_sorted]
    ax.boxplot(data_by_year, tick_labels=years_sorted)
    ax.set_title("Happiness Score Distribution by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Happiness Score")
    path = os.path.join(output_dir, "happiness_by_year.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    # 3. Scatter: GDP per capita vs happiness score
    gdp_col = "gdp_per_capita"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df[gdp_col].dropna(), df["happiness_score"].dropna(),
               alpha=0.3, s=10, color="darkorange")
    ax.set_title("GDP per Capita vs Happiness Score")
    ax.set_xlabel("GDP per Capita")
    ax.set_ylabel("Happiness Score")
    path = os.path.join(output_dir, "gdp_vs_happiness.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    # 4. Correlation heatmap
    numeric_df = df.select_dtypes(include="number").drop(columns=["ranking", "year"],
                                                         errors="ignore")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Pearson Correlation Heatmap")
    path = os.path.join(output_dir, "correlation_heatmap.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


# ---------------------------------------------------------------------------
# Task 4: Hypothesis Testing
# ---------------------------------------------------------------------------
@task
def hypothesis_testing(df):
    logger = get_run_logger()

    # Test 1: 2019 vs 2020 (pre-pandemic vs pandemic onset)
    pre  = df[df["year"] == 2019]["happiness_score"].dropna()
    post = df[df["year"] == 2020]["happiness_score"].dropna()
    t_stat, p_val = stats.ttest_ind(pre, post)
    logger.info("=== T-test: 2019 vs 2020 happiness scores ===")
    logger.info(f"  Mean 2019: {pre.mean():.4f}   Mean 2020: {post.mean():.4f}")
    logger.info(f"  t-statistic: {t_stat:.4f}   p-value: {p_val:.4f}")
    if p_val < 0.05:
        direction = "lower" if post.mean() < pre.mean() else "higher"
        logger.info(
            f"  Result (p < 0.05): The difference IS statistically significant. "
            f"Global happiness was significantly {direction} in 2020 than in 2019, "
            f"suggesting the pandemic onset was associated with a measurable change "
            f"in reported well-being."
        )
    else:
        logger.info(
            "  Result (p >= 0.05): The difference is NOT statistically significant. "
            "We cannot conclude that the pandemic onset changed mean global happiness scores."
        )

    # Test 2: Western Europe vs Sub-Saharan Africa (expected to differ strongly)
    region_col = "regional_indicator"
    we   = df[df[region_col] == "Western Europe"]["happiness_score"].dropna()
    ssa  = df[df[region_col] == "Sub-Saharan Africa"]["happiness_score"].dropna()
    t2, p2 = stats.ttest_ind(we, ssa)
    logger.info("=== T-test: Western Europe vs Sub-Saharan Africa ===")
    logger.info(f"  Mean Western Europe: {we.mean():.4f}   Me an Sub-Saharan Africa: {ssa.mean():.4f}")
    logger.info(f"  t-statistic: {t2:.4f}   p-value: {p2:.6f}")
    if p2 < 0.05:
        logger.info(
            "  Result (p < 0.05): The difference IS statistically significant. "
            "Western Europe reports substantially higher happiness scores than "
            "Sub-Saharan Africa across all years in the dataset."
        )
    else:
        logger.info("  Result (p >= 0.05): No statistically significant difference found.")

    return {"t_2019_2020": (t_stat, p_val), "t_we_ssa": (t2, p2)}


# ---------------------------------------------------------------------------
# Task 5: Correlation and Multiple Comparisons (Bonferroni)
# ---------------------------------------------------------------------------
@task
def correlation_analysis(df):
    logger = get_run_logger()

    numeric_df = df.select_dtypes(include="number").drop(
        columns=["ranking", "year", "happiness_score"], errors="ignore"
    )
    explanatory_vars = numeric_df.columns.tolist()
    score = df["happiness_score"].dropna()
    alpha = 0.05
    results = {}

    logger.info("=== Pearson correlations with happiness_score ===")
    for var in explanatory_vars:
        col = df[var].dropna()
        common_idx = score.index.intersection(col.index)
        r, p = stats.pearsonr(score.loc[common_idx], col.loc[common_idx])
        results[var] = (r, p)
        logger.info(f"  {var}: r={r:.4f}, p={p:.4f}")

    n_tests = len(results)
    adjusted_alpha = alpha / n_tests
    logger.info(f"\nNumber of tests: {n_tests}   Bonferroni-adjusted alpha: {adjusted_alpha:.4f}")

    logger.info("--- Significant at original alpha=0.05 ---")
    for var, (r, p) in results.items():
        if p < alpha:
            logger.info(f"  {var}: r={r:.4f}, p={p:.4f}  ✓")

    logger.info("--- Significant after Bonferroni correction ---")
    significant_after = {}
    for var, (r, p) in results.items():
        if p < adjusted_alpha:
            logger.info(f"  {var}: r={r:.4f}, p={p:.4f}  ✓")
            significant_after[var] = (r, p)

    if significant_after:
        strongest = max(significant_after, key=lambda v: abs(significant_after[v][0]))
        logger.info(f"Strongest correlate after correction: {strongest} "
                    f"(r={significant_after[strongest][0]:.4f})")
    else:
        strongest = max(results, key=lambda v: abs(results[v][0]))
        logger.info(f"No variable survives Bonferroni; strongest overall: {strongest} "
                    f"(r={results[strongest][0]:.4f})")

    return results, strongest


# ---------------------------------------------------------------------------
# Task 6: Summary Report
# ---------------------------------------------------------------------------
@task
def summary_report(df, hypothesis_results, corr_results_tuple):
    logger = get_run_logger()
    corr_results, strongest_corr = corr_results_tuple

    n_countries = df["country"].nunique()
    n_years     = df["year"].nunique()
    logger.info(f"Total countries in dataset: {n_countries}   Years: {n_years} (2015–2024)")

    region_col = "regional_indicator"
    region_means = df.groupby(region_col)["happiness_score"].mean().sort_values(ascending=False)
    top3    = region_means.head(3)
    bottom3 = region_means.tail(3)
    logger.info("Top 3 regions by mean happiness score:")
    for region, mean_val in top3.items():
        logger.info(f"  {region}: {mean_val:.4f}")
    logger.info("Bottom 3 regions by mean happiness score:")
    for region, mean_val in bottom3.items():
        logger.info(f"  {region}: {mean_val:.4f}")

    t_stat, p_val = hypothesis_results["t_2019_2020"]
    if p_val < 0.05:
        logger.info(
            f"2019 vs 2020 t-test: statistically significant (p={p_val:.4f}). "
            "The pandemic onset in 2020 was associated with a measurable shift in "
            "global happiness scores."
        )
    else:
        logger.info(
            f"2019 vs 2020 t-test: NOT significant (p={p_val:.4f}). "
            "No statistically significant change in global happiness scores was detected "
            "between 2019 and 2020."
        )

    r_val = corr_results[strongest_corr][0]
    logger.info(
        f"Variable most strongly correlated with happiness (after Bonferroni): "
        f"{strongest_corr} (r={r_val:.4f})"
    )


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------
@flow
def happiness_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df            = load_data(YEARS, DATA_DIR, OUTPUT_DIR)
    df            = descriptive_stats(df)
    visual_exploration(df, OUTPUT_DIR)
    hyp_results   = hypothesis_testing(df)
    corr_tuple    = correlation_analysis(df)
    summary_report(df, hyp_results, corr_tuple)


if __name__ == "__main__":
    happiness_pipeline()

