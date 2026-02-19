import json
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def ensure_dirs(out_dir: Path):
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)


def read_metrics(metrics_path: Path) -> dict:
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    # Simple, clean LaTeX table for USP templates
    latex = df.to_latex(
        index=False,
        escape=True,
        float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
        column_format="l" * len(df.columns),
    )
    # Wrap in table environment
    return (
        "\\begin{table}[H]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{latex}\n"
        "\\end{table}\n"
    )


def plot_metric_bars(df: pd.DataFrame, group_col: str, metric: str, out_path: Path, title: str):
    # bar chart
    tmp = df[[group_col, metric]].copy()
    tmp = tmp.sort_values(metric, ascending=False)
    plt.figure()
    plt.bar(tmp[group_col].astype(str), tmp[metric])
    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    # Inputs
    metrics_path = Path(os.environ.get("METRICS_JSON", "outputs/metrics.json"))
    race_path    = Path(os.environ.get("FAIRNESS_RACE", "outputs/fairness_by_race.csv"))
    sex_path     = Path(os.environ.get("FAIRNESS_SEX", "outputs/fairness_by_sex.csv"))

    out_dir = Path(os.environ.get("OUT_DIR", "outputs/results"))
    ensure_dirs(out_dir)

    # --- Global metrics ---
    metrics = read_metrics(metrics_path)

    # Try to be robust to key names (your metrics.json may differ)
    # We'll print the keys into an artifact for traceability.
    with open(out_dir / "artifacts" / "metrics_keys.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(metrics.keys())))

    # Extract common fields if present
    global_rows = []
    for k in ["model", "train_size", "test_size", "split", "accuracy", "precision", "recall", "f1"]:
        if k in metrics:
            global_rows.append([k, metrics[k]])

    # If different naming, try nested structure
    if not global_rows and isinstance(metrics, dict):
        # fallback: flatten one level
        for k, v in metrics.items():
            if isinstance(v, (int, float, str)):
                global_rows.append([k, v])

    df_global = pd.DataFrame(global_rows, columns=["Métrica", "Valor"])
    (out_dir / "tables" / "global_metrics.tex").write_text(
        to_latex_table(
            df_global,
            "Métricas globais de desempenho do modelo baseline",
            "tab:global_metrics",
        ),
        encoding="utf-8",
    )

    # --- Fairness tables ---
    df_race = pd.read_csv(race_path)
    df_sex = pd.read_csv(sex_path)

    # Standardize column names if needed
    # Expected columns: TP,TN,FP,FN,FPR,FNR,TPR,TNR, race/sex, N, gaps...
    def select_cols(df, group_col):
        cols = [group_col, "N", "TP", "TN", "FP", "FN", "TPR", "FPR", "FNR", "TNR"]
        cols = [c for c in cols if c in df.columns]
        return df[cols].copy()

    race_group_col = "race" if "race" in df_race.columns else df_race.columns[-1]
    sex_group_col = "sex" if "sex" in df_sex.columns else df_sex.columns[-1]

    df_race_tbl = select_cols(df_race, race_group_col)
    df_sex_tbl = select_cols(df_sex, sex_group_col)

    (out_dir / "tables" / "fairness_race.tex").write_text(
        to_latex_table(
            df_race_tbl,
            "Métricas por grupo (raça) no conjunto de teste",
            "tab:fairness_race",
        ),
        encoding="utf-8",
    )

    (out_dir / "tables" / "fairness_sex.tex").write_text(
        to_latex_table(
            df_sex_tbl,
            "Métricas por grupo (sexo) no conjunto de teste",
            "tab:fairness_sex",
        ),
        encoding="utf-8",
    )

    # --- Key plots for presentation ---
    # Focus on what makes impact: FPR and FNR (and gaps if available)
    for metric in ["FPR", "FNR", "TPR", "TNR"]:
        if metric in df_race.columns:
            plot_metric_bars(
                df_race,
                race_group_col,
                metric,
                out_dir / "figures" / f"race_{metric}.png",
                f"{metric} por raça",
            )
        if metric in df_sex.columns:
            plot_metric_bars(
                df_sex,
                sex_group_col,
                metric,
                out_dir / "figures" / f"sex_{metric}.png",
                f"{metric} por sexo",
            )

    # If gap columns exist, plot them too (super impactful)
    for gap_col in ["FPR_gap_vs_min", "FNR_gap_vs_min"]:
        if gap_col in df_race.columns:
            plot_metric_bars(
                df_race,
                race_group_col,
                gap_col,
                out_dir / "figures" / f"race_{gap_col}.png",
                f"{gap_col} por raça",
            )
        if gap_col in df_sex.columns:
            plot_metric_bars(
                df_sex,
                sex_group_col,
                gap_col,
                out_dir / "figures" / f"sex_{gap_col}.png",
                f"{gap_col} por sexo",
            )

    # Save curated CSVs (ordered) for traceability
    df_race.sort_values("FPR", ascending=False).to_csv(out_dir / "artifacts" / "race_sorted.csv", index=False)
    df_sex.sort_values("FPR", ascending=False).to_csv(out_dir / "artifacts" / "sex_sorted.csv", index=False)

    print("OK: results generated in", out_dir.resolve())


if __name__ == "__main__":
    main()
