import pandas as pd

def confusion_rates(y_true: pd.Series, y_pred: pd.Series) -> dict:
    # y_true e y_pred devem ser 0/1
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    # evitar divisão por zero
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "FPR": fpr, "FNR": fnr, "TPR": tpr, "TNR": tnr
    }

def fairness_by_group(df: pd.DataFrame, group_col: str, y_true_col: str, y_pred_col: str) -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby(group_col):
        rates = confusion_rates(sub[y_true_col], sub[y_pred_col])
        rates[group_col] = g
        rates["N"] = int(len(sub))
        rows.append(rates)
    out = pd.DataFrame(rows).sort_values(by="N", ascending=False)
    # gaps úteis
    if not out.empty:
        out["FPR_gap_vs_min"] = out["FPR"] - out["FPR"].min()
        out["FNR_gap_vs_min"] = out["FNR"] - out["FNR"].min()
    return out
