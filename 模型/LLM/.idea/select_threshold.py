import numpy as np
import pandas as pd
from evaluate import calc_metrics


def search_threshold(y_true, y_prob, start=0.05, end=0.95, step=0.01):
    rows = []
    ths = np.arange(start, end + 1e-12, step)
    for th in ths:
        m = calc_metrics(y_true, y_prob, float(th))
        rows.append(m)
    df = pd.DataFrame(rows)
    df = df.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).reset_index(drop=True)
    best = float(df.iloc[0]["threshold"])
    return best, df