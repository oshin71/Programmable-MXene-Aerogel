import numpy as np
import pandas as pd


def augmente_data(raw_df, x_cols, y_cols, n_augmented=0,
                  y_std_cols=None, x_std=0, y_std=0,
                  id_col='sID', da_col='DA#', random_state=None):
    if y_std_cols is not None:
        y_std_df = raw_df[std_cols]
    else:
        y_std_df = np.ones(raw_df[y_cols].values.shape) * y_std

    x_std_df = np.ones(raw_df[x_cols].values.shape) * x_std

    raw_df = raw_df.copy()
    raw_df[da_col] = 0

    noisy_dfs = [raw_df, ]
    rng = np.random.default_rng(random_state)

    for i in range(n_augmented):
        noisy_x = pd.DataFrame(rng.normal(raw_df[x_cols], x_std_df), columns=x_cols)
        noisy_y = pd.DataFrame(rng.normal(raw_df[y_cols], y_std_df), columns=y_cols)
        noisy_xy = noisy_x.join(noisy_y)
        noisy_sid = raw_df[[id_col]].copy()
        noisy_sid[da_col] = i + 1
        noisy_df = noisy_sid.join(noisy_xy).reset_index(drop=True)
        noisy_dfs.append(noisy_df)
    final_df = pd.concat(noisy_dfs, ignore_index=True).sort_values(by=[id_col, da_col]).reset_index(drop=True)
    return final_df[[id_col, da_col] + x_cols + y_cols]