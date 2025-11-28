from typing import List, Optional

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def plot_hist_boxplot(
        data: pd.DataFrame,
        cols: List[str],
        hue: Optional[str] = None,
        kde: bool = False
) -> None:

    plot_rows = len(cols)
    fig, axes = plt.subplots(nrows=plot_rows, ncols=2, figsize=(14, 5 * plot_rows))

    if plot_rows == 1:
        axes = axes.reshape(1, 2)

    plt.rcParams.update({"font.size": 18})

    for idx, col in enumerate(cols):
        sns.histplot(data=data, x=col, ax=axes[idx, 0], kde=kde, hue=hue)
        sns.boxplot(data=data, x=col, ax=axes[idx, 1], hue=hue)

        axes[idx, 0].set_xlabel(col)
        axes[idx, 1].set_xlabel(col)
        axes[idx, 0].set_ylabel("Количество")

    plt.suptitle("Гистограмма и ящик с усами количественных признаков", fontsize=22, y=1.01)
    fig.tight_layout()

    plt.show()
    plt.close(fig)
