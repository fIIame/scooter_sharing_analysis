from typing import List, Optional
from pathlib import PosixPath

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def plot_hist_boxplot(
    data: pd.DataFrame,
    columns: List[str],
    ncols: int = 2,
    hue: Optional[str] = None,
    kde: bool = False,
    save_path: Optional[PosixPath] = None,
) -> None:
    """
    Строит гистограмму и boxplot (ящик с усами) для указанных числовых признаков.

    Параметры
    ----------
    data : pd.DataFrame
        Исходный датафрейм с данными.
    columns : List[str]
        Список числовых признаков, для которых нужно построить графики.
    hue : Optional[str], default=None
        Переменная для раскраски данных (например, категория).
    kde : bool, default=False
        Отображать ли линию плотности (KDE) на гистограмме.
    ncols : int, default=2
        Количество столбцов подграфиков (по одному для hist и boxplot).
    save_path : Optional[str], default=None
        Путь для сохранения изображения. Если None — показать на экране.

    Возвращает
    ----------
    None
    """

    plot_rows = len(columns)
    fig, axes = plt.subplots(
        nrows=plot_rows, ncols=ncols, figsize=(14, 5 * plot_rows), squeeze=False
    )

    plt.rcParams.update({"font.size": 18})

    for idx, col in enumerate(columns):
        if hue is not None:
            sns.histplot(data=data, x=col, ax=axes[idx, 0], kde=kde, hue=hue)
            sns.boxplot(data=data, x=col, ax=axes[idx, 1], hue=hue)
        else:
            sns.histplot(data=data, x=col, ax=axes[idx, 0], kde=kde)
            sns.boxplot(data=data, x=col, ax=axes[idx, 1])

        axes[idx, 0].set_xlabel(col)
        axes[idx, 1].set_xlabel(col)
        axes[idx, 0].set_ylabel("Количество")

    plt.suptitle(
        "Гистограмма и ящик с усами количественных признаков", fontsize=22, y=1.01
    )
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)


def plot_categorial_pie(
    data: pd.DataFrame,
    columns: List[str],
    ncols: int = 2,
    save_path: Optional[PosixPath] = None,
) -> None:
    """
    Строит круговые диаграммы (pie chart) для категориальных признаков.

    Параметры
    ----------
    data : pd.DataFrame
        Исходный датафрейм с данными.
    columns : List[str]
        Список категориальных признаков для построения диаграмм.
    ncols : int, default=2
        Количество столбцов подграфиков.
    save_path : Optional[str], default=None
        Путь для сохранения изображения. Если None — показать на экране.

    Возвращает
    ----------
    None
    """

    plt.rcParams.update(
        {
            "axes.labelsize": 12,
            "figure.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    plot_rows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(
        nrows=plot_rows, ncols=ncols, figsize=(6 * ncols, 6 * plot_rows), squeeze=False
    )

    for idx, column in enumerate(columns):
        i, j = divmod(idx, ncols)
        counts = data[column].value_counts()
        colors = sns.color_palette("pastel", n_colors=len(counts))
        axes[i, j].pie(
            counts.values, labels=counts.index, colors=colors, autopct="%1.1f%%"
        )
        axes[i, j].set_title(column)

    for ax in axes.flat[len(columns) :]:
        ax.remove()

    plt.suptitle("Круговые диаграммы категориальных признаков", fontsize=22, y=1.01)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)


def plot_scatterplot(
    data: pd.DataFrame,
    x: str,
    ys: List[str],
    hue: Optional[str] = None,
    ncols: int = 2,
    save_path: Optional[PosixPath] = None,
) -> None:
    """
    Строит scatterplot (диаграммы рассеяния) для нескольких признаков относительно одного x.

    Параметры
    ----------
    data : pd.DataFrame
        Исходный датафрейм с данными.
    x : str
        Название признака, который отображается по оси X.
    ys : List[str]
        Список признаков, которые отображаются по оси Y.
    hue : Optional[str], default=None
        Переменная для раскраски точек по категориям.
    ncols : int, default=2
        Количество столбцов подграфиков.
    save_path : Optional[str], default=None
        Путь для сохранения изображения. Если None — показать на экране.

    Возвращает
    ----------
    None
    """

    plot_rows = int(np.ceil(len(ys) / ncols))
    fig, axes = plt.subplots(
        plot_rows, ncols=ncols, figsize=(7 * ncols, 6 * plot_rows), squeeze=False
    )

    for idx, y in enumerate(ys):
        i, j = divmod(idx, ncols)
        ax = axes[i, j]

        if hue is not None:
            sns.scatterplot(data=data, x=x, y=y, hue=hue, ax=ax)
        else:
            sns.scatterplot(data=data, x=x, y=y, ax=ax)

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"Зависимость между {y} и {x}")

    for ax in axes.flat[len(ys) :]:
        ax.set_visible(False)

    plt.suptitle(
        f"Диаграммы рассеяния относительно признака '{x}'", fontsize=22, y=1.01
    )
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)


def plot_topn_bar(
    data: pd.DataFrame,
    columns: List[str],
    n: int = 5,
    ncols: int = 2,
    save_path: Optional[PosixPath] = None,
) -> None:
    """
    Строит столбчатую диаграмму для топ-n наиболее частых значений Series.

    Параметры
    ----------
    data : pd.DataFrame
        Столбец с категориальными данными (например, start_location).
    columns : List[str]
        Список категориальных признаков для построения диаграмм.
    n : int, default=5
        Количество наиболее частых значений, которые будут отображены.
    ncols : int, default=2
        Количество столбцов подграфиков.
    save_path : str, optional
        Путь для сохранения изображения. Если None — график только отображается.

    Возвращает
    -------
    None
    """
    plot_rows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(
        nrows=plot_rows, ncols=ncols, figsize=(6 * ncols, 6 * plot_rows), squeeze=False
    )

    for idx, col in enumerate(columns):
        i, j = divmod(idx, ncols)

        top_n = data[col].value_counts().head(n)
        colors = sns.color_palette("pastel", n_colors=len(top_n))

        axes[i, j].bar(top_n.index, top_n.values, color=colors)
        axes[i, j].set_title(f"Топ {n}: {col}")
        axes[i, j].set_xlabel(col)
        axes[i, j].set_ylabel("Количество")
        axes[i, j].tick_params(axis="x", rotation=45)

    for ax in axes.flat[len(columns) :]:
        ax.remove()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: pd.Series,
    figsize: tuple = (16, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Строит график фактических и предсказанных значений спроса во времени.

    Используется для оценки того, насколько хорошо модель повторяет
    временной паттерн спроса.

    Параметры
    ----------
    y_true : pd.Series
        Фактические значения целевой переменной.
    y_pred : pd.Series
        Предсказанные значения модели.
    figsize : tuple, default=(16, 6)
        Размер графика.
    save_path : Optional[str], default=None
        Путь для сохранения изображения. Если None — график отображается на экране.

    Возвращает
    ----------
    None
    """
    plt.figure(figsize=figsize)
    plt.plot(y_true.index, y_true, label="Фактический спрос", color="blue")
    plt.plot(y_true.index, y_pred, label="Прогноз модели", color="orange", alpha=0.7)

    plt.xlabel("Время")
    plt.ylabel("Спрос (количество поездок)")
    plt.title("Фактический vs прогноз модели")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()


def plot_residuals_distribution(
    y_true: pd.Series,
    y_pred: pd.Series,
    figsize: tuple = (8, 4),
    kde: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Строит гистограмму распределения остатков модели.

    Используется для проверки предположений линейной регрессии:
    - нормальность остатков,
    - наличие асимметрии,
    - выбросы.

    Параметры
    ----------
    y_true : pd.Series
        Фактические значения целевой переменной.
    y_pred : pd.Series
        Предсказанные значения модели.
    figsize : tuple, default=(8, 4)
        Размер графика.
    kde : bool, default=True
        Отображать ли KDE-кривую плотности.
    save_path : Optional[str], default=None
        Путь для сохранения изображения. Если None — график отображается на экране.

    Возвращает
    ----------
    None
    """
    residuals = y_true - y_pred

    plt.figure(figsize=figsize)
    sns.histplot(residuals, kde=kde, color="purple")

    plt.xlabel("Остатки")
    plt.title("Распределение остатков модели")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()


def plot_deficit_heatmap(
        net_long: pd.DataFrame,
        daily_min: pd.DataFrame,
        top_n: int = 20,
        save_path: Optional[str] = None
) -> None:
    """
    Строит heatmap для топ-дефицитных точек по почасовому net_balance.

    Используется для анализа, в какие часы и в каких точках наблюдается дефицит самокатов.

    Параметры
    ----------
    net_long : pd.DataFrame
        DataFrame в длинном формате с колонками ['time', 'point', 'net_balance'].
    daily_min : pd.DataFrame
        DataFrame с расчетом оптимального количества самокатов.
    top_n : int, default=20
        Количество топ-дефицитных точек для отображения.
    save_path : Optional[str], default=None
        Путь для сохранения изображения. Если None — график отображается на экране.

    Возвращает
    ----------
    None
    """
    hourly_profile = (
        net_long
        .groupby([net_long['time'].dt.hour, 'point'])['net_balance']
        .mean()
        .unstack()
    )

    mean_deficit = daily_min.groupby("point")["optimal_count"].mean()
    top_deficit = mean_deficit.sort_values(ascending=False).head(top_n).index

    plt.figure(figsize=(14, 8))
    sns.heatmap(hourly_profile[top_deficit].T, cmap="RdYlGn_r", center=0, linewidths=0.5)
    plt.title(f"Топ {top_n} дефицитных точек по самокатам")
    plt.xlabel("Час дня")
    plt.ylabel("Точки")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_surplus_heatmap(
        net_long: pd.DataFrame,
        daily_min: pd.DataFrame,
        top_n: int = 20,
        save_path: Optional[str] = None
) -> None:
    """
    Строит heatmap для топ-избыточных точек по почасовому net_balance.

    Используется для анализа, в какие часы и в каких точках наблюдается избыток самокатов.

    Параметры
    ----------
    net_long : pd.DataFrame
        DataFrame в длинном формате с колонками ['time', 'point', 'net_balance'].
    daily_min : pd.DataFrame
        DataFrame с расчетом оптимального количества самокатов.
    top_n : int, default=20
        Количество топ-избыточных точек для отображения.
    save_path : Optional[str], default=None
        Путь для сохранения изображения. Если None — график отображается на экране.

    Возвращает
    ----------
    None
    """
    hourly_profile = (
        net_long
        .groupby([net_long['time'].dt.hour, 'point'])['net_balance']
        .mean()
        .unstack()
    )

    mean_deficit = daily_min.groupby("point")["optimal_count"].mean()
    top_surplus = mean_deficit.sort_values(ascending=True).head(top_n).index

    plt.figure(figsize=(14, 8))
    sns.heatmap(hourly_profile[top_surplus].T, cmap="RdYlGn_r", center=0, linewidths=0.5)
    plt.title(f"Топ {top_n} избыточных точек по самокатам")
    plt.xlabel("Час дня")
    plt.ylabel("Точки")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()
