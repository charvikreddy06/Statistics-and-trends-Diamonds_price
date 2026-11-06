"""
Statistics and trends: how carat, cut and price relate in ~54,000 round
diamonds (Kaggle "Diamonds" dataset).

Produces three figures for the report:
    categorical_plot.png  - histogram of price
    relational_plot.png   - scatter of price against carat (log-log)
    statistical_plot.png  - box plot of price by cut grade

Usage:
    python statistics_and_trends.py [path/to/diamonds.csv]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss

BLUE, ORANGE = '#2a78d6', '#eb6834'
CUT_RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']   # ordinal ramp
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
CUT_ORDER = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']


def set_style():
    """Shared, quiet plotting style sized for a two-page A4 report."""
    plt.rcParams.update({
        'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True,
        'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.dpi': 300, 'figure.dpi': 120,
    })


def dollars(ax, axis='x'):
    fmt = plt.FuncFormatter(lambda v, _: f'${v:,.0f}')
    (ax.xaxis if axis == 'x' else ax.yaxis).set_major_formatter(fmt)


def plot_categorical_plot(df):
    """Histogram of price in $250 bins, with mean, median and the price gap."""
    fig, ax = plt.subplots(figsize=(6.85, 2.4))
    ax.hist(df['price'], bins=np.arange(250, 19001, 250), color=BLUE,
            edgecolor='white', linewidth=0.5)
    mean, med = df['price'].mean(), df['price'].median()
    ax.axvline(med, color=INK, linestyle='--', linewidth=1)
    ax.axvline(mean, color=INK, linestyle=':', linewidth=1.2)
    ax.text(med + 150, 6000, f'median ${med:,.0f}', fontsize=7, va='top')
    ax.text(mean + 150, 5300, f'mean ${mean:,.0f}', fontsize=7, va='top')
    inset = ax.inset_axes([0.62, 0.36, 0.36, 0.5])
    inset.hist(df['price'], bins=np.arange(1300, 1701, 10), color=BLUE,
               edgecolor='white', linewidth=0.3)
    inset.set_xlim(1300, 1700)
    inset.set_xticks([1300, 1455, 1545, 1700])
    inset.tick_params(labelsize=6.3)
    dollars(inset, 'x')
    inset.set_title('Zoom, \\$10 bins: no prices from \\$1,455 to \\$1,545',
                    fontsize=6.8, loc='left')
    ax.set_xlim(0, 19000)
    ax.set_ylim(0, 6400)
    dollars(ax, 'x')
    ax.set_xlabel('Price (US\\$, \\$250 bins)')
    ax.set_ylabel('Number of diamonds')
    ax.set_title(f'Distribution of price (n={len(df):,})', loc='left')
    fig.tight_layout()
    fig.savefig('categorical_plot.png')
    plt.close(fig)


def plot_relational_plot(df):
    """Log-log scatter of price against carat with a power-law fit."""
    fig, ax = plt.subplots(figsize=(6.85, 2.65))
    ax.scatter(df['carat'], df['price'], s=1.5, c=BLUE, alpha=0.12,
               linewidths=0, rasterized=True, label='One diamond')
    b, a = np.polyfit(np.log10(df['carat']), np.log10(df['price']), 1)
    xs = np.logspace(np.log10(0.2), np.log10(5.0), 50)
    ax.plot(xs, 10 ** a * xs ** b, color=ORANGE, linewidth=1.6,
            label=f'Least-squares fit: price ~ carat$^{{{b:.2f}}}$')
    for c in (1.0, 1.5, 2.0):
        ax.axvline(c, color=INK2, linewidth=0.7, linestyle=':')
    ax.text(1.03, 380, '1.0, 1.5 and 2.0 ct', fontsize=6.8, color=INK2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([0.2, 0.3, 0.5, 1, 2, 3, 5])
    ax.set_xticklabels(['0.2', '0.3', '0.5', '1', '2', '3', '5'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks([300, 1000, 3000, 10000])
    dollars(ax, 'y')
    ax.set_xlabel('Carat (log scale)')
    ax.set_ylabel('Price (log scale)')
    ax.set_title('Price against carat weight', loc='left')
    leg = ax.legend(frameon=False, loc='upper left', markerscale=6)
    for handle in leg.legend_handles[:1]:
        handle.set_alpha(0.8)
    fig.tight_layout()
    fig.savefig('relational_plot.png')
    plt.close(fig)


def plot_statistical_plot(df):
    """Box plot of price by cut, annotated with each grade's median carat."""
    fig, ax = plt.subplots(figsize=(6.85, 2.55))
    data = [df.loc[df['cut'] == c, 'price'] for c in CUT_ORDER]
    box = ax.boxplot(data, patch_artist=True, widths=0.45, showfliers=True,
                     medianprops=dict(color='white', linewidth=1.6),
                     whiskerprops=dict(color=INK2, linewidth=0.9),
                     capprops=dict(color=INK2, linewidth=0.9),
                     flierprops=dict(marker='o', markersize=1.5, alpha=0.25,
                                     markerfacecolor=INK2, markeredgewidth=0))
    for patch, colour in zip(box['boxes'], CUT_RAMP):
        patch.set_facecolor(colour)
        patch.set_edgecolor(colour)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([
        f"{c}\nmedian {df.loc[df['cut'] == c, 'carat'].median():.2f} ct"
        for c in CUT_ORDER])
    for i, c in enumerate(CUT_ORDER, start=1):
        med = df.loc[df['cut'] == c, 'price'].median()
        ax.text(i + 0.26, med, f'${med:,.0f}', fontsize=6.8, va='center',
                color=INK)
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yscale('log')
    ax.set_yticks([300, 1000, 3000, 10000])
    dollars(ax, 'y')
    ax.set_xlabel('')
    ax.set_ylabel('Price (log scale)')
    ax.set_title('Price by cut grade (labels: median price; below: median carat)',
                 loc='left')
    fig.tight_layout()
    fig.savefig('statistical_plot.png')
    plt.close(fig)


def statistical_analysis(df, col):
    """Return the first four moments of a column (excess kurtosis)."""
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Remove exact duplicates and stones with a zero length/width/depth."""
    before = len(df)
    df = df.drop_duplicates()
    df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)].reset_index(drop=True)
    print(f'rows removed: {before - len(df)} (duplicates and zero dimensions)')
    print(df.describe().round(2))
    print(df[['carat', 'depth', 'table', 'price']].corr().round(3))
    return df


def writing(moments, col):
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    skew = 'right' if moments[2] > 0.5 else 'left' if moments[2] < -0.5 else 'not'
    kurt = ('leptokurtic' if moments[3] > 1 else
            'platykurtic' if moments[3] < -1 else 'mesokurtic')
    print(f'The data was {skew} skewed and {kurt}.')


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'diamonds.csv'
    set_style()
    df = preprocessing(pd.read_csv(path))
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    writing(statistical_analysis(df, 'price'), 'price')
    writing(statistical_analysis(df, 'carat'), 'carat')
    r_log = ss.pearsonr(np.log10(df['carat']), np.log10(df['price']))[0]
    print(f"Pearson r raw = {ss.pearsonr(df['carat'], df['price'])[0]:.3f}, "
          f"log-log = {r_log:.3f}")
    band = df[df['carat'].between(0.9, 1.1)]
    print('median price, 0.9-1.1 ct:')
    print(band.groupby('cut')['price'].median().reindex(CUT_ORDER))


if __name__ == '__main__':
    main()
