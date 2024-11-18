from external_libs import logging, plt, np, pd, sns, os
from libs.colors import PersoColo as persocolo
from config import *



logger = logging.getLogger(__name__)


def batch_statistics(data, column, batch_size):
    min_values, max_values, medians, q1_values, q3_values = [], [], [], [], []
    for start in range(0, len(data), batch_size):
        batch = data[column].iloc[start:start + batch_size].dropna()
        if not batch.empty:
            min_values.append(batch.min())
            max_values.append(batch.max())
            medians.append(batch.median())
            q1_values.append(batch.quantile(0.25))
            q3_values.append(batch.quantile(0.75))
    global_min = min(min_values)
    global_max = max(max_values)
    global_median = np.median(medians)
    global_q1 = np.median(q1_values)
    global_q3 = np.median(q3_values)
    return global_min, global_q1, global_median, global_q3, global_max


def boxplot_generator(data, columns, batch_size=10000):
    for column in columns:
        global_min, global_q1, global_median, global_q3, global_max = batch_statistics(data, column, batch_size)
        fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')
        ax.boxplot([[global_min, global_q1, global_median, global_q3, global_max]], vert=False, patch_artist=True,
                   boxprops=dict(facecolor=persocolo.beige, color='black'),
                   whiskerprops=dict(color='blue'),
                   capprops=dict(color='green'),
                   medianprops=dict(color='red'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.title(f'boxplot for {column}', fontsize=14, color='black')
        plt.xlabel(column, fontsize=13, color='black')
        ax.text(global_min, 1.20, f'min: {global_min:.2f}', ha='right', fontsize=13)
        ax.text(global_median, 1.30, f'median: {global_median:.2f}', ha='center', fontsize=13)
        ax.text(global_max, 1.20, f'max: {global_max:.2f}', ha='left', fontsize=13)
        plt.savefig(os.path.join(GRAPHICS_PATH, f'boxplot_{column}.png'), format='png', dpi=900)
        plt.show()


def correlation_matrix(data, columns, batch_size=10000):
    n = len(columns)
    cov_matrix = np.zeros((n, n))
    count_matrix = np.zeros((n, n))
    for start in range(0, len(data), batch_size):
        batch = data[columns].iloc[start:start + batch_size].dropna()
        if not batch.empty:
            batch_corr = batch.corr().values
            cov_matrix += np.nan_to_num(batch_corr)
            count_matrix += ~np.isnan(batch_corr)
    avg_corr_matrix = cov_matrix / count_matrix
    avg_corr_df = pd.DataFrame(avg_corr_matrix, index=columns, columns=columns)
    plt.figure(figsize=(12, 12))
    sns.heatmap(avg_corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title("correlation matrix", fontsize=14)
    plt.savefig(os.path.join(GRAPHICS_PATH + 'correlation_matrix.png'), format='png', dpi=900)
    plt.show()
