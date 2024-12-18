from external_libs import logging, np, pd, warnings
from config import *



warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


pd.options.display.float_format = '{:.5f}'.format 

def describe_columns(data, columns, batch_size=1000):
    describe_results = []
    num_batches = (len(data) // batch_size) + 1
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_data = data.iloc[start_idx:end_idx]
        batch_description = batch_data[columns].describe(include='all')
        describe_results.append(batch_description)
    final_description = pd.concat(describe_results, axis=1).T.groupby(level=0).mean()
    return final_description.reset_index()

def basic_report(columns, data, batch_size=1000):
    logger.info(f"Missing values: {data.isnull().sum().to_dict()}")  # Check missing values
    logger.info(f"Duplicated data: {data.duplicated().sum()}")  # Check duplicated data
    logger.info(f"Dataframe structure: {data.shape}")  # DataFrame structure
    logger.info(f"Data types: {data.dtypes.value_counts().to_dict()}")  # Identify type of each data
    overall_stats = {
        col: {
            "sum": 0,
            "count": 0,
            "std_sum": 0,
            "values": [],
            "null_count": 0,
            "nan_count": 0
        } for col in columns
    }
    batches = np.array_split(data, max(1, len(data) // batch_size))
    for batch in batches:
        for column in columns:
            batch_count = batch[column].notnull().sum()
            batch_sum = batch[column].sum(skipna=True)
            batch_var = batch[column].var(skipna=True)
            batch_std_sum = batch_var * batch_count
            batch_values = batch[column].dropna().tolist()
            null_count = batch[column].isnull().sum()
            nan_count = batch[column].isna().sum()  # Corrigé ici
            overall_stats[column]["sum"] += batch_sum
            overall_stats[column]["count"] += batch_count
            overall_stats[column]["std_sum"] += batch_std_sum
            overall_stats[column]["values"].extend(batch_values)
            overall_stats[column]["null_count"] += null_count
            overall_stats[column]["nan_count"] += nan_count
    stats_data = []
    for column in columns:
        total_count = overall_stats[column]["count"]
        total_nulls = overall_stats[column]["null_count"]
        total_nans = overall_stats[column]["nan_count"]
        overall_mean = overall_stats[column]["sum"] / total_count if total_count else None
        overall_std = np.sqrt(overall_stats[column]["std_sum"] / total_count) if total_count else None
        all_values = pd.Series(overall_stats[column]["values"])
        global_min = all_values.min()
        global_max = all_values.max()
        global_median = all_values.median()
        global_mode = all_values.mode()[0] if not all_values.mode().empty else None
        global_q1 = all_values.quantile(0.25)
        global_q3 = all_values.quantile(0.75)
        global_iqr = global_q3 - global_q1
        global_vc = (overall_std / global_median) if global_median != 0 else float('nan')
        stats_data.append({
            "column": column,
            "mean": overall_mean,
            "median": global_median,
            "mode": global_mode,
            "std": overall_std,
            "min": global_min,
            "max": global_max,
            "q1": global_q1,
            "q3": global_q3,
            "iqr": global_iqr,
            "variance_coef": global_vc,
            "null_count": total_nulls,
            "nan_count": total_nans
        })
        logger.info(
            f"{column}: mean = {overall_mean}, median = {global_median}, mode = {global_mode}, "
            f"std = {overall_std}, min = {global_min}, max = {global_max}, q1 = {global_q1}, "
            f"q3 = {global_q3}, IQR = {global_iqr}, variance coef = {global_vc}, "
            f"null_count = {total_nulls}, nan_count = {total_nans}"
        )
    return pd.DataFrame(stats_data)


