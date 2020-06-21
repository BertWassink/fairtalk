import pandas as pd

def flatten_group_summary(group_summary_dict):
    dict_flattened = {k: v for (k, v) in group_summary_dict.get('by_group').items()}
    dict_flattened.update({'overall':  group_summary_dict.get('overall')})
    return dict_flattened

def create_df_summaries(summary_function, tp, fp, fn, tn):
    df = pd.DataFrame.from_dict({
        'True positive': summary_function(tp),
        'False positive': summary_function(fp),
        'False negative': summary_function(fn),
        'True negative': summary_function(tn),
    }, orient='index')
    return df