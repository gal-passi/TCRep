import numpy as np
import pandas as pd
import pickle
from dataset_handlers.Curation import filter_df
from dataset_handlers.processing_and_filtering import apply_extra_filter


def build_ms_dataset(self, df, df_h, dataset_type, top_percent=None, top_n_seqs=None, extra_filter=False):
    if dataset_type == 'ms' or 'ms_tcrdb2' in dataset_type or dataset_type == 'ms_no_healthy_ms':
        df_bld, df_hlt = df, df_h
        # Special case for TCRDB2 datasets!
        if 'tcrdb2' in dataset_type:
            if 'plus_hlt_article' in dataset_type:
                # add healthy from article
                df_article = self.get_full_healthy_synapse_mal_id_dataframe(to_recalculate=False, get_all=True)
                df_article_hlt = df_article[df_article["condition"] == "Healthy"]
                if top_n_seqs is not None:
                    topk = int(top_n_seqs * 1000)
                    if len(df_article_hlt) >= topk:
                        df_top = df_article_hlt.nlargest(topk, 'cloneFraction')
                        threshold = df_top['cloneFraction'].min()
                        threshold_df = df_article_hlt[df_article_hlt['cloneFraction'] >= threshold]
                        df_article_hlt = threshold_df
                elif top_percent is not None and 100 > top_percent > 0:
                    threshold = df['cloneFraction'].quantile((100 - top_percent) / 100)
                    df = df[df['cloneFraction'] >= threshold]
                df_hlt = pd.concat([df_hlt, df_article_hlt], axis=0, ignore_index=True)
            if 'extra_ms' in dataset_type:
                # Add to Blood df
                df_extra_bld = self.get_ms_extra_bld_dataframe(top_percent=top_percent, top_n_seqs=top_n_seqs,
                                                               df_bld=df_bld)
                df_bld = pd.concat([df_bld, df_extra_bld], axis=0, ignore_index=True)
            if 'hlt_as_ms' in dataset_type:
                # take 1 patient from each healthy study_id
                chosen_patient_ids = []
                for study_id in df_hlt['study_id'].unique():
                    patient_ids = df_hlt[df_hlt['study_id'] == study_id]['patient_id'].unique()
                    if len(patient_ids) > 0:
                        chosen_patient_ids.append(np.random.choice(patient_ids))
                df_excess_hlt = df_hlt[df_hlt['patient_id'].isin(chosen_patient_ids)]
                df_excess_hlt['patient_id'] = df_excess_hlt['patient_id'].astype(str) + '_hlt_as_ms'
                # remove from hlt those patients that were picked
                df_hlt = df_hlt[~df_hlt['patient_id'].isin(chosen_patient_ids)]
            if 'hlt_article' in dataset_type:
                df_article = self.get_full_healthy_synapse_mal_id_dataframe(to_recalculate=False, get_all=True)
                df_hlt_tmp = df_article[df_article["condition"] == "Healthy"]
                if 'plus_hlt_article' in dataset_type:
                    df_hlt = pd.concat([df_hlt, df_hlt_tmp], axis=0, ignore_index=True)
                else:
                    df_hlt = df_hlt_tmp
                df_hlt = filter_df(df_hlt, top_percent, top_n_seqs)
            if extra_filter:
                df_bld, df_hlt = apply_extra_filter(df_bld, df_hlt, top_n_seqs=top_n_seqs)
            return df_bld, df_hlt if 'hlt_as_ms' not in dataset_type else df_bld, df_excess_hlt, df_hlt
    elif dataset_type == 'ms_hlt_article':
        df_bld, df_hlt = df, df_h
        # Only Healthy df from article
        df_article = self.get_full_healthy_synapse_mal_id_dataframe(to_recalculate=True, get_all=True)
        # df_article = self.get_full_healthy_synapse_mal_id_dataframe(to_recalculate=False, get_all=True)
        df_hlt = df_article[df_article["condition"] == "Healthy"]
    elif dataset_type == 'ms_plus_hlt_article':
        df_bld, df_hlt = df, df_h
        df_article = self.get_full_healthy_synapse_mal_id_dataframe(to_recalculate=False, get_all=True)
        df_article_hlt = df_article[df_article["condition"] == "Healthy"]
        df_hlt = pd.concat([df_hlt, df_article_hlt], axis=0, ignore_index=True)
    elif dataset_type == 'ms_extra':
        df_bld, df_hlt = df, df_h
        # Add to Blood df
        df_extra_bld = self.get_ms_extra_bld_dataframe(df_bld)
        df_bld = pd.concat([df_bld, df_extra_bld], axis=0, ignore_index=True)
    elif dataset_type == 'ms_extra_hlt_article':
        df_bld, df_hlt = df, df_h
        # Add to Blood df
        df_extra_bld = self.get_ms_extra_bld_dataframe(df_bld)
        df_bld = pd.concat([df_bld, df_extra_bld], axis=0, ignore_index=True)
        # Only Healthy df from article
        df_article = self.get_full_healthy_synapse_mal_id_dataframe(to_recalculate=False, get_all=True)
        df_hlt = df_article[df_article["condition"] == "Healthy"]
    elif dataset_type == 'ms_extra_plus_hlt_article':
        df_bld, df_hlt = df, df_h
        # Add to Blood df
        df_extra_bld = self.get_ms_extra_bld_dataframe(df_bld)
        df_bld = pd.concat([df_bld, df_extra_bld], axis=0, ignore_index=True)
        # Add to Healthy df
        df_article = self.get_full_healthy_synapse_mal_id_dataframe(to_recalculate=False, get_all=True)
        df_article_hlt = df_article[df_article["condition"] == "Healthy"]
        df_hlt = pd.concat([df_hlt, df_article_hlt], axis=0, ignore_index=True)
    else:
        return None

    if extra_filter:
        df_bld, df_hlt = apply_extra_filter(df_bld, df_hlt, top_n_seqs=top_n_seqs)

    return df_bld, df_hlt


def build_sle_dataset(self, saved_df, saved_dataset_path, dataset_type, top_percent=None, top_n_seqs=None, extra_filter=False):
    if saved_df is not None:
        df_bld, df_hlt = saved_df
        return df_bld, df_hlt

    df_article = self.get_full_healthy_synapse_mal_id_dataframe(to_recalculate=False, get_all=True)
    if dataset_type == 'article_sle':
        df_bld = df_article[df_article["condition"] == "Lupus"]  # condition options: ['HIV' 'Healthy' 'T1D' 'Lupus' 'Covid19']
        df_hlt = df_article[df_article["condition"] == "Healthy"]
        df_bld = filter_df(df_bld, top_percent, top_n_seqs)
        df_hlt = filter_df(df_hlt, top_percent, top_n_seqs)
    elif dataset_type == 'article_sle_hlt_ms_no_healthy_ms':
        df_bld = df_article[df_article["condition"] == "Lupus"]  # condition options: ['HIV' 'Healthy' 'T1D' 'Lupus' 'Covid19']
        temp_extra_type = f'_top_{top_n_seqs}k' if top_n_seqs is not None else ''
        df_hlt = self.get_all_usable_healthy_data(dataset_type=f'ms_tcrdb2_no_healthy_ms' + temp_extra_type)
        df_bld = filter_df(df_bld, top_percent, top_n_seqs)
        df_hlt = filter_df(df_hlt, top_percent, top_n_seqs)
    elif dataset_type == 'article_sle_plus_hlt_ms_no_healthy_ms':
        df_bld = df_article[df_article["condition"] == "Lupus"]  # condition options: ['HIV' 'Healthy' 'T1D' 'Lupus' 'Covid19']
        df_hlt = df_article[df_article["condition"] == "Healthy"]
        temp_extra_type = f'_top_{top_n_seqs}k' if top_n_seqs is not None else ''
        df_hlt_ms = self.get_all_usable_healthy_data(dataset_type=f'ms_tcrdb2_no_healthy_ms' + temp_extra_type)
        df_hlt = pd.concat([df_hlt, df_hlt_ms], axis=0, ignore_index=True)
        df_bld = filter_df(df_bld, top_percent, top_n_seqs)
        df_hlt = filter_df(df_hlt, top_percent, top_n_seqs)
    else:
        return None

    if extra_filter:
        df_bld, df_hlt = apply_extra_filter(df_bld, df_hlt, top_n_seqs=top_n_seqs)
    with open(saved_dataset_path, 'wb') as f:
        pickle.dump((df_bld, df_hlt), f)

    return df_bld, df_hlt


def build_t1d_dataset(self, saved_df, saved_dataset_path, dataset_type, top_percent=None, top_n_seqs=None, extra_filter=False):
    if saved_df is not None:
        df_bld, df_hlt = saved_df
        return df_bld, df_hlt

    df_article = self.get_full_healthy_synapse_mal_id_dataframe(to_recalculate=False, get_all=True)
    if dataset_type == 't1d':
        df_bld = df_article[df_article["condition"] == "T1D"]  # condition options: ['HIV' 'Healthy' 'T1D' 'Lupus' 'Covid19']
        df_hlt = df_article[df_article["condition"] == "Healthy"]
        df_bld = filter_df(df_bld, top_percent, top_n_seqs)
        df_hlt = filter_df(df_hlt, top_percent, top_n_seqs)
    elif dataset_type == 't1d_hlt_ms_no_healthy_ms':
        df_bld = df_article[df_article["condition"] == "T1D"]  # condition options: ['HIV' 'Healthy' 'T1D' 'Lupus' 'Covid19']
        temp_extra_type = f'_top_{top_n_seqs}k' if top_n_seqs is not None else ''
        df_hlt = self.get_all_usable_healthy_data(dataset_type=f'ms_tcrdb2_no_healthy_ms' + temp_extra_type)
        df_bld = filter_df(df_bld, top_percent, top_n_seqs)
        df_hlt = filter_df(df_hlt, top_percent, top_n_seqs)
    elif dataset_type == 't1d_plus_hlt_ms_no_healthy_ms':
        df_bld = df_article[df_article["condition"] == "T1D"]  # condition options: ['HIV' 'Healthy' 'T1D' 'Lupus' 'Covid19']
        df_hlt = df_article[df_article["condition"] == "Healthy"]
        temp_extra_type = f'_top_{top_n_seqs}k' if top_n_seqs is not None else ''
        df_hlt_ms = self.get_all_usable_healthy_data(dataset_type=f'ms_tcrdb2_no_healthy_ms' + temp_extra_type)
        df_hlt = pd.concat([df_hlt, df_hlt_ms], axis=0, ignore_index=True)
        df_bld = filter_df(df_bld, top_percent, top_n_seqs)
        df_hlt = filter_df(df_hlt, top_percent, top_n_seqs)
    else:
        return None

    if extra_filter:
        df_bld, df_hlt = apply_extra_filter(df_bld, df_hlt, top_n_seqs=top_n_seqs)
    with open(saved_dataset_path, 'wb') as f:
        pickle.dump((df_bld, df_hlt), f)

    return df_bld, df_hlt


def build_other_dataset(self, saved_df, saved_dataset_path, dataset_type, top_percent=None, top_n_seqs=None, extra_filter=False):
    if dataset_type == 'cmv':
        df_cmv = self.get_all_usable_disease_data(disease='CMV', get_all=True)
        df_bld = df_cmv[df_cmv["condition"] == "CMV"]
        filtered_patient_ids = [x[0] for x in df_bld.groupby("patient_id")["AASeq"] if len(x[1]) >= 2000]  # this leaves 25 patients
        df_bld = df_bld[df_bld["patient_id"].isin(filtered_patient_ids)]
        df_hlt = df_cmv[df_cmv["condition"] == "Healthy"]
        return df_bld, df_hlt

    if saved_df is not None:
        df_bld, df_hlt = saved_df
        return df_bld, df_hlt
    if dataset_type == 'article_hiv':
        df_article = self.get_full_healthy_synapse_mal_id_dataframe(to_recalculate=False, get_all=True)
        df_bld = df_article[df_article["condition"] == "HIV"]  # condition options: ['HIV' 'Healthy' 'T1D' 'Lupus' 'Covid19']
        df_hlt = df_article[df_article["condition"] == "Healthy"]
        df_bld = filter_df(df_bld, top_percent, top_n_seqs)
        df_hlt = filter_df(df_hlt, top_percent, top_n_seqs)
    elif dataset_type == 'article_covid19':
        df_article = self.get_full_healthy_synapse_mal_id_dataframe(to_recalculate=False, get_all=True)
        df_bld = df_article[df_article["condition"] == "Covid19"]  # condition options: ['HIV' 'Healthy' 'T1D' 'Lupus' 'Covid19']
        df_hlt = df_article[df_article["condition"] == "Healthy"]
        df_bld = filter_df(df_bld, top_percent, top_n_seqs)
        df_hlt = filter_df(df_hlt, top_percent, top_n_seqs)
    elif dataset_type == 'article_influenza':
        df_article = self.get_full_healthy_synapse_mal_id_dataframe(to_recalculate=False, get_all=True)
        df_bld = df_article[df_article["condition"] == "Influenza"]  # condition options: ['HIV' 'Healthy' 'T1D' 'Lupus' 'Covid19']
        df_hlt = df_article[df_article["condition"] == "Healthy"]
        df_bld = filter_df(df_bld, top_percent, top_n_seqs)
        df_hlt = filter_df(df_hlt, top_percent, top_n_seqs)
    elif dataset_type == 'jia_tcrdb2':
        raise NotImplementedError("JIA TCRDB2 dataset is not implemented yet!")
    else:
        return None

    if extra_filter:
        df_bld, df_hlt = apply_extra_filter(df_bld, df_hlt, top_n_seqs=top_n_seqs)
    with open(saved_dataset_path, 'wb') as f:
        pickle.dump((df_bld, df_hlt), f)

    return df_bld, df_hlt
