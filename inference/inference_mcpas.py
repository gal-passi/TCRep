import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from cache_handler import get_model_config_str


BASE_MCPAS_CACHE = "db/ms_related/cache"


def inference_mcpas(trained_model, args, df_bld, df_hlt, test_patient_ids, valid_patient_ids, valid_pos_seqs,
                    valid_neg_seqs, test_pos_seqs, test_neg_seqs, aaseq_to_ratio, model_non_trained, device):
    os.makedirs(BASE_MCPAS_CACHE, exist_ok=True)

    trained_model.eval()
    model_non_trained.eval()

    model_config_string = get_model_config_str(args)

    # print('Saving McPass...')
    # aggregate_and_save_mcpas_results(args)
    # print('Done saving McPAS!')

    # read the .xlsx file "db/ms_related/IEDB_MS_AB.xlsx"
    df_iedb = pd.read_excel("db/ms_related/IEDB_MS_AB.xlsx")
    seqs_iedb_chain1 = df_iedb["Chain 1 CDR3"].values  # TODO: I think chain 1 is not necessary related to our sequences.
    seqs_iedb_chain2 = df_iedb["Chain 2 CDR3"].values

    # read the .csv file "db/ms_related/McPAS-TCR_MS_seqs.csv"
    df_mcpas_ms = pd.read_csv("db/ms_related/McPAS-TCR_MS_seqs.csv")
    seqs_mcpas_ms = df_mcpas_ms["CDR3.beta.aa"].unique()
    seqs_mcpas_ms = [seq for seq in seqs_mcpas_ms if 10 < len(seq) < 20]
    seqs_mcpas_ms = [seq for seq in seqs_mcpas_ms if seq.startswith('C') and seq.endswith('F')]
    with torch.no_grad():
        embeds = trained_model.get_embeddings(seqs_mcpas_ms)
        logits = trained_model.linear(embeds.to(torch.float32))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    print(f"McPAS MS sequences: Mean: {probs.mean():.4f}, Std: {probs.std():.4f}, Count: {len(seqs_mcpas_ms)}")
    print(f"McPAS MS sequences with prob > 0.5: {np.sum(probs > 0.5)}")

    # read the .csv file "db/ms_related/McPAS-TCR.csv"
    df_mcpas = pd.read_csv("db/ms_related/McPAS-TCR.csv", low_memory=False)
    df_mcpas = df_mcpas[df_mcpas['Species'] == 'Human']
    # remove nans from "CDR3.beta.aa" column
    df_mcpas = df_mcpas.dropna(subset=['CDR3.beta.aa'])

    # try to load from cache
    cache_path = os.path.join(BASE_MCPAS_CACHE, f"mcpas_inference_{model_config_string}.pt")
    if os.path.exists(cache_path):
        # load pathology_results
        pathology_results = torch.load(cache_path, map_location='cpu', weights_only=False)
    else:
        # for each unique Pathology, calculate model output and keep mean and std
        pathology_results = {}
        for pathology in tqdm(df_mcpas['Pathology'].unique()):
            # get all unique CDR3.beta.aa sequences for this pathology and apply filters
            df_pathology = df_mcpas[df_mcpas['Pathology'] == pathology]
            seqs_pathology = df_pathology["CDR3.beta.aa"].unique()
            seqs_pathology = [seq for seq in seqs_pathology if 10 < len(seq) < 20]
            seqs_pathology = [seq for seq in seqs_pathology if seq.startswith('C') and seq.endswith('F')]

            if len(seqs_pathology) < 25:
                continue

            # calculate model output
            if len(seqs_pathology) > 0:
                with torch.no_grad():
                    embeds = trained_model.get_embeddings(seqs_pathology)
                    logits = trained_model.linear(embeds.to(torch.float32))
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                # store mean and std
                pathology_results[pathology] = (probs.mean(), probs.std(), len(seqs_pathology), embeds.cpu(), probs, seqs_pathology)
        # save to cache
        torch.save(pathology_results, cache_path)

    # sort pathology_results by mean in descending order
    pathology_results = dict(sorted(pathology_results.items(), key=lambda item: item[1][0], reverse=True))
    for pathology, (mean, std, count, embeds, probs, seqs) in pathology_results.items():
        print(f"Pathology: {pathology}, Mean: {mean:.4f}, Std: {std:.4f}, Count: {count}")

    # try to load from cache
    cache_path = os.path.join(BASE_MCPAS_CACHE, f"tcrdb_inference_{model_config_string}.pt")
    if os.path.exists(cache_path):
        # load tcrdb_results
        tcrdb_results = torch.load(cache_path, map_location='cpu', weights_only=False)
    else:
        # now do the same for 2000 sequences from valid_pos_seqs, test_pos_seqs and valid_neg_seqs, test_neg_seqs
        tcrdb_results = {}
        for label, seqs in zip(['Valid Pos', 'Test Pos', 'Valid Neg', 'Test Neg'],
                               [valid_pos_seqs, test_pos_seqs, valid_neg_seqs, test_neg_seqs]):
            if len(seqs) > 2000:
                seqs = np.random.choice(seqs, 2000, replace=False)
                with torch.no_grad():
                    embeds = trained_model.get_embeddings(seqs)
                    logits = trained_model.linear(embeds.to(torch.float32))
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                tcrdb_results[label] = (probs.mean(), probs.std(), len(seqs), embeds.cpu(), probs, seqs)
            print(f"{label} Mean: {probs.mean():.4f}, Std: {probs.std():.4f}, Count: {len(seqs)}")
        # save to cache
        torch.save(tcrdb_results, cache_path)

    # create a pca with "Multiple sclerosis (MS)" and "Epstein Barr virus (EBV)" from pathology_results, and all tcrdb_results
    pca = PCA(n_components=2)
    scaler = StandardScaler()
    all_embeds = []
    labels = []
    for pathology, (mean, std, count, embeds, probs, seqs) in pathology_results.items():
        if pathology in ["Multiple sclerosis (MS)", "Epstein Barr virus (EBV)"]:
            all_embeds.append(embeds)
            labels.extend([pathology] * embeds.shape[0])
    for label, (mean, std, count, embeds, probs, seqs) in tcrdb_results.items():
        all_embeds.append(embeds)
        labels.extend([label] * embeds.shape[0])
    all_embeds = torch.cat(all_embeds, dim=0).numpy()
    all_embeds = scaler.fit_transform(all_embeds)
    pca_result = pca.fit_transform(all_embeds)
    df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    df_pca['Label'] = labels
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Label', alpha=0.5)
    plt.title('PCA of TCR Embeddings')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    return


def aggregate_and_save_mcpas_results(args):
    all_pathology_res = []
    for k in range(1, 9):
        args.k_fold = k
        model_config_str = get_model_config_str(args)
        cache_path = os.path.join(BASE_MCPAS_CACHE, f"mcpas_inference_{model_config_str}.pt")
        if os.path.exists(cache_path):
            # load pathology_results
            pathology_results = torch.load(cache_path, map_location='cpu', weights_only=False)
            all_pathology_res.append(pathology_results)
        else:
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

    rows = []
    for fold_idx, pathology_res in enumerate(all_pathology_res):
        for pathology, (mean, std, count, embeds, probs, seqs) in pathology_res.items():
            # Convert embeddings and probs to numpy if needed
            if hasattr(embeds, 'detach'):  # it's a torch tensor
                embeds_np = embeds.detach().cpu().numpy()
            else:
                embeds_np = embeds
            if hasattr(probs, 'detach'):
                probs_np = probs.detach().cpu().numpy()
            else:
                probs_np = probs

            # Ensure lengths match
            assert len(seqs) == embeds_np.shape[0] == len(probs_np), \
                f"Length mismatch for pathology {pathology} in fold {fold_idx}"

            # Add each sequence to rows
            for s, e, p in zip(seqs, embeds_np, probs_np):
                rows.append({
                    'sequence': s,
                    'model_fold': fold_idx,
                    'pathology': pathology,
                    'embedding': e,
                    'probabilities': p
                })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save to CSV (under BASE_MCPAS_CACHE)
    output_csv_path = os.path.join(BASE_MCPAS_CACHE, "mcpas_pathology_results_all_folds.csv")
    df.to_csv(output_csv_path, index=False)



""" Output (on fold 2):
McPAS MS sequences: Mean: 0.0569, Std: 0.1970, Count: 106
McPAS MS sequences with prob > 0.5: 7
Pathology: Parkinson disease, Mean: 0.1681, Std: 0.3111, Count: 90
Pathology: Yellow fever virus, Mean: 0.1647, Std: 0.3193, Count: 220
Pathology: Influenza, Mean: 0.1303, Std: 0.2864, Count: 3308
Pathology: Toxic epidermal necrolysis, Mean: 0.1298, Std: 0.2896, Count: 104
Pathology: M. tuberculosis, Mean: 0.1231, Std: 0.2813, Count: 1195
Pathology: Neoantigen, Mean: 0.1216, Std: 0.2749, Count: 909
Pathology: Tumor associated antigen (TAA), Mean: 0.1159, Std: 0.2724, Count: 132
Pathology: Epstein Barr virus (EBV), Mean: 0.1089, Std: 0.2686, Count: 1133
Pathology: COVID-19, Mean: 0.1067, Std: 0.2627, Count: 169
Pathology: M.Tuberculosis, Mean: 0.1060, Std: 0.2680, Count: 14739
Pathology: Calcified Aortic Stenosis disease, Mean: 0.1058, Std: 0.2851, Count: 25
Pathology: Cytomegalovirus (CMV), Mean: 0.1026, Std: 0.2568, Count: 2065
Pathology: Rheumatoid Arthritis (RA), Mean: 0.0964, Std: 0.2591, Count: 264
Pathology: Celiac disease, Mean: 0.0854, Std: 0.2379, Count: 208
Pathology: HTLV-1, Mean: 0.0811, Std: 0.2315, Count: 201
Pathology: Narcolepsy, Mean: 0.0809, Std: 0.2220, Count: 71
Pathology: Clear cell renal carcinoma, Mean: 0.0794, Std: 0.2352, Count: 65
Pathology: Diabetes Type 1, Mean: 0.0789, Std: 0.2261, Count: 874
Pathology: Allergy, Mean: 0.0775, Std: 0.2133, Count: 251
Pathology: Melanoma, Mean: 0.0716, Std: 0.2241, Count: 525
Pathology: Alzheimer's disease, Mean: 0.0584, Std: 0.2016, Count: 107
Pathology: Psoriatic arthritis, Mean: 0.0575, Std: 0.1991, Count: 157
Pathology: Multiple sclerosis (MS), Mean: 0.0569, Std: 0.1970, Count: 106
Pathology: Human immunodeficiency virus (HIV), Mean: 0.0555, Std: 0.1934, Count: 851
Pathology: Hepatitis C virus, Mean: 0.0410, Std: 0.1712, Count: 85
Pathology: Colorectal cancer, Mean: 0.0335, Std: 0.1605, Count: 37
Pathology: Herpes simplex virus 2 (HSV2), Mean: 0.0256, Std: 0.0786, Count: 30
Pathology: Merkel cell carcinoma, Mean: 0.0146, Std: 0.1091, Count: 440
"""
