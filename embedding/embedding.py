import os
import pickle
import numpy as np
import torch
import math

from torch.nn.functional import embedding
from tqdm import tqdm
from embedding.esmc_finetuning import load_fine_tuned_esmc
import sys
sys.path.append('other_models/CVC')

def embed_cvc(seqs, to_mean=True):
    raise NotImplementedError("This function is not implemented yet")
    from cvc.embbeding_wrapper import EmbeddingWrapper  # , GenericModelEmbeddings
    from lab_notebooks.utils import TRANSFORMER

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from bin.my_train_cvc import main
    model = main()
    model = EmbeddingWrapper(model=TRANSFORMER, device=device, sequences_df=None)

    exit(0)


class ESMCSingleTon:
    model = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            from esm.models.esmc import ESMC
            cls.model = ESMC.from_pretrained("esmc_300m")
        return cls.model


def embed_esmc(seqs, batch_size=512, to_mean=True, model_type='esmc', to_tqdm=True, use_pre_loaded=False):
    if use_pre_loaded:
        model = ESMCSingleTon.get_model()
    else:
        # load model according to model type
        if model_type == 'esmc':
            from esm.models.esmc import ESMC
            model = ESMC.from_pretrained("esmc_300m")
        else:
            model = load_fine_tuned_esmc("cache/esm_c_checkpoints")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    # Convert seqs to numpy array
    seqs_np = np.array(seqs)
    # List to accumulate final embeddings
    all_embeds = []
    # Process in batches
    iterator = range(0, len(seqs), batch_size)
    if to_tqdm:
        iterator = tqdm(iterator, total=math.ceil(len(seqs) / batch_size))
    for start_idx in iterator:
        end_idx = min(start_idx + batch_size, len(seqs))
        batch_seqs = seqs_np[start_idx:end_idx]
        with torch.no_grad():
            if model_type == 'esmc':
                # Tokenize the batch sequences
                tokenized = model._tokenize(batch_seqs)
                # Get embeddings for the batch
                embeds = model(tokenized).embeddings.to(torch.float32).cpu()
            else:
                # Tokenize the batch sequences
                tokenized = model.esmc_model._tokenize(batch_seqs)
                # Get embeddings for the batch
                embeds = model(sequences=tokenized).to(torch.float32).cpu()
        # Clean the start and end of each sequence in the embedding
        batch_embeds = [embed[1:len(seq) - 1] for embed, seq in zip(embeds, batch_seqs)]
        # Apply mean if needed
        if to_mean:
            batch_embeds = [embeds.mean(dim=0) for embeds in batch_embeds]
        else:
            batch_embeds = [embeds for embeds in batch_embeds]
        # Accumulate the batch embeddings
        all_embeds.extend(batch_embeds)
    return all_embeds


def get_cached_embeddings(sequences, study_name, name='', cache_dir="cache/esm_c/", embed_type='esmc'):
    """
    Load cached embeddings if available, otherwise compute and cache them.

    Args:
        sequences: List of sequences to embed
        cache_dir: Directory to store cached embeddings
        embed_fn: Function to compute embeddings if not cached

    Returns:
        numpy array of embeddings
    """
    cache_dir = os.path.join(cache_dir, study_name)
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"embeds_{name}.pkl")

    # Load with pickle
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    if embed_type == 'cvc':
        embeddings = embed_cvc(sequences)
    else:
        embeddings = embed_esmc(sequences, model_type=embed_type)

    # saving with pickle
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
    return embeddings
