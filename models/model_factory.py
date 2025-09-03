from models.ff_model import FeedForwardClassifier
from models.cvc_model import CVCClassifierModel
from models.cvc_combine_reshef_inference import CVCCombinedReshefModel
from models.esmc_ff_model import ESMCFeedForwardClassifier
from models.cvc_full_model import CVCClassifierModelFullEmbed


def build_model(model_type, positive_seqs=None, batch_size=None, ch_dropout=0.2,
                cvc_layers_to_train=3, freeze_embed_model=False, lora=False,
                ch_type="none", device="cpu", reshef_negative_part=0, args=None):
    """
    Factory function to build a model given the model_type and its parameters.
    """

    if model_type == "ff":
        if positive_seqs is None:
            raise ValueError("positive_seqs is required for FeedForward model")
        max_seq_len = max(len(seq) for seq in positive_seqs)
        return FeedForwardClassifier(max_seq_len).to(device)
    elif model_type == "cvc":
        return CVCClassifierModel(
            batch_size=batch_size,
            ch_dropout=ch_dropout,
            cvc_layers_to_train=cvc_layers_to_train,
            freeze_embed_model=freeze_embed_model,
            lora=lora,
            ch_type=ch_type,
            device=device,
        )
    elif model_type == "cvc_combined_reshef":
        return CVCCombinedReshefModel(reshef_negative_part, args, device=device)
    elif model_type == "esmc":
        return ESMCFeedForwardClassifier(device=device)
    elif model_type in ["cvc_full", "cvc_weighted"]:
        method = "weighted" if model_type == "cvc_weighted" else "full"
        return CVCClassifierModelFullEmbed(
            batch_size=batch_size,
            method=method,
            ch_dropout=ch_dropout,
            cvc_layers_to_train=cvc_layers_to_train,
            freeze_embed_model=freeze_embed_model,
            lora=lora,
            ch_type=ch_type,
            device=device,
        )
    else:
        raise ValueError(f"Model type {model_type} is not supported")
