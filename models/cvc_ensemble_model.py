import sys
# sys.path.append('other_models/CVC')
import numpy as np
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from models.cvc_model import CVCClassifierModel
from cache_handler import load_model_state


class CVCEnsembleModel(nn.Module):
    def __init__(self, args, device, models_weights=None):
        super(CVCEnsembleModel, self).__init__()

        self.models = []
        args.to_ensemble = False
        for i in range(1, 6):
            args.negative_partition = i
            model = CVCClassifierModel(batch_size=args.batch_size, ch_dropout=args.classification_dropout,
                                       cvc_layers_to_train=args.cvc_layers_to_train, freeze_embed_model=args.freeze_embed_model,
                                       lora=args.lora, ch_type=args.ch_type, device=device)
            trained_model = load_model_state(model, args, args.epochs - 1, device)
            if trained_model is not None:
                self.models.append(trained_model)
            else:
                raise ValueError(f"Model for negative partition {i} could not be loaded. Check the model path or training process.")
        args.to_ensemble = True
        self.weights = models_weights if models_weights is not None else [1.0] * len(self.models)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        outputs = []
        for model in self.models:
            output = model(x).cpu()
            outputs.append(self.softmax(output)[:, 1])

        weighted_sum = sum(w * out for w, out in zip(self.weights, outputs))
        return weighted_sum

