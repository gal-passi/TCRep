import torch
import torch.nn as nn
from models.cvc_model import CVCClassifierModel
from utils.cache_handler import load_model_state


class CVCCombinedReshefModel(nn.Module):
    def __init__(self, reshef_negative_part, args, device):
        super(CVCCombinedReshefModel, self).__init__()

        self.models = []
        self.device = device
        self.cvc_layers_to_train = args.cvc_layers_to_train
        self.num_models = len(self.models)

        old_model_type, args.model_type = args.model_type, "cvc"
        for epochs in range(0, args.epochs):
            model = CVCClassifierModel(batch_size=args.batch_size, ch_dropout=args.classification_dropout, cvc_layers_to_train=args.cvc_layers_to_train,
                                       freeze_embed_model=args.freeze_embed_model, lora=args.lora, ch_type=args.ch_type, device=device)
            trained_model = load_model_state(model, args, epochs, device)
            # trained_model.model.method = 'attention_layers'
            trained_model.eval()
            trained_model.to(device)
            self.models.append(trained_model)
        args.model_type = old_model_type

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass that runs frozen layers once and trainable parts for all models

        Returns:
            List of outputs from all 20 models
        """

        outputs = []
        with torch.no_grad():
            for model in self.models:
                outputs.append(model(input_ids))

        # Combine outputs from all models (by avg)
        outputs = torch.stack(outputs, dim=0)
        outputs = torch.mean(outputs, dim=0)
        return outputs
