import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import time
import wandb
from utils.cache_handler import save_model_state, load_model_state
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, ExponentialLR


# Costume loss with L2 regularization term
def custom_l2_loss(logits, labels, R=0.1):
    ce_loss = F.cross_entropy(logits, labels)
    probs = F.softmax(logits, dim=1)  # Apply softmax on logits to get probabilities
    l2_norm = torch.norm(probs, p=2, dim=1).mean()  # Compute L2 norm
    reg_term = (1 - l2_norm)  # Regularization term
    return ce_loss + R * reg_term  # Combined loss


# Costume loss with entropy regularization term
def entropy_loss(logits):
    probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()  # Compute entropy
    return entropy


def custom_loss_entropy(logits, labels, R=0.1, n_classes=2):
    ce_loss = F.cross_entropy(logits, labels)
    max_entropy = torch.log(torch.tensor(n_classes, dtype=torch.float32))
    reg_term = (1 - (max_entropy - entropy_loss(logits)) / max_entropy)
    return ce_loss + R * reg_term


class CustomLossCriterion(nn.Module):
    def __init__(self, device, loss_type='ce', class_weights=None, R=0.1, n_classes=2, ratio=False, aaseq_to_ratio=None, aaseq_to_dist=None, aaseq_to_nneighbors=None):
        """
        Flexible loss criterion that supports different loss types and class weights.

        Args:
            loss_type (str): Type of loss to use
            class_weights (torch.Tensor, optional): Weights for each class
            R (float): Regularization strength for custom losses
            n_classes (int): Number of classes for entropy-based regularization
        """
        super(CustomLossCriterion, self).__init__()

        self.loss_type = loss_type
        self.class_weights = class_weights
        self.R = R
        self.n_classes = n_classes
        self.ratio = ratio
        self.aaseq_to_ratio = aaseq_to_ratio
        self.aaseq_to_dist = aaseq_to_dist
        self.aaseq_to_nneighbors = aaseq_to_nneighbors
        self.device = device

    def forward(self, logits, labels, batch_samples):
        """
        Compute loss based on specified loss type.

        Args:
            logits (torch.Tensor): Model output logits
            labels (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Computed loss
        """
        # Apply class weights if provided
        if self.class_weights is not None:
            # Ensure class_weights is on the same device as labels
            self.class_weights.to(self.device)

        if self.ratio:
            batch_sample_ratios = self.aaseq_to_ratio(batch_samples).to(self.device)
            per_sample_losses = F.cross_entropy(logits, labels, weight=self.class_weights, reduction='none')
            sample_weights = torch.ones_like(per_sample_losses)
            positive_indices = labels == 1
            sample_weights[positive_indices] = batch_sample_ratios[positive_indices].type(torch.float32)
            weighted_losses = per_sample_losses * sample_weights
            ce_loss = weighted_losses.mean()
        elif self.aaseq_to_dist is not None:
            # apply aaseq_to_dist function to get the distance on the batch samples
            batch_sample_distances = self.aaseq_to_dist(batch_samples).to(self.device)
            per_sample_losses = F.cross_entropy(logits, labels, weight=self.class_weights, reduction='none')
            sample_weights = batch_sample_distances.type(torch.float32)
            negative_indices = labels == 0
            sample_weights[negative_indices] = batch_sample_distances[negative_indices].type(torch.float32)
            weighted_losses = per_sample_losses * sample_weights
            ce_loss = weighted_losses.mean()
        elif self.aaseq_to_nneighbors is not None:
            # apply aaseq_to_nneighbors function to get the number of neighbors on the batch samples
            batch_sample_nneighbors = self.aaseq_to_nneighbors(batch_samples).to(self.device)
            per_sample_losses = F.cross_entropy(logits, labels, weight=self.class_weights, reduction='none')
            sample_weights = batch_sample_nneighbors.type(torch.float32)  # Added this line to run on all samples
            # sample_weights = torch.ones_like(per_sample_losses)
            # positive_indices = labels == 1
            # sample_weights[positive_indices] = batch_sample_nneighbors[positive_indices].type(torch.float32)
            weighted_losses = per_sample_losses * sample_weights
            ce_loss = weighted_losses.mean()
        else:
            ce_loss = F.cross_entropy(logits, labels, weight=self.class_weights)

        if self.loss_type == 'ce':
            return ce_loss

        elif self.loss_type == 'ce_l2':
            # Custom L2 regularized loss
            probs = F.softmax(logits, dim=1)
            l2_norm = torch.norm(probs, p=2, dim=1).mean()
            reg_term = (1 - l2_norm)
            return ce_loss + self.R * reg_term

        elif self.loss_type == 'ce_entropy':
            # Custom entropy regularized loss
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            max_entropy = torch.log(torch.tensor(self.n_classes, dtype=torch.float32))
            reg_term = (1 - (max_entropy - entropy) / max_entropy)
            return ce_loss + self.R * reg_term

        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def get_scheduler(optimizer, scheduler_type, **kwargs):
    """Returns the selected scheduler based on the given string."""
    if scheduler_type == "StepLR".lower():
        return StepLR(optimizer, step_size=kwargs.get("step_size", 8), gamma=kwargs.get("gamma", 0.1))
    elif scheduler_type == "ReduceLROnPlateau".lower():
        return ReduceLROnPlateau(optimizer, mode="min", factor=kwargs.get("factor", 0.1),
                                 patience=kwargs.get("patience", 5), verbose=True)
    elif scheduler_type == "CosineAnnealingLR".lower():
        return CosineAnnealingLR(optimizer, T_max=kwargs.get("T_max", 50), eta_min=kwargs.get("eta_min", 1e-6))
    elif scheduler_type == "ExponentialLR".lower():
        return ExponentialLR(optimizer, gamma=kwargs.get("gamma", 0.95))
    elif scheduler_type == "none":
        return None
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def train_model(model, train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs,
                log_wandb, model_type, loss_type, freeze_embed_model, special_criterion,
                embedding_lr, reg_coef, pos_weights, aaseq_to_ratio, aaseq_to_dist, change_negatives, optimizer_type, args,
                aaseq_to_nneighbors=None, masking=False, ratio=False, scheduler_type='none',
                epochs=10, lr=0.0005, pos_batch_size=30, neg_pos_ratio=10, is_sweep=False,
                reshef_inference=False, reshef_filter_train=False, reshef_negative_part=0,
                test_pos_seqs=None, test_neg_seqs=None):  # pos_batch_size=256
    """
    Train a binary classification model with positive and negative sequences,
    while validating on a separate validation set during training.

    Args:
        model: The model that takes a list of strings and returns logits of shape (n, 2)
        train_pos_seqs: numpy array of positive sequences (strings)
        neg_seqs: numpy array of negative sequences (strings)
        valid_pos_seqs: numpy array of positive sequences for validation (strings)
        valid_neg_seqs: numpy array of negative sequences for validation (strings)
        epochs: Number of training epochs
        lr: Learning rate
        pos_batch_size: Number of positive samples per batch
        neg_pos_ratio: Ratio of negative to positive samples in each batch

    Returns:
        Trained model and training history
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Print the trainable parameters in the model
    print_trainable_parameters(model)

    # Define loss function and optimizer
    # Note: nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss, so we use raw logits
    class_weights = torch.tensor([1.0, pos_weights], dtype=torch.float, device=device)  # Weight negatives as 1, positives as pos_weight

    criterion = CustomLossCriterion(loss_type=loss_type, class_weights=class_weights, R=reg_coef, ratio=ratio, aaseq_to_ratio=aaseq_to_ratio, aaseq_to_dist=aaseq_to_dist, aaseq_to_nneighbors=aaseq_to_nneighbors, device=device)

    if optimizer_type == "adam":
        base_optimizer = optim.Adam
    elif optimizer_type == "adafactor":
        base_optimizer = optim.Adafactor
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    if model_type == "cvc" and not freeze_embed_model and special_criterion:
        encoder_lr = embedding_lr  # this is the default learning rate for BERT
        classification_head_lr = lr
        optimizer = base_optimizer([
            {'params': model.model.model.encoder.layer[12-args.cvc_layers_to_train:].parameters(), 'lr': encoder_lr},  # Later layers
            {'params': model.linear.parameters(), 'lr': classification_head_lr}  # Classification head
        ])
    else:
        optimizer = base_optimizer(model.parameters(), lr=lr)

    scheduler = get_scheduler(optimizer, scheduler_type)

    # Calculate number of batches
    num_pos_samples = len(train_pos_seqs)
    num_batches = num_pos_samples // pos_batch_size
    if num_pos_samples % pos_batch_size != 0:
        num_batches += 1

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_auc': [],
        'train_prauc': [],
        'train_pos_acc': [],
        'train_neg_acc': [],
        'train_precision': [],
        'train_recall': [],
        'train_tpr': [],  # True Positive Rate (Sensitivity)
        'train_tnr': [],  # True Negative Rate (Specificity)
        'train_fpr': [],  # False Positive Rate
        'train_fnr': [],  # False Negative Rate
        'train_f1': [],  # F1 Score
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_prauc': [],
        'val_pos_acc': [],
        'val_neg_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_tpr': [],  # True Positive Rate (Sensitivity)
        'val_tnr': [],  # True Negative Rate (Specificity)
        'val_fpr': [],  # False Positive Rate
        'val_fnr': [],  # False Negative Rate
        'val_f1': []  # F1 Score
    }

    # Setting up training negative sequences for this run
    if not change_negatives:
        # print the percentage of positives out of the full all negatives set: (only in training)
        percent_positives = (num_pos_samples / (num_pos_samples + len(neg_seqs))) * 100
        print(f"Percentage of positives out of all training data: {percent_positives:.2f}%")
        if len(neg_seqs) < num_pos_samples * neg_pos_ratio:
            raise ValueError(f"Not enough negative samples. Need at least {num_pos_samples * neg_pos_ratio}, but have only {len(neg_seqs)}.")
        train_neg_seqs = np.random.choice(neg_seqs, size=num_pos_samples * neg_pos_ratio, replace=False)
        print(f"Training on {num_pos_samples} positive samples and {len(train_neg_seqs)} negative samples (out of {len(neg_seqs)} all negatives)...")
    else:
        print(f"Training on {num_pos_samples} positive samples and {len(neg_seqs)} negative samples...")

    if reshef_inference:
        if reshef_negative_part > 0:
            rng = np.random.default_rng(42)
            # shuffle neg_seqs:
            rng.shuffle(neg_seqs)
            # Take the reshef_negative_part chunk from neg_seqs
            chunk_size = num_pos_samples * neg_pos_ratio
            start, end = int(chunk_size * (reshef_negative_part - 1)), int(chunk_size * reshef_negative_part)
            if end > len(neg_seqs):
                raise ValueError(f"reshef_negative_part {reshef_negative_part} is too large for the number of negative sequences {len(neg_seqs)}.")
            train_neg_seqs = neg_seqs[start:end]

        from utils.cache_handler import get_model_config_str
        model_config_string = get_model_config_str(args)
        if reshef_negative_part > 0:
            reshef_cache_folder = os.path.join("../cache", "reshef_inference", f"partition_{reshef_negative_part}", model_config_string)
        else:
            reshef_cache_folder = os.path.join("../cache", "reshef_inference", model_config_string)
        os.makedirs(reshef_cache_folder, exist_ok=True)

        if reshef_filter_train:
            reshef_inference_data_path = os.path.join(reshef_cache_folder, "reshef_inference_data.npz")
            combined_cache_folder = os.path.join("../cache", "reshef_inference", "combined_partitions", "combined_reshef_inference_data.npz")
            if os.path.exists(combined_cache_folder) and reshef_negative_part == 0:
                print(f"Loading Reshef inference data from combined cache: {combined_cache_folder}")
                # load to neg_seqs_to_train & pos_seqs_to_train
                reshef_inference_data = np.load(combined_cache_folder)
                pos_seqs_to_train = reshef_inference_data['pos_seqs_to_train']
                neg_seqs_to_train = reshef_inference_data['neg_seqs_to_train']
                # Set train_neg_seqs accordingly (by picking the right amount out of the pos_seqs to train:
                if len(neg_seqs_to_train) < num_pos_samples * neg_pos_ratio:
                    raise ValueError(f"Not enough negative sequences in combined cache. Need at least {num_pos_samples * neg_pos_ratio}, but have only {len(neg_seqs_to_train)}.")
                train_neg_seqs = np.random.choice(neg_seqs_to_train, size=num_pos_samples * neg_pos_ratio, replace=False)
            elif os.path.exists(reshef_inference_data_path):
                print(f"Loading Reshef inference data from: {reshef_inference_data_path}")
                reshef_inference_data = np.load(reshef_inference_data_path)

                # load the arrays from the npz file
                pos_seqs_to_train = reshef_inference_data['pos_seqs_to_train']
                neg_seqs_to_train = reshef_inference_data['neg_seqs_to_train']
                train_neg_seqs = neg_seqs_to_train
                neg_pos_ratio = 8
                args.neg_pos_ratio = neg_pos_ratio
                # keep only sequences that appear in train_pos_seqs and pos_seqs_to_train
                # mask = np.isin(train_pos_seqs, pos_seqs_to_train)
                # train_pos_seqs = train_pos_seqs[mask]
                # mask = np.isin(train_neg_seqs, neg_seqs_to_train)
                # train_neg_seqs = train_neg_seqs[mask]
            else:
                raise FileNotFoundError(f"Reshef inference data not found at {reshef_inference_data_path}. Please run the Reshef inference data generation script first.")

        # First swapping the first n_swap samples in train with the first n_swap samples in neg_seqs
        n_swap = 250  # Number of samples to swap for Reshef inference
        train_pos_seqs = np.array(train_pos_seqs)
        train_neg_seqs = np.array(train_neg_seqs)
        if len(train_pos_seqs) < n_swap or len(train_neg_seqs) < n_swap:
            raise ValueError("Not enough samples for Reshef inference. Need at least n_swap positive and n_swap negative samples.")
        train_pos_seqs[:n_swap], train_neg_seqs[:n_swap] = train_neg_seqs[:n_swap], train_pos_seqs[:n_swap]
        # Concatenate the sequences for Reshef inference
        full_data = np.concatenate([train_pos_seqs, train_neg_seqs,
                                    valid_pos_seqs, valid_neg_seqs])
        if test_pos_seqs is not None and test_neg_seqs is not None:
            test_data = np.concatenate([test_pos_seqs, test_neg_seqs])
            full_data = np.concatenate([full_data, test_data])
        # create a labels array for Reshef inference (1 for positive, 0 for negative), but make sure label the first n_swap correctly because of the swap
        full_labels = np.concatenate([np.zeros(len(train_pos_seqs[:n_swap])), np.ones(len(train_pos_seqs[n_swap:])),
                                      np.ones(len(train_neg_seqs[:n_swap])), np.zeros(len(train_neg_seqs[n_swap:])),
                                      np.ones(len(valid_pos_seqs)), np.zeros(len(valid_neg_seqs))])
        if test_pos_seqs is not None and test_neg_seqs is not None:
            test_labels = np.concatenate([np.ones(len(test_pos_seqs)), np.zeros(len(test_neg_seqs))])
            full_labels = np.concatenate([full_labels, test_labels])
        inds = [0, len(train_pos_seqs[:n_swap]), len(train_pos_seqs[n_swap:]),
                len(train_neg_seqs[:n_swap]), len(train_neg_seqs[n_swap:]),
                len(valid_pos_seqs), len(valid_neg_seqs)]
        if test_pos_seqs is not None and test_neg_seqs is not None:
            inds.append(len(test_pos_seqs))
            inds.append(len(test_neg_seqs))
        for i in range(1, len(inds)):
            inds[i] = inds[i - 1] + inds[i]
        inds = np.array(inds)
        # save the full data & labels for Reshef inference under "cache/reshef_inference"
        np.save(os.path.join(reshef_cache_folder, "full_data.npy"), full_data)
        np.save(os.path.join(reshef_cache_folder, "full_labels.npy"), full_labels)
        np.save(os.path.join(reshef_cache_folder, "n_swap.npy"), np.array([n_swap]))
        np.savez(os.path.join(reshef_cache_folder, "parameters.npz"), n_swap=n_swap, reshef_negative_part=reshef_negative_part)
        np.save(os.path.join(reshef_cache_folder, "inds.npy"), inds)

    # Load model state if available
    start_epoch = 0
    if not is_sweep and False:  # TODO: Added false because this might cause problems!
        # Try to load the model from epoch args.epoch to 1 if it exists
        for epoch in range(epochs, 0, -1):
            trained_model = load_model_state(model, args, epoch, device)
            if trained_model is not None:
                start_epoch = epoch
                model = trained_model
                print(f"Loaded model state from epoch {epoch}")
                break

    do_inference_instead_of_train = False
    if do_inference_instead_of_train:
        from utils.cache_handler import get_model_config_str
        model_config_string = get_model_config_str(args)

        import json
        from itertools import product
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

        # -------- One-hot encoding helper --------
        def one_hot_encode_sequences(seqs, alphabet=None, max_len=None):
            """One-hot encode sequences with zero-padding to max_len, flatten for sklearn."""
            if alphabet is None:
                alphabet = sorted(set(''.join(seqs)))
            if max_len is None:
                max_len = max(len(s) for s in seqs)
            char_index = {c: i for i, c in enumerate(alphabet)}
            encoded = np.zeros((len(seqs), max_len, len(alphabet)), dtype=np.float32)
            for i, seq in enumerate(seqs):
                for j, c in enumerate(seq):
                    if j < max_len and c in char_index:
                        encoded[i, j, char_index[c]] = 1.0
            return encoded.reshape(len(seqs), -1), alphabet, max_len

        # -------- Get embeddings using trained model --------
        def get_model_embeddings(model, seqs, device='cuda', batch_size=330):
            """Get embeddings from model.get_embeddings for given sequences in batches."""
            model.eval()
            all_embs = []
            with torch.no_grad():
                for i in range(0, len(seqs), batch_size):
                    batch = seqs[i:i + batch_size]
                    emb = model.get_embeddings(batch)
                    if isinstance(emb, torch.Tensor):
                        emb = emb.to('cpu').numpy()
                    all_embs.append(emb)
            return np.concatenate(all_embs, axis=0)

        # -------- Prepare one-hot encoded data --------
        def prepare_onehot_data(train_pos, train_neg, valid_pos, valid_neg, test_pos, test_neg):
            all_train = np.concatenate([train_pos, train_neg])
            all_valid = np.concatenate([valid_pos, valid_neg])
            all_test = np.concatenate([test_pos, test_neg])
            alphabet = sorted(set(''.join(all_train) + ''.join(all_valid) + ''.join(all_test)))
            max_len = max(len(s) for s in np.concatenate([all_train, all_valid, all_test]))

            X_train, alphabet, max_len = one_hot_encode_sequences(all_train, alphabet, max_len)
            y_train = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])

            X_valid, _, _ = one_hot_encode_sequences(all_valid, alphabet, max_len)
            y_valid = np.concatenate([np.ones(len(valid_pos)), np.zeros(len(valid_neg))])

            X_test, _, _ = one_hot_encode_sequences(all_test, alphabet, max_len)
            y_test = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])

            return X_train, y_train, X_valid, y_valid, X_test, y_test

        # -------- Prepare embedding-based data --------
        def prepare_embedding_data(model, train_pos, train_neg, valid_pos, valid_neg, test_pos, test_neg, device='cuda',
                                   batch_size=330):
            X_train = get_model_embeddings(model, np.concatenate([train_pos, train_neg]), device, batch_size)
            y_train = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])

            X_valid = get_model_embeddings(model, np.concatenate([valid_pos, valid_neg]), device, batch_size)
            y_valid = np.concatenate([np.ones(len(valid_pos)), np.zeros(len(valid_neg))])

            X_test = get_model_embeddings(model, np.concatenate([test_pos, test_neg]), device, batch_size)
            y_test = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])

            return X_train, y_train, X_valid, y_valid, X_test, y_test

        # -------- Generic KNN fitting and evaluation --------
        def tune_and_evaluate_knn(X_train, y_train, X_valid, y_valid, X_test, y_test):
            k_values = [1, 3, 5, 7, 9]
            weights_options = ['uniform', 'distance']
            best_f1, best_params, best_model = -1, None, None
            best_tpr, best_tnr, best_fpr, best_fnr = -1, -1, -1, -1

            for k, w in product(k_values, weights_options):
                knn = KNeighborsClassifier(n_neighbors=k, weights=w, n_jobs=-1)
                knn.fit(X_train, y_train)
                y_val_pred = knn.predict(X_valid)
                f1 = f1_score(y_valid, y_val_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = (k, w)
                    best_model = knn
                    tn, fp, fn, tp = confusion_matrix(y_valid, y_val_pred).ravel()
                    best_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity
                    best_tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
                    best_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Fall-out
                    best_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # Miss Rate

            y_test_pred = best_model.predict(X_test)
            metrics = {
                "best_validation_f1": best_f1,
                "best_k": best_params[0],
                "best_weights": best_params[1],
                "test_f1": f1_score(y_test, y_test_pred),
                "test_precision": precision_score(y_test, y_test_pred),
                "test_recall": recall_score(y_test, y_test_pred),
                "test_confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist(),
                "test_tpr": best_tpr,
                "test_tnr": best_tnr,
                "test_fpr": best_fpr,
                "test_fnr": best_fnr
            }
            return metrics

        # -------- Main pipeline for both versions --------
        def run_and_save_all(model,
                             train_pos_seqs, train_neg_seqs,
                             valid_pos_seqs, valid_neg_seqs,
                             test_pos_seqs, test_neg_seqs,
                             model_config_string,
                             device='cuda', batch_size=330):
            # Ensure output folder exists
            out_dir = os.path.join("../naive_model_results", model_config_string)
            os.makedirs(out_dir, exist_ok=True)

            # 1) One-hot KNN
            X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_onehot_data(
                train_pos_seqs, train_neg_seqs, valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs)
            onehot_metrics = tune_and_evaluate_knn(X_train, y_train, X_valid, y_valid, X_test, y_test)

            # Save one-hot results
            with open(os.path.join(out_dir, "onehot_results.json"), "w") as f:
                json.dump(onehot_metrics, f, indent=4)
            print("One-hot KNN results saved.")

            # 2) Embedding KNN
            X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_embedding_data(
                model, train_pos_seqs, train_neg_seqs, valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs, device,
                batch_size)
            embedding_metrics = tune_and_evaluate_knn(X_train, y_train, X_valid, y_valid, X_test, y_test)

            # Save embedding results
            with open(os.path.join(out_dir, "embedding_results.json"), "w") as f:
                json.dump(embedding_metrics, f, indent=4)

            print(f"All results saved under: {out_dir}")
            return onehot_metrics, embedding_metrics

        onehot_metrics, embedding_metrics = run_and_save_all(model, train_pos_seqs, train_neg_seqs,
                                                             valid_pos_seqs, valid_neg_seqs,
                                                             test_pos_seqs, test_neg_seqs,
                                                             model_config_string, device='cuda', batch_size=330)
        exit(0)

    # Training loop
    for epoch in range(start_epoch, epochs):
        start_time = time.time()

        # Train phase
        model.train()
        epoch_loss = 0
        train_total = 0
        train_correct = 0
        correct = 0
        total = 0
        train_y_true = []
        train_y_pred = []
        train_y_scores = []

        if reshef_inference:
            # Run the model on the full data for Reshef inference in batches, each time move to cpu
            model.eval()
            model_outputs = []
            with torch.no_grad():
                batch_size = pos_batch_size * (neg_pos_ratio + 1)  # Batch size for Reshef inference
                for i in range(0, len(full_data), batch_size):
                    batch_samples = full_data[i:i + batch_size]
                    logits = model(batch_samples).cpu().numpy()
                    model_outputs.append(logits)
            model_outputs = np.concatenate(model_outputs, axis=0)
            # Save the model outputs for Reshef inference under "cache/reshef_inference"
            np.save(os.path.join(reshef_cache_folder, f"model_outputs_epoch-{epoch}.npy"), model_outputs)
            print(f"Saved model outputs for Reshef inference at epoch {epoch} in: {reshef_cache_folder}")
            model.train()

        # Shuffle positive samples for this epoch
        pos_indices = np.arange(num_pos_samples)
        np.random.shuffle(pos_indices)
        # Shuffle negative samples for this epoch
        if not change_negatives:
            neg_indices = np.arange(len(train_neg_seqs))
            np.random.shuffle(neg_indices)

        for batch_idx in range(num_batches):
            # Get positive samples for this batch
            start_idx = batch_idx * pos_batch_size
            end_idx = min((batch_idx + 1) * pos_batch_size, num_pos_samples)
            batch_pos_indices = pos_indices[start_idx:end_idx]
            batch_pos_samples = train_pos_seqs[batch_pos_indices]

            # Get negative samples for this batch
            if not change_negatives:
                neg_batch_size = len(batch_pos_samples) * neg_pos_ratio
                start_idx = batch_idx * neg_batch_size
                end_idx = min((batch_idx + 1) * neg_batch_size, len(train_neg_seqs))
                batch_neg_indices = neg_indices[start_idx:end_idx]
                batch_neg_samples = train_neg_seqs[batch_neg_indices]
            else:
                # Get negative samples for this batch (without repetition)
                neg_batch_size = len(batch_pos_samples) * neg_pos_ratio
                batch_neg_samples = np.random.choice(neg_seqs, size=neg_batch_size, replace=False)

            # Combine positive and negative samples
            batch_samples = np.concatenate([batch_pos_samples, batch_neg_samples])

            # Create labels: 1 for positive, 0 for negative
            batch_labels = torch.zeros(len(batch_samples), dtype=torch.long)
            batch_labels[:len(batch_pos_samples)] = 1

            # Shuffle samples and labels together
            indices = torch.randperm(len(batch_samples))
            batch_samples = batch_samples[indices.numpy()]
            batch_labels = batch_labels[indices].to(device)

            if masking and model_type == "cvc":
                model.set_mask_on()

            # Forward pass - get logits
            logits = model(batch_samples)

            if masking and model_type == "cvc":
                model.set_mask_off()

            # Calculate loss using raw logits (CrossEntropyLoss applies softmax internally)
            loss = criterion(logits, batch_labels, batch_samples)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # remove from gpu
            logits = logits.detach()
            batch_labels = batch_labels.detach()

            # For accuracy calculation, we need to get predictions from logits
            probabilities = torch.softmax(logits, dim=1)
            _, predicted = torch.max(probabilities, 1)

            # Update statistics
            epoch_loss += loss.item()
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

            # Update statistics
            epoch_loss += loss.item()
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

            # Collect predictions and true labels for metrics calculation
            train_y_true.extend(batch_labels.cpu().numpy())
            train_y_pred.extend(predicted.cpu().numpy())
            train_y_scores.extend(probabilities[:, 1].detach().cpu().numpy())

        # Calculate training statistics
        train_loss = epoch_loss / num_batches
        train_acc = 100 * correct / total

        # Calculate training confusion matrix metrics
        train_y_true = np.array(train_y_true)
        train_y_pred = np.array(train_y_pred)
        train_y_scores = np.array(train_y_scores)

        train_tp = ((train_y_pred == 1) & (train_y_true == 1)).sum()
        train_fp = ((train_y_pred == 1) & (train_y_true == 0)).sum()
        train_tn = ((train_y_pred == 0) & (train_y_true == 0)).sum()
        train_fn = ((train_y_pred == 0) & (train_y_true == 1)).sum()

        # Calculate training rates
        train_pos_total = (train_y_true == 1).sum()
        train_neg_total = (train_y_true == 0).sum()

        train_pos_acc = 100 * train_tp / train_pos_total if train_pos_total > 0 else 0
        train_neg_acc = 100 * train_tn / train_neg_total if train_neg_total > 0 else 0

        train_precision = train_tp / (train_tp + train_fp) if (train_tp + train_fp) > 0 else 0
        train_recall = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0
        train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
        # Same as recall
        train_tpr = train_recall
        train_tnr = train_tn / train_neg_total if train_neg_total > 0 else 0  # Specificity
        train_fpr = train_fp / train_neg_total if train_neg_total > 0 else 0  # Fall-out
        train_fnr = train_fn / train_pos_total if train_pos_total > 0 else 0  # Miss rate

        # Validation phase
        val_metrics = evaluate_model(model, valid_pos_seqs, valid_neg_seqs, criterion, device)
        val_loss, val_acc, val_auc, val_prauc, val_tp, val_fp, val_tn, val_fn, val_pos_acc, val_neg_acc, val_precision, val_recall, val_tpr, val_tnr, val_fpr, val_fnr, val_f1 = val_metrics

        # Calculate training AUC and PR-AUC
        try:
            train_auc = roc_auc_score(train_y_true, train_y_scores)
            train_pr_precision, train_pr_recall, _ = precision_recall_curve(train_y_true, train_y_scores)
            train_prauc = auc(train_pr_recall, train_pr_precision)
        except:
            train_auc = 0
            train_prauc = 0
        if scheduler is not None:
            if scheduler_type == "ReduceLROnPlateau".lower():
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Update history with all metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_auc'].append(train_auc)
        history['train_prauc'].append(train_prauc)
        history['train_pos_acc'].append(train_pos_acc)
        history['train_neg_acc'].append(train_neg_acc)
        history['train_precision'].append(train_precision)
        history['train_recall'].append(train_recall)
        history['train_tpr'].append(train_tpr)
        history['train_tnr'].append(train_tnr)
        history['train_fpr'].append(train_fpr)
        history['train_fnr'].append(train_fnr)
        history['train_f1'].append(train_f1)

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_prauc'].append(val_prauc)
        history['val_pos_acc'].append(val_pos_acc)
        history['val_neg_acc'].append(val_neg_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_tpr'].append(val_tpr)
        history['val_tnr'].append(val_tnr)
        history['val_fpr'].append(val_fpr)
        history['val_fnr'].append(val_fnr)
        history['val_f1'].append(val_f1)

        # Calculate epoch time
        epoch_time = time.time() - start_time

        # Print epoch results with expanded metrics
        print(f'Epoch {epoch + 1}/{epochs} - {epoch_time:.2f}s')
        print(
            f'  Train: Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% - AUC: {train_auc:.4f} - PRAUC: {train_prauc:.4f}')
        print(f'         TP: {train_tp} - FP: {train_fp} - TN: {train_tn} - FN: {train_fn}')
        print(
            f'         TPR: {train_tpr:.4f} - TNR: {train_tnr:.4f} - Precision: {train_precision:.4f} - F1: {train_f1:.4f}')
        print(f'  Val:   Loss: {val_loss:.4f} - Acc: {val_acc:.2f}% - AUC: {val_auc:.4f} - PRAUC: {val_prauc:.4f}')
        print(f'         TP: {val_tp} - FP: {val_fp} - TN: {val_tn} - FN: {val_fn}')
        print(f'         TPR: {val_tpr:.4f} - TNR: {val_tnr:.4f} - Precision: {val_precision:.4f} - F1: {val_f1:.4f}')

        if log_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "epoch_time": epoch_time,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_auc": train_auc,
                "train_prauc": train_prauc,
                "train_tp": train_tp,
                "train_fp": train_fp,
                "train_tn": train_tn,
                "train_fn": train_fn,
                "train_pos_acc": train_pos_acc,
                "train_neg_acc": train_neg_acc,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_tpr": train_tpr,
                "train_tnr": train_tnr,
                "train_fpr": train_fpr,
                "train_fnr": train_fnr,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "val_prauc": val_prauc,
                "val_tp": val_tp,
                "val_fp": val_fp,
                "val_tn": val_tn,
                "val_fn": val_fn,
                "val_pos_acc": val_pos_acc,
                "val_neg_acc": val_neg_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_tpr": val_tpr,
                "val_tnr": val_tnr,
                "val_fpr": val_fpr,
                "val_fnr": val_fnr,
                "val_f1": val_f1
            })

        # saving the model for this epoch on odd epochs or on last epoch
        # is_odd_or_last_epoch = ((epoch + 1) % 2 == 1 and epoch >= 15) or epoch + 1 == epochs
        # if not is_sweep and is_odd_or_last_epoch:
        # saving every epoch after 5
        if not is_sweep and (epoch == 8 or epoch == epochs - 1):
            save_model_state(model, args, epoch)
        elif reshef_inference:
            save_model_state(model, args, epoch)

    return model, history


def evaluate_model(model, pos_seqs, neg_seqs, criterion=None, device=None):
    """
    Evaluate the model on positive and negative sequences.

    Args:
        model: The model to evaluate (returns logits)
        loss_type: Type of loss function
        aaseq_to_ratio: Function that maps amino acid sequences to ratios
        pos_seqs: numpy array of positive sequences (strings)
        neg_seqs: numpy array of negative sequences (strings)
        criterion: Loss function (optional)
        device: Torch device (optional)

    Returns:
        loss, accuracy, AUC, PR-AUC, TP, FP, TN, FN, pos_acc, neg_acc, precision, recall
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()

    # Prepare data
    all_samples = np.concatenate([pos_seqs, neg_seqs])
    all_labels = torch.zeros(len(all_samples), dtype=torch.long)
    all_labels[:len(pos_seqs)] = 1

    # Shuffle
    indices = torch.randperm(len(all_samples))
    all_samples = all_samples[indices.numpy()]
    all_labels = all_labels[indices].to(device)

    with torch.no_grad():
        # Forward pass to get logits
        logits = model(all_samples)

        # Calculate loss using raw logits
        loss = criterion(logits, all_labels, all_samples).item()

        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=1)

        # Calculate accuracy
        _, predicted = torch.max(probabilities, 1)
        total = all_labels.size(0)
        correct = (predicted == all_labels).sum().item()
        accuracy = 100 * correct / total

        # Calculate confusion matrix metrics
        true_labels = all_labels.cpu().numpy()
        predicted_labels = predicted.cpu().numpy()

        # Calculate true positives, false positives, true negatives, false negatives
        TP = ((predicted_labels == 1) & (true_labels == 1)).sum().item()
        FP = ((predicted_labels == 1) & (true_labels == 0)).sum().item()
        TN = ((predicted_labels == 0) & (true_labels == 0)).sum().item()
        FN = ((predicted_labels == 0) & (true_labels == 1)).sum().item()

        # Calculate accuracy on positives and negatives
        pos_total = (true_labels == 1).sum()
        neg_total = (true_labels == 0).sum()
        pos_acc = 100 * TP / pos_total if pos_total > 0 else 0
        neg_acc = 100 * TN / neg_total if neg_total > 0 else 0

        # Calculate precision and recall
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        # Calculate additional rates
        tpr = recall  # Same as recall
        tnr = TN / neg_total if neg_total > 0 else 0  # Specificity
        fpr = FP / neg_total if neg_total > 0 else 0  # Fall-out
        fnr = FN / pos_total if pos_total > 0 else 0  # Miss rate

        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate AUC and PR-AUC
        pos_probs = probabilities[:, 1].cpu().numpy()

        try:
            roc_auc = roc_auc_score(true_labels, pos_probs)
            pr_precision, pr_recall, _ = precision_recall_curve(true_labels, pos_probs)
            pr_auc = auc(pr_recall, pr_precision)
        except:
            roc_auc = 0
            pr_auc = 0

    return (loss, accuracy, roc_auc, pr_auc, TP, FP, TN, FN, pos_acc, neg_acc,
            precision, recall, tpr, tnr, fpr, fnr, f1)


def display_training_results(history, model_type, figsize=(15, 10)):
    """
    Display training results with matplotlib.

    Args:
        history: Training history dictionary
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)

    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Plot AUC
    plt.subplot(2, 2, 3)
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.title('ROC AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)

    # Plot PR-AUC
    plt.subplot(2, 2, 4)
    plt.plot(history['val_prauc'], label='Validation PR-AUC')
    plt.title('Precision-Recall AUC')
    plt.xlabel('Epoch')
    plt.ylabel('PR-AUC')
    plt.legend()
    plt.grid(True)

    # Save and show figure
    plt.tight_layout()
    os.makedirs(f"plots/{model_type}_model", exist_ok=True)
    plt.savefig(f"plots/{model_type}_model/training_results.png")
    plt.show()
