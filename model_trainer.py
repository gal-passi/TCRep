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
from cache_handler import save_model_state
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
    def __init__(self, loss_type='ce', class_weights=None, R=0.1, n_classes=2):
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

    def forward(self, logits, labels):
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
            class_weights = self.class_weights.to(labels.device)

        if self.loss_type == 'ce':
            # Standard Cross Entropy with optional class weights
            return F.cross_entropy(logits, labels, weight=self.class_weights)

        elif self.loss_type == 'ce_l2':
            # Custom L2 regularized loss
            ce_loss = F.cross_entropy(logits, labels, weight=self.class_weights)
            probs = F.softmax(logits, dim=1)
            l2_norm = torch.norm(probs, p=2, dim=1).mean()
            reg_term = (1 - l2_norm)
            return ce_loss + self.R * reg_term

        elif self.loss_type == 'ce_entropy':
            # Custom entropy regularized loss
            ce_loss = F.cross_entropy(logits, labels, weight=self.class_weights)
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
                embedding_lr, reg_coef, pos_weights, aaseq_to_ratio, args, scheduler_type='none',
                epochs=10, lr=0.0005, pos_batch_size=30, neg_pos_ratio=10, is_sweep=False):  # pos_batch_size=256
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

    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')  # TODO: Changing this for ratio tests!
    else:
        criterion = CustomLossCriterion(loss_type=loss_type, class_weights=class_weights, R=reg_coef)

    if model_type == "cvc" and not freeze_embed_model and special_criterion:
        encoder_lr = embedding_lr  # this is the default learning rate for BERT
        classification_head_lr = lr
        optimizer = optim.Adam([
            {'params': model.model.model.encoder.layer[12-args.cvc_layers_to_train:].parameters(), 'lr': encoder_lr},  # Later layers
            {'params': model.linear.parameters(), 'lr': classification_head_lr}  # Classification head
        ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

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
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_prauc': []
    }

    # Starting training
    print(f"Training on {num_pos_samples} positive samples and {len(neg_seqs)} negative samples...")

    # Training loop
    for epoch in range(epochs):
        start_time = time.time()

        # Train phase
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        # Shuffle positive samples for this epoch
        pos_indices = np.arange(num_pos_samples)
        np.random.shuffle(pos_indices)

        for batch_idx in range(num_batches):
            # Get positive samples for this batch
            start_idx = batch_idx * pos_batch_size
            end_idx = min((batch_idx + 1) * pos_batch_size, num_pos_samples)
            batch_pos_indices = pos_indices[start_idx:end_idx]
            batch_pos_samples = train_pos_seqs[batch_pos_indices]

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

            # Forward pass - get logits
            logits = model(batch_samples)

            # Calculate loss using raw logits (CrossEntropyLoss applies softmax internally)
            if loss_type == "ce":  # TODO: For now, making use of ratio is possible only when using ce loss!!! Change this later!
                batch_sample_ratios = aaseq_to_ratio(batch_samples)
                per_sample_losses = criterion(logits, batch_labels)
                sample_weights = torch.ones_like(per_sample_losses)
                positive_indices = batch_labels == 1
                sample_weights[positive_indices] = torch.tensor(
                    batch_sample_ratios[positive_indices],
                    dtype=torch.float32,
                    device=per_sample_losses.device
                )
                weighted_losses = per_sample_losses * sample_weights
                loss = weighted_losses.mean()
            else:
                loss = criterion(logits, batch_labels)


            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # For accuracy calculation, we need to get predictions from logits
            probabilities = torch.softmax(logits, dim=1)
            _, predicted = torch.max(probabilities, 1)

            # Update statistics
            epoch_loss += loss.item()
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        # Calculate training statistics
        train_loss = epoch_loss / num_batches
        train_acc = 100 * correct / total

        # Validation phase
        val_loss, val_acc, val_auc, val_prauc = evaluate_model(model, loss_type, aaseq_to_ratio, valid_pos_seqs, valid_neg_seqs, criterion, device)
        if scheduler is not None:
            if scheduler_type == "ReduceLROnPlateau".lower():
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_prauc'].append(val_prauc)

        # Calculate epoch time
        epoch_time = time.time() - start_time

        # Print epoch results
        print(f'Epoch {epoch + 1}/{epochs} - {epoch_time:.2f}s - Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% - '
              f'Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}% - Val AUC: {val_auc:.4f} - Val PRAUC: {val_prauc:.4f}')
        if log_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "epoch_time": epoch_time,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "val_prauc": val_prauc
            })

        # saving the model for this epoch on odd epochs or on last epoch
        is_odd_or_last_epoch = ((epoch + 1) % 2 == 1 and epoch >= 15) or epoch + 1 == epochs
        if not is_sweep and is_odd_or_last_epoch:
            save_model_state(model, args, epoch)

    return model, history


def evaluate_model(model, loss_type, aaseq_to_ratio, pos_seqs, neg_seqs, criterion=None, device=None):
    """
    Evaluate the model on positive and negative sequences.

    Args:
        model: The model to evaluate (returns logits)
        pos_seqs: numpy array of positive sequences (strings)
        neg_seqs: numpy array of negative sequences (strings)
        criterion: Loss function (optional)
        device: Torch device (optional)

    Returns:
        loss, accuracy, AUC, PR-AUC
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
        if loss_type == "ce":  # TODO: For now, making use of ratio is possible only when using ce loss!!! Change this later!
            batch_sample_ratios = aaseq_to_ratio(all_samples)
            per_sample_losses = criterion(logits, all_labels)
            sample_weights = torch.ones_like(per_sample_losses)
            positive_indices = all_labels == 1
            sample_weights[positive_indices] = torch.tensor(
                batch_sample_ratios[positive_indices],
                dtype=torch.float32,
                device=per_sample_losses.device
            )
            weighted_losses = per_sample_losses * sample_weights
            loss = weighted_losses.mean()
        else:
            loss = criterion(logits, all_labels).item()

        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=1)

        # Calculate accuracy
        _, predicted = torch.max(probabilities, 1)
        total = all_labels.size(0)
        correct = (predicted == all_labels).sum().item()
        accuracy = 100 * correct / total

        # Calculate AUC and PR-AUC
        pos_probs = probabilities[:, 1].cpu().numpy()
        true_labels = all_labels.cpu().numpy()

        try:
            roc_auc = roc_auc_score(true_labels, pos_probs)
            precision, recall, _ = precision_recall_curve(true_labels, pos_probs)
            pr_auc = auc(recall, precision)
        except:
            roc_auc = 0.5
            pr_auc = 0.5

    return loss, accuracy, roc_auc, pr_auc


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
