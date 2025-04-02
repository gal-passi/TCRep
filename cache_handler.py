import os
import torch


def get_model_config_str(args):
    config_str = f"{args.model_type}_loss-{args.loss_type}_epochs-{args.epochs}_" \
                 f"batch-{args.batch_size}_ratio-{args.neg_pos_ratio}_weights-{args.pos_weights}_" \
                 f"lr-{args.learning_rate}_regcoef-{args.regularization_coefficient}_freeze-{args.freeze_embed_model}_criterion-{args.special_criterion}"
    return config_str


def get_model_save_path(args, epoch, make_dirs=True):
    # Define the base save directory
    base_dir = "cache/models"
    if make_dirs:
        os.makedirs(base_dir, exist_ok=True)

    # Create a readable folder name based on the model configuration
    config_str = get_model_config_str(args)
    save_dir = os.path.join(base_dir, config_str)
    if make_dirs:
        os.makedirs(save_dir, exist_ok=True)

    # Define the save path for the model state_dict
    model_save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")

    return model_save_path


def save_model_state(model, args, epoch):
    # Get the save path for the model state_dict
    model_save_path = get_model_save_path(args, epoch, make_dirs=True)

    # Save the model state_dict
    torch.save(model.state_dict(), model_save_path)


def load_model_state(model, args, epoch, device):
    # Get the expected model save path
    model_save_path = get_model_save_path(args, epoch, make_dirs=False)

    # Load the model if the file exists
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        return model
    else:
        print("No saved model found for the given epoch and configuration.")
        return None