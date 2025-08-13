import os
import torch


def get_model_config_str(args):
    metadata = ""
    if args.k_fold > 0:
        metadata += f"_fold-{args.k_fold}"
    if args.negative_partition > 0:
        metadata += f"_neg-{args.negative_partition}"
    if args.to_ensemble:
        metadata += f"_ensemble-{args.to_ensemble}"
    if args.dataset_filter_num_of_patients != 3:
        metadata += f"_npatients-{args.dataset_filter_num_of_patients}"
    if args.dataset_filter_num_of_healthy != args.dataset_filter_num_of_patients and args.dataset_filter_num_of_healthy != -1:
        metadata += f"_nhealthy-{args.dataset_filter_num_of_healthy}"

    config_str = f"{args.model_type}_loss-{args.loss_type}_dataset-{args.dataset_type}{metadata}_epochs-{args.epochs}_" \
                 f"batch-{args.batch_size}_ratio-{args.neg_pos_ratio}_weights-{args.pos_weights}_" \
                 f"lr-{args.learning_rate}_regcoef-{args.regularization_coefficient}_freeze-{args.freeze_embed_model}_criterion-{args.special_criterion}"
    return config_str


def get_model_dir(args, make_dirs=True):
    # Define the base directory for saving models
    base_dir = "cache/models"
    if make_dirs:
        os.makedirs(base_dir, exist_ok=True)

    # Create a readable folder name based on the model configuration
    config_str = get_model_config_str(args)
    save_dir = os.path.join(base_dir, config_str)
    if make_dirs:
        os.makedirs(save_dir, exist_ok=True)

    return save_dir


def get_model_save_path(args, epoch, make_dirs=True):
    # Get the directory for saving the model
    save_dir = get_model_dir(args, make_dirs=make_dirs)

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

def get_embedding_save_path(args, metadata='', make_dirs=True):
    save_path = get_model_dir(args, make_dirs=make_dirs)
    return os.path.join(save_path, f"embedding{metadata}.csv")
