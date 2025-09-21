import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import os
import argparse
import torch
import numpy as np
import wandb
import yaml
import pandas as pd
from dataset_handlers.dataset_loader import DatasetLoader
from models.model_factory import build_model
from training.model_trainer import train_model
from training.model_loader import load_or_train_model
from inference.plot_handler import plot_all_training, plot_all_dataset
from inference.inference_handler import run_inference
from vae.vae_handler import run_vae


# Display options
to_display_dict = {
    "TO_DISPLAY_LENGTHS_HIST": False,
    "TO_DISPLAY_COMMON_SEQUENCES": True,
    "TO_DISPLAY_ACCURACY_BIN_BY_DIST": False,
    "TO_DISPLAY_RESULTS": False,
    "TO_DISPLAY_RESULTS_PLOT_TSNE": False,
    "TO_LOAD_FULL_SYNAPSE_DATA": False,
    "TO_DISPLAY_RATIO_FIGURES": False,
}

# Inference options
inference_dict = {
    "INFERENCE_TO_RANDOM_FOREST": False,
    "INFERENCE_TO_RF_PLOT_DIST_PER_PATIENT": False,
    "INFERENCE_TO_DISPLAY_OTHER_DATASET_DISTS": False,
    "INFERENCE_CLASSIFICATION_MODEL": False,
    "INFERENCE_TO_PLOT_EMBEDDING_MAPPINGS": False,
    "INFERENCE_TO_CLASSIFICATION_MODEL": True,
    "INFERENCE_RESHEF": False,
    "INFERENCE_MCPAS": False,
}


def wand_init(model_type, loss_type, dataset_type, epochs, batch_size, neg_pos_ratio, pos_weights,
              learning_rate, reg_coef, freeze_embed_model, special_criterion, embedding_lr, ch_dropout,
              scheduler_type, cvc_layers_to_train, k_fold, lora, masking, ratio, use_dist_loss, dist_param, use_recurrence_loss, recurrence_gamma, ch_type, neg_partition,
              use_similar_negatives, filter_num_of_patients, filter_num_of_healthy, filter_to_inflate, change_negatives, sample_plots, optimizer_type, device):
    wandb.login(key="c8ebb98c8047d30555fd4d042ea969052ca18607")  # Replace with your API key

    # Start a new wandb run to track this script.
    run = wandb.init(
        entity="amir-weinfeld",  # Set the wandb entity where your project will be logged
        project="TCRep",  # Set the wandb project where this run will be logged
        config={
            "model_type": model_type,
            "loss_type": loss_type,
            "dataset_type": dataset_type,
            "optimizer_type": optimizer_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "neg_pos_ratio": neg_pos_ratio,
            "pos_weights": pos_weights,
            "learning_rate": learning_rate,
            "reg_coef": reg_coef,
            "freeze_embed_model": freeze_embed_model,
            "special_criterion": special_criterion,
            "embedding_lr": embedding_lr,
            "ch_dropout": ch_dropout,
            "scheduler_type": scheduler_type,
            "cvc_layers_to_train": cvc_layers_to_train,
            "k_fold": k_fold,
            "lora": lora,
            "masking": masking,
            "ratio": ratio,
            "use_dist_loss": use_dist_loss,
            "dist_param": dist_param,
            "use_recurrence_loss": use_recurrence_loss,
            "recurrence_gamma": recurrence_gamma,
            "ch_type": ch_type,
            "neg_partition": neg_partition,
            "use_similar_negatives": use_similar_negatives,
            "filter_num_of_patients": filter_num_of_patients,
            "filter_num_of_healthy": filter_num_of_healthy,
            "filter_to_inflate": filter_to_inflate,
            "change_negatives": change_negatives,
            "sample_plots": sample_plots,
            "device": device,
        },
        notes="Added dropout on classification head of 0.2",
    )
    return run


def get_dataset_loader(dataset_type, k_fold=0, to_k_fold=True, use_dist_loss=False, dist_param=0.0, neg_partition=0,
                        use_similar_negatives=False, neg_pos_ratio=10, filter_num_of_patients=None, filter_num_of_healthy=None, ratio=None,
                        filter_to_inflate=False, remove_seqs_by_len=None, top_percent=None, top_n_seqs=None, extra_filter=False,
                       use_recurrence_loss=False, recurrence_gamma=1.0, loss_version=0, run_on_full_data=False, sweep_loader_mode=False, fisher_mode=False,
                       extract_dataset_info=False, verbose=True):
    np.random.seed(42)
    # Load data
    unique_patient_ids = None
    if dataset_type == 'cmv':
        num_test_patients = 4
    elif 'article_sle' in dataset_type or 't1d' in dataset_type:
        num_test_patients = 10
    else:
        num_test_patients = 8
    if to_k_fold:
        dataset_loader = DatasetLoader(dataset_type=dataset_type, get_only_unique_patient_ids=True, top_percent=top_percent,
                                       extra_filter=extra_filter, top_n_seqs=top_n_seqs, use_recurrence_loss=use_recurrence_loss, recurrence_gamma=recurrence_gamma)
        df_bld, df_hlt = dataset_loader.get_dfs()
        unique_patient_ids = df_bld["patient_id"].unique()
        unique_patient_ids = np.random.permutation(unique_patient_ids)
        def generate_shifted_lists(patient_ids):
            """
            Generate altered lists by shifting the original list of patient IDs.
            Each altered list is shifted by increments of 8 positions to the right.
            Stops generating lists when any element from the first 8 positions would reappear.

            Args:
                patient_ids: List of unique patient ID strings

            Returns:
                A list of altered lists
            """
            n = len(patient_ids)

            # If the list has 8 or fewer elements, we can only create one list
            if n <= num_test_patients:
                return [patient_ids.copy()]

            # Calculate how many shifts we can make without bringing back elements from first num_test_patients positions
            first_eight = set(patient_ids[:num_test_patients])
            max_shifts = (n // num_test_patients) - 1

            # Create the altered lists
            altered_lists = []

            for shift_count in range(max_shifts + 1):
                # Calculate the shift amount
                shift = (shift_count * num_test_patients) % n

                # Create a new shifted list
                shifted_list = patient_ids[shift:] + patient_ids[:shift]

                # Check if any of the first num_test_patients elements are in the shifted list
                if len(first_eight.intersection(set(shifted_list[:num_test_patients]))) > 0 and shift_count > 0:
                    print(shifted_list[:num_test_patients], patient_ids[:num_test_patients])

                # Add to our collection of altered lists
                altered_lists.append(shifted_list)

            return np.array(altered_lists)

        altered_lists = generate_shifted_lists(list(unique_patient_ids))
        unique_patient_ids = altered_lists[k_fold]
        print(f"Using unique patient IDs for k-fold {k_fold}: {unique_patient_ids}")

    dataset_loader = DatasetLoader(dataset_type=dataset_type, unique_patient_ids=unique_patient_ids,
                                   k_fold=k_fold, use_dist_loss=use_dist_loss, dist_param=dist_param, neg_partition=neg_partition,
                                   use_similar_negatives=use_similar_negatives, neg_pos_ratio=neg_pos_ratio,
                                   filter_num_of_patients=filter_num_of_patients, filter_num_of_healthy=filter_num_of_healthy,
                                   ratio=ratio, filter_to_inflate=filter_to_inflate,
                                   remove_seqs_by_len=remove_seqs_by_len, top_percent=top_percent, top_n_seqs=top_n_seqs, extra_filter=extra_filter,
                                   use_recurrence_loss=use_recurrence_loss, recurrence_gamma=recurrence_gamma, loss_version=loss_version,
                                   run_on_full_data=run_on_full_data, num_test_patients=num_test_patients,
                                   sweep_loader_mode=sweep_loader_mode, fisher_mode=fisher_mode, extract_dataset_info=extract_dataset_info, verbose=verbose)
    return dataset_loader


def sweep_model():
    wandb.init()

    # Define the sweep configuration
    model_type = wandb.config.model_type
    loss_type = wandb.config.loss_type
    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    neg_pos_ratio = wandb.config.neg_pos_ratio
    pos_weights = wandb.config.pos_weights
    learning_rate = wandb.config.learning_rate
    reg_coef = wandb.config.reg_coef
    freeze_embed_model = wandb.config.freeze_embed_model
    special_criterion = wandb.config.special_criterion
    embedding_lr = wandb.config.embedding_lr
    ch_dropout = wandb.config.classification_dropout
    log_wandb = not wandb.config.no_wandb_log
    scheduler_type = wandb.config.scheduler_type.lower()
    cvc_layers_to_train = wandb.config.cvc_layers_to_train
    lora = wandb.config.lora
    masking = wandb.config.masking
    ratio = wandb.config.ratio
    use_dist_loss = wandb.config.use_dist_loss
    dist_param = wandb.config.dist_param if use_dist_loss else 0.0
    neg_partition = wandb.config.negative_partition
    use_similar_negatives = wandb.config.use_similar_negatives
    filter_num_of_patients = wandb.config.dataset_filter_num_of_patients
    filter_num_of_healthy = wandb.config.dataset_filter_num_of_healthy
    filter_to_inflate = not wandb.config.dataset_filter_dont_inflate
    remove_seqs_by_len = wandb.config.remove_seqs_by_len
    top_percent = wandb.config.top_percent
    top_n_seqs = wandb.config.top_n_seqs
    extra_filter = wandb.config.extra_filter
    use_recurrence_loss = wandb.config.use_recurrence_loss
    recurrence_gamma = wandb.config.recurrence_gamma if use_recurrence_loss else 1.0
    loss_version = wandb.config.loss_version
    dataset_type = wandb.config.dataset_type
    extra_ms_from_pregnant = wandb.config.extra_ms_from_pregnant
    plus_healthy_mal_id = wandb.config.plus_healthy_mal_id
    use_healthy_as_ms = wandb.config.use_healthy_as_ms
    ch_type = wandb.config.ch_type
    # changing_negatives = wandb.config.changing_negatives
    changing_negatives = False
    sample_plots = wandb.config.sample_plots
    k_fold = wandb.config.k_fold
    sweep_loader_mode = wandb.config.sweep_loader_mode
    optimizer_type = 'adam'  # TODO: We can add this to sweep config file!
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reshef_negative_part = 0

    if top_percent == 'None':
        top_percent = None
    if top_n_seqs == 'None':
        top_n_seqs = None

    dataset_type = update_dataset_type(
        dataset_type,
        plus_healthy_mal_id=plus_healthy_mal_id,
        extra_ms_from_pregnant=extra_ms_from_pregnant,
        use_healthy_as_ms=use_healthy_as_ms,
        top_percent=top_percent,
        top_n_seqs=top_n_seqs,
        sweep_loader_mode=sweep_loader_mode,
    )

    # Load the dataset
    dataset_loader = get_dataset_loader(dataset_type, k_fold=k_fold, to_k_fold=False,
                                        use_dist_loss=use_dist_loss, dist_param=dist_param, neg_partition=neg_partition,
                                        use_similar_negatives=use_similar_negatives, neg_pos_ratio=neg_pos_ratio,
                                        filter_num_of_patients=filter_num_of_patients, filter_num_of_healthy=filter_num_of_healthy,
                                        ratio=ratio, filter_to_inflate=filter_to_inflate, remove_seqs_by_len=remove_seqs_by_len,
                                        top_percent=top_percent, top_n_seqs=top_n_seqs, extra_filter=extra_filter,
                                        use_recurrence_loss=use_recurrence_loss, recurrence_gamma=recurrence_gamma, loss_version=loss_version,
                                        sweep_loader_mode=sweep_loader_mode, verbose=True)
    df_bld, df_hlt = dataset_loader.get_dfs()
    positive_seqs = dataset_loader.positive_seqs
    train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs = dataset_loader.get_seqs()
    train_patient_inds, valid_patient_inds, test_patient_inds = dataset_loader.get_patient_inds()
    train_masks, valid_masks, test_masks = dataset_loader.get_masks()
    train_inds = dataset_loader.train_inds
    unique_patient_ids = dataset_loader.unique_patient_ids
    patient_id_masks = dataset_loader.patient_id_masks
    aaseq_to_ratio = dataset_loader.get_aaseq_to_ratio_func()
    aaseq_to_dist = dataset_loader.get_aaseq_to_distance_func()
    aaseq_to_recurrence = dataset_loader.get_aaseq_to_recurrence_func()

    # Initialize Weights & Biases
    model = build_model(
        model_type,
        positive_seqs=positive_seqs,
        batch_size=batch_size,
        ch_dropout=ch_dropout,
        cvc_layers_to_train=cvc_layers_to_train,
        freeze_embed_model=freeze_embed_model,
        lora=lora,
        ch_type=ch_type,
        device=device,
        reshef_negative_part=reshef_negative_part,
        args=args,
    )

    # load the model if possible
    trained_model, history = train_model(model, train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs,
                                         epochs=epochs,
                                         lr=learning_rate,
                                         pos_batch_size=batch_size // neg_pos_ratio,
                                         neg_pos_ratio=neg_pos_ratio,
                                         log_wandb=log_wandb,
                                         model_type=model_type,
                                         loss_type=loss_type,
                                         freeze_embed_model=freeze_embed_model,
                                         special_criterion=special_criterion,
                                         embedding_lr=embedding_lr,
                                         reg_coef=reg_coef,
                                         pos_weights=pos_weights,
                                         scheduler_type=scheduler_type,
                                         aaseq_to_ratio=aaseq_to_ratio,
                                         aaseq_to_dist=aaseq_to_dist,
                                         aaseq_to_recurrence=aaseq_to_recurrence,
                                         masking=masking,
                                         ratio=ratio,
                                         change_negatives=changing_negatives,
                                         optimizer_type=optimizer_type,
                                         test_pos_seqs=test_pos_seqs,
                                         test_neg_seqs=test_neg_seqs,
                                         args=args,
                                         )

    from inference.plot_training import plot_output_distributions_per_patient
    plot_output_distributions_per_patient(trained_model, test_patient_inds, valid_patient_inds, unique_patient_ids,
                                          test_masks, valid_masks, positive_seqs, df_bld,
                                          df_hlt, model_type, log_wandb, sample_plots, args, device)


def do_sweep(sweep_version, project_name='TCRep_Sweeps'):
    file_version = '' if sweep_version == 0 else f'_v{sweep_version}'
    sweep_id_path = f'sweep_yaml/sweep_id{file_version}.txt'

    with open(f'sweep_yaml/sweep{file_version}.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)

    # Check if a sweep ID already exists
    if os.path.exists(sweep_id_path):
        with open(sweep_id_path, 'r') as f:
            sweep_id = f.read().strip()
        print(f"Resuming existing sweep: {sweep_id}")
    else:
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        with open(sweep_id_path, 'w') as f:
            f.write(sweep_id)
        print(f"Created new sweep: {sweep_id}")

    wandb.agent(sweep_id, function=sweep_model, count=50, project=project_name, entity='amir-weinfeld')  # Run sweeps one after the other for count runs


def get_arg_parser():
    # get program arguments
    model_types = ['ff', 'cvc', 'esmc', 'cvc_full', 'cvc_weighted', 'cvc_combined_reshef']
    loss_types = ['ce', 'ce_l2', 'ce_entropy']
    scheduler_types = ['None', 'StepLR', 'ReduceLROnPlateau', 'CosineAnnealingLR', 'ExponentialLR']
    dataset_types = ['ms', 'ms_hlt_article', 'ms_plus_hlt_article',
                     'ms_extra', 'ms_extra_hlt_article', 'ms_extra_plus_hlt_article',
                     'ms_no_healthy_ms',
                     'article', 'cmv',
                     'article_sle', 'article_sle_hlt_ms_no_healthy_ms', 'article_sle_plus_hlt_ms_no_healthy_ms',
                     't1d', 't1d_hlt_ms_no_healthy_ms', 't1d_plus_hlt_ms_no_healthy_ms',
                     'ms_tcrdb2', 'ms_tcrdb2_no_healthy_ms', 'ms_tcrdb2_hlt_article',
                     'ms_tcrdb2_no_healthy_ms_plus_hlt_article',
                     'article_hiv', 'article_covid19', 'article_influenza',
                     'jia_tcrdb2']  # ms is TCRdb Multiple Sclerosis, article is Mal-ID Diabetes Type 1, article 2 is TCR MS CSF dataset, CMV is TCRdb CMV.
    ch_types = ['none', 'v1', 'v2']
    optimizer_types = ['Adam', 'Adafactor']

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=model_types, default='cvc', help='Type of model to train')
    parser.add_argument('--loss_type', type=str, choices=loss_types, default='ce', help='Type of loss function to use')
    parser.add_argument('--scheduler_type', type=str, choices=scheduler_types, default='None', help='Type of schedulers to use')
    parser.add_argument('--dataset_type', type=str, choices=dataset_types, default='ms', help='Type of the dataset to run on')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=330, help='Batch size for training')
    parser.add_argument('--neg_pos_ratio', type=int, default=10, help='Negative to positive sample ratio')
    parser.add_argument('--pos_weights', type=float, default=3, help='Positive class weight for loss function')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for optimizer')
    parser.add_argument('--regularization_coefficient', '--reg_coef', type=float, default=0.25, help='Coefficient for the regularization term')
    parser.add_argument('--freeze_embed_model', '-freeze', action='store_true', help='Freeze the embedding model (the classification layers are unfrozen)')
    parser.add_argument('--special_criterion', '-scrit', action='store_true', help='Using a more complex criterion for the model (different lrs)')
    parser.add_argument('--embedding_lr', '--embed_lr', type=float, default=0.00005, help='Learning rate for optimizer of the embedding model only')
    parser.add_argument('--no_wandb_log', '-nolog', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--test_mode_epoch', type=int, default=-1, help='Only Inferencing mode. Loading the model instead of training in the given epoch')
    parser.add_argument('--classification_dropout', '--dropout', type=float, default=0.2, help='Dropout rate for the classification head')
    parser.add_argument('--to_sweep', '--sweep', '-sweep', action='store_true', help='Sweep the hyperparameters using Weights & Biases')
    parser.add_argument('--force_retrain', '-retrain', action='store_true', help='Forces the model to retrain even if a similar model .pth file already exists')
    parser.add_argument('-dont_inference', action='store_true', help='Do not inference')
    parser.add_argument('-dont_plot', action='store_true', help='Do not create plots')
    parser.add_argument('--cvc_layers_to_train', type=int, default=3, help='Number of layers to train in case we use the CVC model')
    parser.add_argument('--k_fold', type=int, default=0, help='K-Fold Index (0 for no k-fold)')
    parser.add_argument('--lora', '-lora', action='store_true', help='Use LoRA')
    parser.add_argument('--sweep_version', type=int, default=0, help='Version of the sweep file to use')
    parser.add_argument('--masking', '-mask', action='store_true', help='Use masking for the model. Only for CVC model')
    parser.add_argument('--ratio', '-ratio', action='store_true', help='Incorporate Ratio into the loss of the model during training')
    parser.add_argument('--ch_type', type=str, choices=ch_types, default='none', help='Type of classification head to use')
    parser.add_argument('--negative_partition', '--neg_partition', type=int, default=0, help='Negative Partition Index (0 for no partitioning of the negative samples)')
    parser.add_argument('--to_ensemble', '--ensemble', '-ensemble', action='store_true', help='Ensemble the models (Only applicable after first training with all 1..5 negative_partitioning)')
    parser.add_argument('--use_similar_negatives', action='store_true', help='Use similar negatives to training positives for training (similar according to Levenstein distance)')
    parser.add_argument('--combine_classification', '-comb_class', action='store_true', help='Combine classification model results (Only applicable after first training with all 1..5 k-folds)')
    parser.add_argument('--dataset_filter_num_of_patients', type=int, default=3, help='Number of patients to filter the dataset by (take positive from this num of patients)')
    parser.add_argument('--dataset_filter_num_of_healthy', type=int, default=-1, help='Number of healthy subjects to filter the dataset with (remove from positives from this num of patients)')
    parser.add_argument('--dataset_filter_dont_inflate', '-no_inflate', action='store_true', help='Do not inflate the dataset when filtering positives and negatives')
    parser.add_argument('--changing_negatives', '-change_neg', action='store_true', help='Whether to run sample the negatives each epoch or not')
    parser.add_argument('--remove_seqs_by_len', type=int, default=False, help='Whether to remove sequences from valid/test sets by length or not')
    parser.add_argument('--train_vae', action='store_true', default=False, help='Train ControlVAE on positive sequences instead of classification model')
    parser.add_argument('--top_percent', type=int, default=None, help='The top percent of sequences take when loading data from TCRdb2')
    parser.add_argument('--top_n_seqs', type=int, default=None, help='The top n*1000 sequences take when loading data from TCRdb2')
    parser.add_argument('--plus_healthy_mal_id', action='store_true', default=False, help='Whether to add healthy Mal-ID sequences to the dataset or not')
    parser.add_argument('--extra_ms_from_pregnant', action='store_true', default=False, help='Whether to add extra MS data of pregnant MS study (only for MS dataset)')
    parser.add_argument('--use_healthy_as_ms', action='store_true', default=False, help='Whether to use some of the healthy patients as MS patients (only for MS dataset)')
    parser.add_argument('--extra_filter', action='store_true', default=False, help='Whether to filter the data more than the default filtering (Remove seqs of certain lengths and remove subjects with not a lot of sequences)')
    parser.add_argument('--use_dist_loss', action='store_true', default=False, help='Type of distribution loss to use')
    parser.add_argument('--dist_param', type=float, default=0.0, help='Hyperparam for distance loss')
    parser.add_argument('--use_recurrence_loss', action='store_true', default=False, help='Add recurrence term into loss calculation')
    parser.add_argument('--recurrence_gamma', type=float, default=1.0, help='Hyperparam for recurrence loss')
    parser.add_argument('--loss_version', type=int, default=0, help='The version of the loss to use')
    parser.add_argument('--sample_plots', type=int, default=0, help='Sampling when plotting instead of running on all sequences')
    parser.add_argument('--classification_v2', action='store_true', default=False, help='Use the new classification model with a different architecture (Version 2)')
    parser.add_argument('--optimizer_type', type=str, choices=optimizer_types, default='Adam', help='Type of optimizer to use')
    parser.add_argument('-reshef_inference', action='store_true', default=False, help='Whether to save information for Reshef inference or not')
    parser.add_argument('-reshef_filter_train', action='store_true', default=False, help='Whether to save information for Reshef inference or not')
    parser.add_argument('-reshef_negative_part', type=int, default=0, help='Part of the negative partition to use for Reshef inference (0 for no partitioning)')
    parser.add_argument('-run_on_full_data', action='store_true', default=False, help='Whether to save information for Reshef inference or not')
    parser.add_argument('-dont_cache_inference', action='store_true', default=False, help='Whether to cache the inference results or not. If True, it will not cache the results and will run inference every time.')
    parser.add_argument('-filter_uncertain_seqs', action='store_true', default=False, help='Whether to filter uncertain sequences in the inference classification v2 part.')
    parser.add_argument('-sweep_loader_mode', action='store_true', default=False, help='Whether to use dataloader in a simpler way that will calculate positives with all patients and split positives to both test/valid 20% randomly. Useful when we do not want to overfit to a single fold.')
    parser.add_argument('-fisher_mode', action='store_true', default=False, help='.') #TODO: ADD!
    parser.add_argument('--extract_dataset_info', action='store_true', default=False, help='Whether to extract information to cache about num of positives of the dataset with the given parameters.')

    return parser


def adjust_args(args):
    """Apply quick fixes and dependencies between arguments."""
    if args.combine_classification and not args.dont_plot:
        args.dont_plot = True
    return args


def validate_config(args):
    """Validate configuration to avoid illegal argument combinations."""
    assert not (args.freeze_embed_model and args.special_criterion), \
        "Cannot use both freeze_embed_model and special_criterion"
    assert not (args.freeze_embed_model and args.model_type != 'cvc'), \
        "Only CVC model can freeze the embedding model"
    assert not (args.to_sweep and not (not args.no_wandb_log)), \
        "Cannot sweep hyperparameters without logging to wandb"
    if args.test_mode_epoch >= 0:
        assert args.no_wandb_log, "Cannot log to wandb in test mode"
    assert not (args.k_fold > 0 and args.to_sweep), \
        "Cannot do k-fold cross-validation and sweep at the same time"
    assert not (args.loss_type == 'ce' and args.dataset_type in ['article', 'article_sle']), \
        "Cannot use ce loss with article or article_sle datasets. Due to Ratio loss"
    assert not ((args.negative_partition > 0) and args.to_sweep), \
        "Cannot use negative partitioning and sweep at the same time"
    assert not ((args.negative_partition > 0) and args.to_ensemble), \
        "Cannot use negative partitioning and ensemble at the same time"
    assert not (args.top_percent is not None and 'tcrdb2' not in args.dataset_type), \
        "Cannot use top_percent when dataset_type does not contain 'tcrdb2'"
    assert not (args.top_percent is not None and args.top_n_seqs is not None), \
        "Cannot use both top_percent and top_n_seqs at the same time"
    assert not (args.extra_ms_from_pregnant and 'ms' not in args.dataset_type), \
        "extra_ms_from_pregnant can only be used with MS dataset of tcrdb2.0"
    assert not (args.use_healthy_as_ms and 'ms' not in args.dataset_type), \
        "use_healthy_as_ms can only be used with MS dataset of tcrdb2.0"
    assert not (args.reshef_inference and args.changing_negatives), \
        "Reshef inference is not applicable if changing negatives"
    assert not (args.reshef_filter_train and not args.reshef_inference), \
        "Reshef filter train must be on when reshef inference is on"
    assert not (args.sweep_loader_mode and args.k_fold > 0), \
        "sweep_loader_mode can be true only when k_fold is 0"
    assert not (args.extract_dataset_info and (args.k_fold > 0 or not args.no_wandb_log)), \
        "extract_dataset_info can be true only when k_fold is 0 and we are not logging to wandb"


def update_dataset_type(dataset_type: str,
                        plus_healthy_mal_id: bool = False,
                        extra_ms_from_pregnant: bool = False,
                        use_healthy_as_ms: bool = False,
                        top_percent: int | None = None,
                        top_n_seqs: int | None = None,
                        run_on_full_data: bool = False,
                        sweep_loader_mode: bool = False,
                        fisher_mode: bool = False) -> str:
    """
    Modify dataset_type string based on options.
    Returns a new dataset_type string (does not mutate inputs).
    """
    if plus_healthy_mal_id:
        dataset_type += "_plus_hlt_article"
    if extra_ms_from_pregnant:
        dataset_type += "_extra_ms"
    if use_healthy_as_ms:
        dataset_type += "_hlt_as_ms"
    if top_percent is not None:
        dataset_type += f"_top_{top_percent}"
    if top_n_seqs is not None:
        dataset_type += f"_top_{top_n_seqs}k"
    if run_on_full_data:
        dataset_type += "_run_on_full_data"
    if sweep_loader_mode:
        dataset_type += "_sweep_loader_mode"
    if fisher_mode:
        dataset_type += "_fisher_mode"
    return dataset_type


def print_run_configuration(args):
    """Pretty-print the run configuration."""
    print("RUN CONFIGURATION:")
    print(f"\tModel Type: {args.model_type}")
    print(f"\tLoss Type: {args.loss_type}")
    print(f"\tDataset Type: {args.dataset_type}")
    print(f"\tOptimizer Type: {args.optimizer_type}")
    print(f"\tEpochs: {args.epochs}")
    print(f"\tBatch Size: {args.batch_size}")
    print(f"\tNegative to Positive Ratio: {args.neg_pos_ratio}")
    print(f"\tPositive Class Weight: {args.pos_weights}")
    print(f"\tLearning Rate: {args.learning_rate}")
    print(f"\tRegularization Coefficient: {args.regularization_coefficient}")
    print(f"\tFreeze Embedding Model: {args.freeze_embed_model}")
    print(f"\tSpecial Criterion: {args.special_criterion}")
    print(f"\tEmbedding Learning Rate: {args.embedding_lr}")
    print(f"\tClassification Head Dropout: {args.classification_dropout}")
    print(f"\tScheduler Type: {args.scheduler_type}")
    print(f"\tForce Retrain: {args.force_retrain}")
    print(f"\tCVC model layers to train: {args.cvc_layers_to_train}")
    print(f"\tDo K-Fold Cross-Validation: {args.k_fold}")
    print(f"\tNegative Partition Index: {args.negative_partition}")
    print(f"\tUse LoRA: {args.lora}")
    print(f"\tMasking: {args.masking}")
    print(f"\tUsing Ratio: {args.ratio}")
    print(f"\tClassification Head Type: {args.ch_type}")
    print(f"\tDataset Filter Number of Patients: {args.dataset_filter_num_of_patients}")
    print(f"\tDataset Filter Number of Healthy: {args.dataset_filter_num_of_healthy}")
    print(f"\tDataset Filter Inflate: {not args.dataset_filter_dont_inflate}")
    print(f"\tChanging Negatives: {args.changing_negatives}")
    print(f"\tLoss Version: {args.loss_version}")
    print(f"\tUse Distance Loss: {args.use_dist_loss}")
    print(f"\tDistance Loss Parameter: {args.dist_param}")
    print(f"\tUse Recurrence Loss: {args.use_recurrence_loss}")
    print(f"\tRecurrence Gamma: {args.recurrence_gamma}")
    print(f"\tSample Plots: {args.sample_plots}")
    print(f"\tRun on Full Data: {args.run_on_full_data}")
    print("\tDevice:", "cuda" if torch.cuda.is_available() else "cpu")
    print("\n")


if __name__ == '__main__':
    # Weights & Biases setup
    parser = get_arg_parser()
    args = parser.parse_args()

    # Sweep
    to_sweep = args.to_sweep
    sweep_version = args.sweep_version
    if to_sweep:
        do_sweep(sweep_version)
        print("Completed sweeping. Exiting.")
        exit(0)

    # Parse arguments for run
    model_type = args.model_type.lower()
    loss_type = args.loss_type.lower()
    dataset_type = args.dataset_type.lower()
    epochs = args.epochs
    batch_size = args.batch_size
    neg_pos_ratio = args.neg_pos_ratio
    pos_weights = args.pos_weights
    learning_rate = args.learning_rate
    reg_coef = args.regularization_coefficient if loss_type != 'ce' else 0  # Regularization only for 'ce_l2' and 'ce_entropy'
    freeze_embed_model = args.freeze_embed_model if 'cvc' in model_type else False  # Only CVC model can freeze the embedding model
    special_criterion = args.special_criterion
    embedding_lr = args.embedding_lr if special_criterion else 0  # Only used when special_criterion is True
    log_wandb = not args.no_wandb_log
    test_mode_epoch = args.test_mode_epoch
    ch_dropout = args.classification_dropout
    scheduler_type = args.scheduler_type.lower()
    force_retrain = args.force_retrain
    dont_inference = args.dont_inference
    dont_plot = args.dont_plot
    cvc_layers_to_train = args.cvc_layers_to_train if not freeze_embed_model else 0  # No layers to train if embedding model is frozen
    k_fold = args.k_fold if args.k_fold >= 0 else 0  # Set to 0 if negative
    to_k_fold = k_fold > 0
    lora = args.lora if 'cvc' in model_type else False  # LoRA is only applicable for CVC model
    masking = args.masking if 'cvc' in model_type else False  # Masking is only applicable for CVC model
    ratio = args.ratio
    use_dist_loss = args.use_dist_loss
    dist_param = args.dist_param if use_dist_loss else 0.0
    ch_type = args.ch_type.lower() if 'cvc' in model_type else 'none'  # Only CVC model can use dist loss
    neg_partition = args.negative_partition
    to_ensemble = args.to_ensemble
    use_similar_negatives = args.use_similar_negatives
    combine_classification = args.combine_classification
    filter_num_of_patients = args.dataset_filter_num_of_patients
    filter_num_of_healthy = args.dataset_filter_num_of_healthy if args.dataset_filter_num_of_healthy >= 0 else filter_num_of_patients
    filter_to_inflate = not args.dataset_filter_dont_inflate
    changing_negatives = args.changing_negatives
    remove_seqs_by_len = args.remove_seqs_by_len
    top_percent = args.top_percent
    top_n_seqs = args.top_n_seqs
    plus_healthy_mal_id = args.plus_healthy_mal_id
    extra_ms_from_pregnant = args.extra_ms_from_pregnant
    use_healthy_as_ms = args.use_healthy_as_ms
    extra_filter = args.extra_filter
    use_recurrence_loss = args.use_recurrence_loss
    recurrence_gamma = args.recurrence_gamma if use_recurrence_loss else 1.0
    loss_version = args.loss_version
    sample_plots = args.sample_plots
    classification_v2 = args.classification_v2
    optimizer_type = args.optimizer_type.lower()
    reshef_inference = args.reshef_inference
    reshef_filter_train = args.reshef_filter_train
    reshef_negative_part = args.reshef_negative_part if reshef_inference else 0
    run_on_full_data = args.run_on_full_data
    dont_cache_inference = args.dont_cache_inference
    filter_uncertain_seqs = args.filter_uncertain_seqs if classification_v2 else False
    sweep_loader_mode = args.sweep_loader_mode
    fisher_mode = args.fisher_mode
    extract_dataset_info = args.extract_dataset_info

    # Adjust and validate arguments
    args = adjust_args(args)
    validate_config(args)
    dataset_type = update_dataset_type(
        dataset_type,
        plus_healthy_mal_id=args.plus_healthy_mal_id,
        extra_ms_from_pregnant=args.extra_ms_from_pregnant,
        use_healthy_as_ms=args.use_healthy_as_ms,
        top_percent=args.top_percent,
        top_n_seqs=args.top_n_seqs,
        run_on_full_data=args.run_on_full_data,
        sweep_loader_mode=args.sweep_loader_mode,
        fisher_mode=fisher_mode
    )
    print_run_configuration(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the dataset
    dataset_loader = get_dataset_loader(dataset_type, k_fold=k_fold, to_k_fold=to_k_fold,
                                        use_dist_loss=use_dist_loss, dist_param=dist_param, neg_partition=neg_partition,
                                        use_similar_negatives=use_similar_negatives, neg_pos_ratio=neg_pos_ratio,
                                        filter_num_of_patients=filter_num_of_patients, filter_num_of_healthy=filter_num_of_healthy,
                                        ratio=ratio, filter_to_inflate=filter_to_inflate, remove_seqs_by_len=remove_seqs_by_len,
                                        top_percent=top_percent, top_n_seqs=top_n_seqs, extra_filter=extra_filter,
                                        use_recurrence_loss=use_recurrence_loss, recurrence_gamma=recurrence_gamma,
                                        loss_version=loss_version, run_on_full_data=run_on_full_data, fisher_mode=fisher_mode,
                                        extract_dataset_info=extract_dataset_info, verbose=True)
    df_bld, df_hlt = dataset_loader.get_dfs()
    positive_seqs = dataset_loader.positive_seqs
    train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs = dataset_loader.get_seqs()
    train_patient_ids, valid_patient_ids, test_patient_ids = dataset_loader.get_patient_ids()
    train_patient_inds, valid_patient_inds, test_patient_inds = dataset_loader.get_patient_inds()
    train_masks, valid_masks, test_masks = dataset_loader.get_masks()
    train_inds = dataset_loader.train_inds
    unique_patient_ids = dataset_loader.unique_patient_ids
    patient_id_masks = dataset_loader.patient_id_masks
    aaseq_to_ratio = dataset_loader.get_aaseq_to_ratio_func()
    aaseq_to_dist = dataset_loader.get_aaseq_to_distance_func()
    aaseq_to_recurrence = dataset_loader.get_aaseq_to_recurrence_func()

    # TODO: TEST CODE. REMOVE LATER!
    temp_tests = False
    if temp_tests:
        df_bld_train = df_bld[df_bld['patient_id'].isin(train_patient_ids)]
        df_bld_valid = df_bld[df_bld['patient_id'].isin(np.concatenate([valid_patient_ids, test_patient_ids]))]
        # df_bld_test = df_bld[df_bld['patient_id'].isin(test_patient_ids)]

        # use seed to split healthy to train/valid/test (20% each for valid/test)
        rng = np.random.default_rng(seed=42)
        hlt_patient_ids = sorted(df_hlt['patient_id'].unique())
        rng.shuffle(hlt_patient_ids)
        num_hlt = len(hlt_patient_ids)
        num_hlt_valid = num_hlt_test = num_hlt // 5
        hlt_valid_ids = set(hlt_patient_ids[:num_hlt_valid])
        hlt_test_ids = set(hlt_patient_ids[num_hlt_valid:2 * num_hlt_valid])
        hlt_train_ids = set(hlt_patient_ids[2 * num_hlt_valid:])
        df_hlt_train = df_hlt[df_hlt['patient_id'].isin(hlt_train_ids)]
        df_hlt_valid = df_hlt[df_hlt['patient_id'].isin(hlt_valid_ids)]
        # df_hlt_test = df_hlt[df_hlt['patient_id'].isin(hlt_test_ids)]

        def build_patient_seq_table(df, label):
            """
            Build patient-level table:
            For each AASeq, count how many unique patients have it.
            """
            seq_patient_counts = (
                df.groupby("AASeq")["patient_id"]
                .nunique()
                .reset_index(name=f"{label}_patients")
            )
            return seq_patient_counts
        # Disease and healthy patient-level sequence counts
        disease_counts = build_patient_seq_table(df_bld_train, "disease")
        healthy_counts = build_patient_seq_table(df_hlt_train, "healthy")

        # Merge into one frequency table
        freq_table = pd.merge(disease_counts, healthy_counts, on="AASeq", how="outer").fillna(0)
        total_disease = df_bld_train['patient_id'].nunique()
        total_healthy = df_hlt_train['patient_id'].nunique()

        from scipy.stats import fisher_exact
        def fisher_pval(row):
            table = [
                [row['disease_patients'], total_disease - row['disease_patients']],
                [row['healthy_patients'], total_healthy - row['healthy_patients']]
            ]
            _, pval = fisher_exact(table, alternative="greater")  # enrichment in disease
            return pval

        freq_table["pval"] = freq_table.apply(fisher_pval, axis=1)
        freq_table = freq_table.sort_values("pval")

        # disease_panel = set(freq_table.query("pval < 0.05")["AASeq"].tolist())
        disease_panel = set(freq_table[(freq_table["healthy_patients"] < 3) & (freq_table["pval"] < 0.1)]["AASeq"].tolist())

        def score_patient(df_patient, disease_panel):
            seqs = set(df_patient["AASeq"].tolist())
            overlap = seqs.intersection(disease_panel)
            return len(overlap), len(overlap) / len(seqs)  # raw count and fraction

        # Score all patients in validation
        val_scores = []
        for pid, group in df_bld_valid.groupby("patient_id"):
            score = score_patient(group, disease_panel)
            val_scores.append({"patient_id": pid, "label": "disease", "score": score[1]})

        for pid, group in df_hlt_valid.groupby("patient_id"):
            score = score_patient(group, disease_panel)
            val_scores.append({"patient_id": pid, "label": "healthy", "score": score[1]})

        val_df = pd.DataFrame(val_scores)

        from sklearn.metrics import roc_auc_score

        y_true = (val_df["label"] == "disease").astype(int)
        y_scores = val_df["score"]

        auc = roc_auc_score(y_true, y_scores)
        print("Validation AUC:", auc)

    # Run VAE if specified
    # run_vae(args, dataset_loader, device)

    # Plot dataset information
    plot_all_dataset(dataset_loader, df_bld, df_hlt, dataset_type, train_patient_ids,
                     positive_seqs, neg_seqs, aaseq_to_ratio, to_display_dict)

    # wandb init
    if log_wandb:
        run = wand_init(
            model_type=model_type,
            loss_type=loss_type,
            dataset_type=dataset_type,
            epochs=epochs,
            batch_size=batch_size,
            neg_pos_ratio=neg_pos_ratio,
            pos_weights=pos_weights,
            learning_rate=learning_rate,
            reg_coef=reg_coef,
            freeze_embed_model=freeze_embed_model,
            special_criterion=special_criterion,
            embedding_lr=embedding_lr,
            ch_dropout=ch_dropout,
            scheduler_type=scheduler_type,
            cvc_layers_to_train=cvc_layers_to_train,
            k_fold=k_fold,
            lora=lora,
            masking=masking,
            ratio=ratio,
            use_dist_loss=use_dist_loss,
            dist_param=dist_param,
            use_recurrence_loss=use_recurrence_loss,
            recurrence_gamma=recurrence_gamma,
            ch_type=ch_type,
            neg_partition=neg_partition,
            use_similar_negatives=use_similar_negatives,
            filter_num_of_patients=filter_num_of_patients,
            filter_num_of_healthy=filter_num_of_healthy,
            filter_to_inflate=filter_to_inflate,
            change_negatives=changing_negatives,
            sample_plots=sample_plots,
            optimizer_type=optimizer_type,
            device=device,
        )

    # build the model
    model = build_model(
        model_type,
        positive_seqs=positive_seqs,
        batch_size=batch_size,
        ch_dropout=ch_dropout,
        cvc_layers_to_train=cvc_layers_to_train,
        freeze_embed_model=freeze_embed_model,
        lora=lora,
        ch_type=ch_type,
        device=device,
        reshef_negative_part=reshef_negative_part,
        args=args,
    )

    # load the model if possible
    trained_model = load_or_train_model(
        model, args,
        train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs,
        test_pos_seqs, test_neg_seqs,
        aaseq_to_ratio, aaseq_to_dist, aaseq_to_recurrence,
        device,
        epochs, learning_rate, batch_size, neg_pos_ratio,
        log_wandb, model_type, loss_type,
        freeze_embed_model, special_criterion,
        embedding_lr, reg_coef, pos_weights,
        scheduler_type, masking, ratio,
        changing_negatives, optimizer_type,
        reshef_inference, reshef_filter_train, reshef_negative_part
    )

    # TODO: Moving here for tests!
    # Run VAE if specified
    model_for_vae = build_model(model_type, positive_seqs=positive_seqs, batch_size=batch_size, ch_dropout=ch_dropout,
                                cvc_layers_to_train=cvc_layers_to_train, freeze_embed_model=freeze_embed_model, lora=lora,
                                ch_type=ch_type, device=device, reshef_negative_part=reshef_negative_part, args=args)
    run_vae(args, dataset_loader, device, trained_model, model_for_vae)

    # after training
    plot_all_training(
        trained_model, args,
        positive_seqs, neg_seqs, valid_neg_seqs, test_neg_seqs,
        test_patient_inds, valid_patient_inds, unique_patient_ids,
        test_masks, valid_masks, df_bld, df_hlt,
        sample_plots, device
    )

    # after training & plotting:
    run_inference(
        trained_model, args,
        train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs,
        test_pos_seqs, test_neg_seqs, df_bld, df_hlt, positive_seqs,
        test_patient_inds, valid_patient_inds, test_patient_ids, valid_patient_ids, unique_patient_ids,
        test_masks, valid_masks, aaseq_to_ratio, aaseq_to_dist,
        batch_size, ch_dropout, cvc_layers_to_train,
        freeze_embed_model, lora, ch_type,
        pos_weights, reg_coef, ratio, inference_dict,
        device, get_dataset_loader,
        classification_v2=classification_v2,
        dont_cache_inference=dont_cache_inference,
        filter_uncertain_seqs=filter_uncertain_seqs,
    )
