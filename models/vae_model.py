import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm


class SingleSeqProcessing:
    def __init__(self, MSA_location, theta=0.2, use_weights=False, weights_location=None, alphabet="ACDEFGHIKLMNPQRSTVWY"):
        self.MSA_location = MSA_location
        self.theta = theta
        self.use_weights = use_weights
        self.weights_location = weights_location
        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.Neff = 1

        from other_models.eve.utils import data_utils
        data = data_utils.MSA_processing(
            MSA_location=MSA_location,
            theta=theta,
            use_weights=True,
            weights_location=weights_location
        )
        self.one_hot_encoding_msa = data.one_hot_encoding
        self.weights_msa = data.weights
        self.seq_len_msa = data.seq_len

        # load my sequences this time
        sequences = self.load_valid_sequences()[::50]
        self.one_hot_encoding = self.one_hot_encode(sequences)
        self.weights = np.ones(len(sequences), dtype=np.float64)
        self.seq_len = self.one_hot_encoding.shape[1]
        self.num_sequences = self.one_hot_encoding.shape[0]

        assert self.one_hot_encoding.shape[0] == self.weights.shape[0], "One-hot encoding shape mismatch WITH WEIGHTS!"
        assert self.one_hot_encoding.shape[0] == self.num_sequences, "One-hot encoding shape mismatch WITH NUM SEQUENCES!"
        assert self.one_hot_encoding.shape[1] == self.seq_len, "One-hot encoding shape mismatch WITH SEQ LEN!"
        assert self.one_hot_encoding.shape[2] == self.alphabet_size, "One-hot encoding shape mismatch WITH ALPHABET SIZE!"

    @staticmethod
    def one_hot_encode(sequences, alphabet="ACDEFGHIKLMNPQRSTVWY", max_length=None):
        """
        One-hot encode an array of protein sequences.

        Args:
            sequences (numpy.ndarray): Array of string sequences
            alphabet (str): Alphabet to use for one-hot encoding, default is 20 standard amino acids
            max_length (int, optional): Maximum sequence length. If None, uses the length of the longest sequence

        Returns:
            torch.Tensor: One-hot encoded sequences with shape (n_sequences, max_length, len(alphabet))
        """
        # Create mapping from amino acid to position
        aa_to_idx = {aa: idx for idx, aa in enumerate(alphabet)}

        # Determine max_length if not provided
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)

        n_sequences = len(sequences)
        alphabet_size = len(alphabet)

        # Pre-allocate tensor
        one_hot = torch.zeros((n_sequences, max_length, alphabet_size), dtype=torch.float32)

        # Process each sequence
        for i, seq in tqdm(enumerate(sequences), total=n_sequences, desc="One-hot encoding"):
            seq_len = len(seq)

            # Calculate padding for centering
            pad_left = (max_length - seq_len) // 2

            # For each position in the sequence
            for j, aa in enumerate(seq):
                if aa in aa_to_idx:  # Handle case where amino acid is not in alphabet
                    # Calculate the position in the padded sequence
                    pos = pad_left + j

                    # Only set value if position is within bounds
                    if 0 <= pos < max_length:
                        one_hot[i, pos, aa_to_idx[aa]] = 1.0

        return one_hot

    @staticmethod
    def load_valid_sequences():
        # TODO: This data extraction should be done in a separate function (which will unify the data extraction process)
        raise NotImplementedError("This function should be implemented in a separate function!")
        df = get_all_usable_disease_data(disease='Multiple sclerosis')
        df_h = get_all_usable_healthy_data()
        valid_seqs_disease = calculate_valid_near_sequences(df, save_name='disease', lev_dist_accept=1, num_of_patients=3)
        # valid_seqs_healthy = calculate_valid_near_sequences(df_h, save_name='healthy', lev_dist_accept=1, num_of_patients=3)
        valid_seqs_healthy = find_all_common_sequences(df_h, num_of_patients=2)  # TODO: Added for now to speed up the process!

        positive_seqs = set([x[0] for x in valid_seqs_disease])
        negative_seqs = set([x[0] for x in valid_seqs_healthy])
        # add negative and positives to a single list
        all_seqs = list(positive_seqs) + list(negative_seqs)
        # sequences_d = df['AASeq'].unique()  # [::35]  # TODO: Removing some sequences for faster embedding
        # sequences_h = df_h['AASeq'].unique()
        # sequences = np.unique(np.concatenate([sequences_d, sequences_h]))
        return np.array(all_seqs)


def train_vae_eve_model():
    import json
    from other_models.eve.EVE.VAE_model import VAE_model

    # Define the base path
    base_path = r'/cs/labs/dina/amir_2000/TCRep/other_models/eve'

    # Default values
    MSA_data_folder = base_path + '/data/MSA'
    MSA_list = base_path + '/data/mappings/example_mapping.csv'
    protein_index = 0
    MSA_weights_location = base_path + '/data/weights'
    theta_reweighting = None  # Default: None
    VAE_checkpoint_location = base_path + '/results/VAE_parameters'
    model_name_suffix = 'Jan1_PTEN_example'
    model_parameters_location = base_path + '/EVE/default_model_params.json'
    training_logs_location = base_path + '/logs'
    seed = 42

    # Load mapping file and extract protein data
    mapping_file = pd.read_csv(MSA_list)
    protein_name = mapping_file['protein_name'][protein_index]
    msa_location = MSA_data_folder + os.sep + mapping_file['msa_location'][protein_index]
    print("Protein name: " + str(protein_name))
    print("MSA file: " + str(msa_location))

    # Determine theta value (if not provided, default to 0.2)
    if theta_reweighting is not None:
        theta = theta_reweighting
    else:
        try:
            theta = float(mapping_file['theta'][protein_index])
        except:
            theta = 0.2
    print("Theta MSA re-weighting: " + str(theta))

    # Process MSA data
    # data = data_utils.MSA_processing(
    #     MSA_location=msa_location,
    #     theta=theta,
    #     use_weights=True,
    #     weights_location=MSA_weights_location + os.sep + protein_name + '_theta_' + str(theta) + '.npy'
    # )
    data = SingleSeqProcessing(
        MSA_location=msa_location,
        theta=theta,
        use_weights=True,
        weights_location=MSA_weights_location + os.sep + protein_name + '_theta_' + str(theta) + '.npy'
    )

    # Construct model name
    model_name = protein_name + "_" + model_name_suffix
    print("Model name: " + str(model_name))

    # Load model parameters
    model_params = json.load(open(model_parameters_location))

    # Initialize model
    model = VAE_model(
        model_name=model_name,
        data=data,
        encoder_parameters=model_params["encoder_parameters"],
        decoder_parameters=model_params["decoder_parameters"],
        random_seed=seed
    )
    model = model.to(model.device)

    # Update training parameters with checkpoint and log locations
    model_params["training_parameters"]['training_logs_location'] = training_logs_location
    model_params["training_parameters"]['model_checkpoint_location'] = VAE_checkpoint_location

    # TODO: Added the next params myself
    model_params["training_parameters"]['log_training_info'] = True
    model_params["training_parameters"]['num_training_steps'] = 40000

    # Train the model
    print("Starting to train model: " + model_name)
    model.train_model(data=data, training_parameters=model_params["training_parameters"])

    # Save the model
    print("Saving model: " + model_name)
    model.save(
        model_checkpoint=model_params["training_parameters"][
                             'model_checkpoint_location'] + os.sep + model_name + "_final",
        encoder_parameters=model_params["encoder_parameters"],
        decoder_parameters=model_params["decoder_parameters"],
        training_parameters=model_params["training_parameters"]
    )