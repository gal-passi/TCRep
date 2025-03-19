import torch
import os
from tqdm import tqdm


def print_trainable_parameters(model):
    """
  printing the number of trainable paramters in the model
  """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


class MockConfig:
    def __init__(self, model_type="esm", tie_word_embeddings=False):
        self.model_type = model_type

    def get(self, key, default=None):
        return getattr(self, key, default)


class CustomModelWrapperesm(torch.nn.Module):
    def __init__(self, esmc_model):
        super(CustomModelWrapperesm, self).__init__()
        self.esmc_model = esmc_model
        self.config = MockConfig(model_type="esmc")

    def forward(self, sequences, **kwargs):
        return self.esmc_model(sequences).embeddings

        # sequences = [ESMProtein(sequence=seq).sequence for seq in input]
        # max_len = max(len(seq) for seq in sequences)
        # masks = torch.zeros((len(sequences), max_len))
        # for i, seq in enumerate(sequences):
        #     masks[i, :len(seq)] = 1
        input_ids = self.esmc_model._tokenize(sequences)#, attention_mask=masks)
        output = self.esmc_model(input_ids).embeddings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self.esmc_model._detokenize(input_ids)
        # self.esmc_model._detokenize(sequences)
        return output

    def prepare_inputs_for_generation(self, inputs, **kwargs):
        # Dummy implementation, does nothing
        return inputs

    def __getattr__(self, name):
        # Ensure attributes in this wrapper class are handled first
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Delegate to the wrapped model
            if hasattr(self.esmc_model, name):
                return getattr(self.esmc_model, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def fine_tune_esmc_tanya():
    from esm.models.esmc import ESMC
    from peft import LoraConfig, get_peft_model, TaskType

    device = "cuda" if torch.cuda.is_available() else "cpu"

    peft_config_esmc = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias='none',
        layers_to_transform=[29, 28, 27, 26, 25, 24, 23, 22, 21],
        task_type=TaskType.CAUSAL_LM,
        target_modules=['attn.out_proj']
    )

    clientc = ESMC.from_pretrained("esmc_300m")  # .to("cuda") # or "cpu"
    # for name, module in clientc.named_modules():
    #     print(name)
    #     for sub in module.named_modules():
    #         print(sub)
    # print(inspect.signature(clientc._tokenize))
    clientc = CustomModelWrapperesm(clientc)
    clientc.config = MockConfig(model_type="esmc")

    peft_model = get_peft_model(clientc, peft_config_esmc)
    # print_trainable_parameters(clientc)

    # print(clientc.config.model_type)
    # if torch.cuda.is_available():
    #     clientc = clientc.to("cuda")
    # out = clientc(["AAAAA", "GG"])  # , "." * 10])
    peft_model = peft_model.to(device)
    # out = peft_model(["AAAAA", "GG"])
    out = peft_model(sequences=["AAAAA", "GG"])

    # save the embeddings
    # torch.save(out, "ESMC_retrained.pt")
    print(f"Output shape: {out.shape}")


def fine_tune_esmc(sequences, batch_size=8, num_epochs=5, learning_rate=5e-5, checkpoint_dir="checkpoints",
                   save_every=1, resume_from=None):
    from esm.models.esmc import ESMC
    from peft import LoraConfig, get_peft_model, TaskType
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import glob
    import re

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Automatically find the most recent checkpoint if resume_from is None
    if resume_from is None:
        # Look for checkpoint files in the checkpoint directory
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))

        if checkpoint_files:
            # Extract epoch numbers from checkpoint filenames
            epoch_numbers = []
            for file_path in checkpoint_files:
                match = re.search(r'checkpoint_epoch_(\d+)\.pt', file_path)
                if match:
                    epoch_numbers.append((int(match.group(1)), file_path))

            # Find the most recent checkpoint
            if epoch_numbers:
                _, resume_from = max(epoch_numbers, key=lambda x: x[0])
                print(f"Auto-resuming from most recent checkpoint: {resume_from}")

        # Also check for final_model.pt
        final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
        if os.path.exists(final_model_path) and (not checkpoint_files or
                                                 os.path.getmtime(final_model_path) > os.path.getmtime(resume_from)):
            resume_from = final_model_path
            print(f"Auto-resuming from final model checkpoint: {resume_from}")

    # Define PEFT configuration
    peft_config_esmc = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        layers_to_transform=[29, 28, 27, 26, 25, 24, 23, 22, 21],
        task_type=TaskType.CAUSAL_LM,
        # target_modules=['attn.out_proj'],
        target_modules=['attn.out_proj', 'attn.layernorm_qkv.1', 'fnn.1', 'fnn.3', 'sequence_head.0', 'sequence_head.3'],  # My Change
    )

    # Load pre-trained model
    clientc = ESMC.from_pretrained("esmc_300m")
    clientc = CustomModelWrapperesm(clientc)
    peft_model = get_peft_model(clientc, peft_config_esmc)
    clientc.config = MockConfig(model_type="esmc")

    # Apply PEFT
    peft_model = get_peft_model(clientc, peft_config_esmc)
    peft_model = peft_model.to(device)

    # Create dataset class
    class ProteinDataset(Dataset):
        def __init__(self, sequences):
            self.sequences = sequences

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return self.sequences[idx]

    # Masking function similar to ESM approach
    def mask_tokens(sequence, mask_prob=0.15):
        # Tokenize sequence using the model's tokenizer
        tokens = peft_model._tokenize(sequence)
        tokens = tokens.to(device)

        # Create a clone to use as labels
        labels = tokens.clone()

        # Determine which tokens to mask
        probability_matrix = torch.full(tokens.shape, mask_prob)

        # Don't mask special tokens like BOS, EOS, etc.
        special_tokens_mask = [
            peft_model.tokenizer.convert_tokens_to_ids("<cls>"),
            peft_model.tokenizer.convert_tokens_to_ids("<eos>"),
            peft_model.tokenizer.convert_tokens_to_ids("<pad>")
        ]
        for special_token in special_tokens_mask:
            probability_matrix[tokens == special_token] = 0.0

        # Select tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Create input by replacing masked tokens
        inputs = tokens.clone()

        # In ESM-style masking, we:
        # - 80% of the time, replace with <mask>
        # - 10% of the time, replace with random amino acid
        # - 10% of the time, keep the original token

        # Get the mask token ID
        mask_token_id = peft_model.tokenizer.convert_tokens_to_ids("<mask>")

        # Indices to replace with mask token (80% of masked tokens)
        indices_mask = torch.bernoulli(torch.full(masked_indices.shape, 0.8)).bool() & masked_indices
        inputs[indices_mask] = mask_token_id

        # Indices to replace with random token (10% of masked tokens)
        indices_random = torch.bernoulli(torch.full(masked_indices.shape, 0.5)).bool() & masked_indices & ~indices_mask
        random_amino_acids = torch.randint(5, 25, indices_random.sum().shape,
                                           device=device)  # Amino acid tokens typically in this range
        inputs[indices_random] = random_amino_acids

        # The remaining 10% masked tokens are kept unchanged

        # For computing loss, we only consider masked tokens
        # Set non-masked tokens to -100 (ignored by CrossEntropyLoss)
        labels[~masked_indices] = -100

        return inputs, labels

    # Create dataset and dataloader
    dataset = ProteinDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(peft_model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Variables to track best model and training state
    best_loss = float('inf')
    start_epoch = 0
    training_history = []

    # Load from checkpoint if specified or auto-detected
    if resume_from is not None and os.path.exists(resume_from):
        print(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        peft_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        training_history = checkpoint.get('training_history', [])

        print(f"Resuming from epoch {start_epoch} with best loss: {best_loss:.4f}")
    else:
        if resume_from is not None:
            print(f"Checkpoint file {resume_from} not found. Starting from scratch.")
        else:
            print("No checkpoints found. Starting from scratch.")

    # Check if we've already completed all epochs
    if start_epoch >= num_epochs:
        print(f"Training already completed (start_epoch={start_epoch}, num_epochs={num_epochs})")
        # Load best model if available
        best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            peft_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model with loss: {checkpoint['loss']:.4f}")

        # Get embeddings and return
        peft_model.eval()
        with torch.no_grad():
            tokenized_seqs = peft_model.esmc_model._tokenize(sequences[:10])
            embeddings = peft_model(sequences=tokenized_seqs)

        return {
            "model": peft_model,
            "embeddings": embeddings,
            "training_history": training_history,
            "best_loss": best_loss,
            "checkpoint_dir": checkpoint_dir,
            "status": "already_completed"
        }

    # Training loop
    peft_model.train()
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            # Process batch of sequences
            batch_inputs = []
            batch_labels = []

            for seq in batch:
                inputs, labels = mask_tokens(seq)
                batch_inputs.append(inputs)
                batch_labels.append(labels)

            # Pad sequences to same length in batch
            max_len = max(len(inp) for inp in batch_inputs)
            pad_token_id = peft_model.tokenizer.convert_tokens_to_ids("<pad>")

            # Pad inputs and labels
            padded_inputs = torch.ones((len(batch), max_len), dtype=torch.long, device=device) * pad_token_id
            padded_labels = torch.ones((len(batch), max_len), dtype=torch.long, device=device) * -100

            for i, (inp, lab) in enumerate(zip(batch_inputs, batch_labels)):
                padded_inputs[i, :len(inp)] = inp[:, 1]  # TODO: Added the slicing part on my own, idk if its correct
                padded_labels[i, :len(lab)] = lab[:, 1]

            # Forward pass
            optimizer.zero_grad()
            outputs = peft_model.forward(sequences=padded_inputs)

            # Reshape for loss calculation
            logits = outputs.view(-1, outputs.size(-1))  # (batch_size * seq_len, vocab_size)
            labels_view = padded_labels.view(-1)  # (batch_size * seq_len)

            # Calculate loss
            loss = criterion(logits, labels_view)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": epoch_loss / (progress_bar.n + 1)})

        # Calculate average loss for this epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

        # Update learning rate based on validation loss
        scheduler.step(avg_epoch_loss)

        # Track training history
        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_epoch_loss,
            'lr': optimizer.param_groups[0]['lr']
        })

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': peft_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
                'best_loss': best_loss,
                'training_history': training_history
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': peft_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
                'best_loss': best_loss,
                'training_history': training_history
            }, best_model_path)
            print(f"New best model saved with loss: {best_loss:.4f}")

    # Save the final model
    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save({
        'epoch': num_epochs - 1,  # -1 since epoch is 0-indexed
        'model_state_dict': peft_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_epoch_loss,
        'best_loss': best_loss,
        'training_history': training_history
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Save the model in PEFT format for easier reloading
    # peft_model.save_pretrained(os.path.join(checkpoint_dir, "peft_model"))

    # Get embeddings from the fine-tuned model
    peft_model.eval()
    with torch.no_grad():
        tokenized_seqs = peft_model.esmc_model._tokenize(sequences[:10])
        embeddings = peft_model(input=tokenized_seqs)  # Get embeddings for first 10 sequences as an example

    print(f"Output embeddings shape: {embeddings.shape}")

    # Return useful objects
    return {
        "model": peft_model,
        "embeddings": embeddings,
        "training_history": training_history,
        "best_loss": best_loss,
        "final_loss": avg_epoch_loss,
        "checkpoint_dir": checkpoint_dir,
        "status": "completed"
    }


def load_fine_tuned_esmc(checkpoint_dir, resume_from=None):
    from esm.models.esmc import ESMC
    from peft import LoraConfig, get_peft_model, TaskType
    import glob
    import re

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Automatically find the most recent checkpoint if resume_from is None
    if resume_from is None:
        # Look for checkpoint files in the checkpoint directory
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))

        if checkpoint_files:
            # Extract epoch numbers from checkpoint filenames
            epoch_numbers = []
            for file_path in checkpoint_files:
                match = re.search(r'checkpoint_epoch_(\d+)\.pt', file_path)
                if match:
                    epoch_numbers.append((int(match.group(1)), file_path))

            # Find the most recent checkpoint
            if epoch_numbers:
                _, resume_from = max(epoch_numbers, key=lambda x: x[0])
                print(f"Auto-resuming from most recent checkpoint: {resume_from}")

        # Also check for final_model.pt
        final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
        if os.path.exists(final_model_path) and (not checkpoint_files or
                                                 os.path.getmtime(final_model_path) > os.path.getmtime(resume_from)):
            resume_from = final_model_path
            print(f"Auto-resuming from final model checkpoint: {resume_from}")

    # Define PEFT configuration
    peft_config_esmc = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias='none',
        layers_to_transform=[29, 28, 27, 26, 25, 24, 23, 22, 21],
        task_type=TaskType.CAUSAL_LM,
        target_modules=['attn.out_proj']
    )

    # Load pre-trained model
    clientc = ESMC.from_pretrained("esmc_300m")
    clientc = CustomModelWrapperesm(clientc)
    clientc.config = MockConfig(model_type="esmc")

    # Apply PEFT
    peft_model = get_peft_model(clientc, peft_config_esmc)
    peft_model = peft_model.to(device)

    # Load from checkpoint if specified or auto-detected
    if resume_from is not None and os.path.exists(resume_from):
        print(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        peft_model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']

        print(f"Resuming from epoch {start_epoch} with best loss: {best_loss:.4f}")
    else:
        if resume_from is not None:
            print(f"Checkpoint file {resume_from} not found. Starting from scratch.")
        else:
            print("No checkpoints found. Starting from scratch.")

    # Set model to eval
    peft_model.eval()

    # Return the model
    return peft_model
