

def run_vae(args, dataset_loader, device):
    if not args.train_vae:
        return

    print("Running VAE training and inference...")
    from vae.vae_training import run_vae_training_and_inference
    run_vae_training_and_inference(args, dataset_loader, device)

    # Note: we won't get to this part because there is an exit command in the previous vae line
    from vae.vae_training_dynamic import train_and_inference_vae
    train_and_inference_vae(args, dataset_loader, device)
