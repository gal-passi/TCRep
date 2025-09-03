from models.cvc_ensemble_model import CVCEnsembleModel
from utils.cache_handler import load_model_state, get_model_dir
from training.model_trainer import train_model

def load_or_train_model(model, args,
                        train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs,
                        test_pos_seqs, test_neg_seqs,
                        aaseq_to_ratio, aaseq_to_dist, aaseq_to_nneighbors,
                        device,
                        epochs, learning_rate, batch_size, neg_pos_ratio,
                        log_wandb, model_type, loss_type,
                        freeze_embed_model, special_criterion,
                        embedding_lr, reg_coef, pos_weights,
                        scheduler_type, masking, ratio,
                        changing_negatives, optimizer_type,
                        reshef_inference, reshef_filter_train, reshef_negative_part):
    """
    Either load a previously trained model (if available and allowed),
    or train a new one and return it.
    """

    trained_model = None

    if not args.force_retrain:
        if args.combine_classification:
            trained_model = model

        elif args.to_ensemble:
            cache_dir = get_model_dir(args)
            trained_model = CVCEnsembleModel(args, device,
                                             cache_dir=cache_dir,
                                             default_to_return="min")  # or "weighted_sum"

        elif args.test_mode_epoch >= 0:
            trained_model = load_model_state(model, args, args.test_mode_epoch, device)
            if trained_model is None:
                print(f"Model for epoch {args.test_mode_epoch} is not available!")
                exit(1)

        elif model_type == "cvc_combined_reshef":
            trained_model = model

        else:
            trained_model = load_model_state(model, args, args.epochs - 1, device)

    if trained_model is None:
        trained_model, history = train_model(
            model,
            train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs,
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
            aaseq_to_nneighbors=aaseq_to_nneighbors,
            masking=masking,
            ratio=ratio,
            change_negatives=changing_negatives,
            optimizer_type=optimizer_type,
            reshef_inference=reshef_inference,
            reshef_filter_train=reshef_filter_train,
            reshef_negative_part=reshef_negative_part,
            test_pos_seqs=test_pos_seqs,
            test_neg_seqs=test_neg_seqs,
            args=args,
        )

    return trained_model
