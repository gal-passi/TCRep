import numpy as np
import torch
from training.model_trainer import evaluate_model, CustomLossCriterion
from utils.cache_handler import get_model_config_str
from models.cvc_model import CVCClassifierModel
from inference.inference_ensemble import plot_confusion_matrices
from inference.plot_training import kde_normalizer
from inference import (
    inference_testing,
    inference_classification,
    reshef_inference as reshef_inference_module,
    inference_mcpas,
)


def run_inference(trained_model, args,
                  train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs,
                  test_pos_seqs, test_neg_seqs,
                  df_bld, df_hlt,
                  positive_seqs,
                  test_patient_inds, valid_patient_inds,
                  test_patient_ids, valid_patient_ids, unique_patient_ids,
                  test_masks, valid_masks,
                  aaseq_to_ratio, aaseq_to_dist,
                  batch_size, ch_dropout, cvc_layers_to_train,
                  freeze_embed_model, lora, ch_type,
                  pos_weights, reg_coef, ratio, inference_dict,
                  device, get_dataset_loader,
                  classification_v2=False,
                  dont_cache_inference=False,
                  filter_uncertain_seqs=False):
    """
    Master inference pipeline – runs all enabled inference steps depending on args.
    """
    if args.dont_inference:
        return

    print("\n=== Inference Stage ===")

    # Ensemble inference
    if args.to_ensemble and not args.combine_classification:
        run_ensemble_inference(trained_model, args,
                               valid_pos_seqs, valid_neg_seqs,
                               test_pos_seqs, test_neg_seqs,
                               device, pos_weights, reg_coef, ratio,
                               aaseq_to_ratio, aaseq_to_dist)

    # Negative partition inference
    if args.negative_partition > 0:
        run_partition_inference(trained_model, args,
                                valid_pos_seqs, valid_neg_seqs,
                                device, pos_weights, reg_coef, ratio,
                                aaseq_to_ratio, aaseq_to_dist)

    # Random forest embedding analysis
    if inference_dict.get("INFERENCE_TO_RANDOM_FOREST", False):
        run_random_forest_inference(args, train_pos_seqs, neg_seqs,
                                    valid_pos_seqs, valid_neg_seqs,
                                    test_pos_seqs, test_neg_seqs,
                                    batch_size, ch_dropout, cvc_layers_to_train,
                                    freeze_embed_model, lora, ch_type, device,
                                    getattr(args, "INFERENCE_TO_RF_PLOT_DIST_PER_PATIENT", False),
                                    test_patient_inds, valid_patient_inds,
                                    unique_patient_ids, test_masks, valid_masks,
                                    positive_seqs, df_bld, df_hlt,
                                    args.model_type)

    # # Inference on other datasets  # TODO: does not work after refactoring!
    # if inference_dict.get("INFERENCE_TO_DISPLAY_OTHER_DATASET_DISTS", False):
    #     run_other_dataset_inference(trained_model, args,
    #                                 args.dataset_type, unique_patient_ids,
    #                                 test_patient_inds, valid_patient_inds,
    #                                 df_bld, df_hlt,
    #                                 args.k_fold, args.dist_loss_type, device)

    # Embedding mappings
    if inference_dict.get("INFERENCE_TO_PLOT_EMBEDDING_MAPPINGS", False):
        from inference.plot_handler import plot_embedding_mappings
        import numpy as np
        df_bld_val_test = df_bld[df_bld['patient_id'].isin(
            np.concatenate([valid_patient_inds, test_patient_inds]))]
        model_config = f"{args.dataset_type}_" + get_model_config_str(args)
        plot_embedding_mappings(trained_model, df_bld_val_test,
                                valid_pos_seqs, test_pos_seqs,
                                valid_patient_inds, test_patient_inds,
                                model_config, device=device)

    # Classification model inference
    if inference_dict.get("INFERENCE_TO_CLASSIFICATION_MODEL", False):
        run_classification_inference(trained_model, args, df_bld, df_hlt,
                                     test_patient_ids, valid_patient_ids,
                                     valid_pos_seqs, valid_neg_seqs,
                                     test_pos_seqs, test_neg_seqs,
                                     aaseq_to_ratio, args.to_ensemble, device,
                                     args.k_fold, args.combine_classification,
                                     classification_v2, dont_cache_inference,
                                     filter_uncertain_seqs, batch_size,
                                     ch_dropout, cvc_layers_to_train,
                                     freeze_embed_model, lora, ch_type, get_dataset_loader)

    # Reshef inference
    if inference_dict.get("INFERENCE_RESHEF", False) and args.reshef_inference:
        run_reshef_inference(train_pos_seqs, neg_seqs,
                             valid_pos_seqs, valid_neg_seqs,
                             args.reshef_negative_part, args,
                             args.reshef_filter_train)

    # MCPAS inference
    if inference_dict.get("INFERENCE_MCPAS", False):
        run_mcpas_inference(trained_model, args, df_bld, df_hlt,
                            test_patient_ids, valid_patient_ids,
                            valid_pos_seqs, valid_neg_seqs,
                            test_pos_seqs, test_neg_seqs,
                            aaseq_to_ratio,
                            batch_size, ch_dropout, cvc_layers_to_train,
                            freeze_embed_model, lora, ch_type, device)



def run_ensemble_inference(trained_model, args, valid_pos_seqs, valid_neg_seqs,
                           test_pos_seqs, test_neg_seqs, device,
                           pos_weights, reg_coef, ratio,
                           aaseq_to_ratio, aaseq_to_dist):
    print("Running ensemble inference...")
    np.random.seed(42)

    class_weights = torch.tensor([1.0, pos_weights], dtype=torch.float, device=device)
    criterion = CustomLossCriterion(loss_type=args.loss_type,
                                    class_weights=class_weights,
                                    R=reg_coef, ratio=ratio,
                                    aaseq_to_ratio=aaseq_to_ratio,
                                    aaseq_to_dist=aaseq_to_dist,
                                    device=device)

    # Confusion matrices
    model_string = get_model_config_str(args)
    plot_confusion_matrices(trained_model, valid_pos_seqs, valid_neg_seqs,
                            test_pos_seqs, test_neg_seqs, model_string, device)

    # Weighted sum adjustment
    if trained_model.default_to_return == "weighted_sum":
        models_tprs = []
        for i in range(5):
            trained_model.default_to_return = i
            test_metrics = evaluate_model(trained_model, test_pos_seqs, test_neg_seqs, criterion, device)
            models_tprs.append(test_metrics[12])
        model_weights = np.array(models_tprs) / np.sum(models_tprs)
        trained_model.weights = model_weights
        trained_model.default_to_return = "weighted_sum"

    val_metrics = evaluate_model(trained_model, valid_pos_seqs, valid_neg_seqs, criterion, device)
    print("Validation:", val_metrics)
    return val_metrics


def run_partition_inference(trained_model, args, valid_pos_seqs, valid_neg_seqs,
                            device, pos_weights, reg_coef, ratio,
                            aaseq_to_ratio, aaseq_to_dist):
    print("Running negative partition inference...")
    np.random.seed(42)

    class_weights = torch.tensor([1.0, pos_weights], dtype=torch.float, device=device)
    criterion = CustomLossCriterion(loss_type=args.loss_type,
                                    class_weights=class_weights,
                                    R=reg_coef, ratio=ratio,
                                    aaseq_to_ratio=aaseq_to_ratio,
                                    aaseq_to_dist=aaseq_to_dist,
                                    device=device)

    val_metrics = evaluate_model(trained_model, valid_pos_seqs, valid_neg_seqs, criterion, device)
    print("Validation:", val_metrics)
    return val_metrics


def run_random_forest_inference(args, train_pos_seqs, neg_seqs, valid_pos_seqs,
                                valid_neg_seqs, test_pos_seqs, test_neg_seqs,
                                batch_size, ch_dropout, cvc_layers_to_train,
                                freeze_embed_model, lora, ch_type, device,
                                INFERENCE_TO_RF_PLOT_DIST_PER_PATIENT,
                                test_patient_inds, valid_patient_inds,
                                unique_patient_ids, test_masks, valid_masks,
                                positive_seqs, df_bld, df_hlt, model_type):
    print("Running random forest inference...")
    np.random.seed(42)

    df_embed_onehot = inference_testing.get_df_embeddings_onehot(
        train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs)

    untrained_model = CVCClassifierModel(batch_size=batch_size, ch_dropout=ch_dropout,
                                         cvc_layers_to_train=cvc_layers_to_train,
                                         freeze_embed_model=freeze_embed_model, lora=lora,
                                         ch_type=ch_type, device=device)

    df_embed = inference_testing.get_df_embeddings(args, untrained_model,
                                                   train_pos_seqs, neg_seqs,
                                                   valid_pos_seqs, valid_neg_seqs,
                                                   test_pos_seqs, test_neg_seqs,
                                                   metadata_name="untrained")

    # Analyze embeddings with RF
    rf_classifier = inference_testing.analyze_embeddings(
        df_embed, train_pos_seqs, neg_seqs,
        valid_pos_seqs, valid_neg_seqs,
        test_pos_seqs, test_neg_seqs,
        x=10, nw=1, pw=20, threshold=0.145,
        n_estimators=100, to_balance=True, to_plot=True)[1]

    inference_testing.analyze_embeddings(
        df_embed_onehot, train_pos_seqs, neg_seqs,
        valid_pos_seqs, valid_neg_seqs,
        test_pos_seqs, test_neg_seqs,
        x=10, nw=1, pw=20, threshold=0.145,
        n_estimators=100, to_balance=True, to_plot=True)

    if INFERENCE_TO_RF_PLOT_DIST_PER_PATIENT:
        inference_testing.plot_output_distributions_per_patient_random_forest(
            untrained_model, rf_classifier, test_patient_inds, valid_patient_inds, unique_patient_ids,
            test_masks, valid_masks, positive_seqs, df_bld, df_hlt, model_type, args, device)


# def run_other_dataset_inference(trained_model, args, dataset_type, unique_patient_ids,
#                                 test_patient_inds, valid_patient_inds, df_bld, df_hlt,
#                                 k_fold, dist_loss_type, device):
#     print("Running inference on other dataset...")
#     np.random.seed(42)
#
#     other_dataset_type = "article" if dataset_type == "ms" else "ms"
#     from datasets.dataset_loader import DatasetLoader
#     other_dataset_loader = DatasetLoader(dataset_type=other_dataset_type,
#                                          unique_patient_ids=unique_patient_ids,
#                                          k_fold=k_fold, dist_loss_type=dist_loss_type)
#     df_bld_other, df_hlt_other = other_dataset_loader.get_dfs()
#
#     inference_testing.display_distributions_on_different_sets(
#         trained_model, dataset_type, other_dataset_type,
#         unique_patient_ids, test_patient_inds, valid_patient_inds,
#         df_hlt, df_bld, df_bld_other, df_hlt_other, kde_normalizer, device)


def run_classification_inference(trained_model, args, df_bld, df_hlt,
                                 test_patient_ids, valid_patient_ids,
                                 valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs,
                                 aaseq_to_ratio, to_ensemble, device,
                                 k_fold, combine_classification,
                                 classification_v2, dont_cache_inference,
                                 filter_uncertain_seqs, batch_size,
                                 ch_dropout, cvc_layers_to_train,
                                 freeze_embed_model, lora, ch_type, get_dataset_loader):
    print("Running classification inference...")
    np.random.seed(42)

    if k_fold == 0 and combine_classification:
        inference_classification.inference_classification_model_combined(
            trained_model, args, to_ensemble,
            lambda k_fold_index: get_dataset_loader(args.dataset_type, k_fold=k_fold_index, to_k_fold=True,
                                                    use_dist_loss=args.use_dist_loss, dist_param=args.dist_param, neg_partition=args.negative_partition,
                                                    use_similar_negatives=args.use_similar_negatives,
                                                    neg_pos_ratio=args.neg_pos_ratio,
                                                    filter_num_of_patients=args.dataset_filter_num_of_patients,
                                                    filter_num_of_healthy=args.dataset_filter_num_of_healthy,
                                                    ratio=args.ratio,
                                                    filter_to_inflate=not args.dataset_filter_dont_inflate,
                                                    remove_seqs_by_len=args.remove_seqs_by_len,
                                                    top_percent=args.top_percent,
                                                    top_n_seqs=args.top_n_seqs,
                                                    extra_filter=args.extra_filter,
                                                    use_recurrence_loss=args.use_recurrence_loss,
                                                    loss_version=args.loss_version, verbose=True),
            device)
    else:
        if classification_v2:
            model_non_trained = CVCClassifierModel(batch_size=batch_size, ch_dropout=ch_dropout,
                                                   cvc_layers_to_train=cvc_layers_to_train,
                                                   freeze_embed_model=freeze_embed_model, lora=lora,
                                                   ch_type=ch_type, device=device)
            inference_classification.inference_classification_model_version2(
                trained_model, args, df_bld, df_hlt,
                test_patient_ids, valid_patient_ids,
                valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs,
                aaseq_to_ratio, to_ensemble, model_non_trained, device,
                k_fold_disease=1, chosen_components=2,
                dont_cache_inference=dont_cache_inference,
                filter_uncertain_seqs=filter_uncertain_seqs)
        else:
            inference_classification.inference_classification_model(
                trained_model, args, df_bld, df_hlt,
                test_patient_ids, valid_patient_ids,
                valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs,
                aaseq_to_ratio, to_ensemble, device)


def run_reshef_inference(train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs,
                         reshef_negative_part, args, reshef_filter_train):
    print("Running reshef inference...")
    np.random.seed(42)
    to_save_train_data = not reshef_filter_train
    reshef_inference_module.reshef_inference(train_pos_seqs, neg_seqs,
                                             valid_pos_seqs, valid_neg_seqs,
                                             reshef_negative_part, args,
                                             to_save_train_data=to_save_train_data)


def run_mcpas_inference(trained_model, args, df_bld, df_hlt,
                        test_patient_ids, valid_patient_ids,
                        valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs,
                        aaseq_to_ratio, batch_size, ch_dropout, cvc_layers_to_train,
                        freeze_embed_model, lora, ch_type, device):
    print("Running MCPAS inference...")
    model_non_trained = CVCClassifierModel(batch_size=batch_size, ch_dropout=ch_dropout,
                                           cvc_layers_to_train=cvc_layers_to_train,
                                           freeze_embed_model=freeze_embed_model, lora=lora,
                                           ch_type=ch_type, device=device)
    inference_mcpas.inference_mcpas(trained_model, args, df_bld, df_hlt,
                                    test_patient_ids, valid_patient_ids,
                                    valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs,
                                    aaseq_to_ratio, model_non_trained, device)
