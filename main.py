from Curation import Study
import pandas as pd

STUDY_ID = 'PRJNA393498'
STUDY_ID2 = 'immunoSEQ47'


if __name__ == '__main__':
    study = Study(STUDY_ID)
    # return numpy array of unique sequences - see docstring
    sequence = study.build_train_representations(samples=None, save=False, path=None)
    print(sequence)

    study = Study(STUDY_ID2)
    # return numpy array of unique sequences - see docstring
    sequence = study.build_train_representations(samples=None, save=False, path=None)
    print(sequence)

    # use esm_2 to extract representation
    model_checkpoint = "facebook/esm2_t4815B__UR50D"
    # pad to length of the longest sequence - can be done through esm parameters (no need to change sequences)
    # save a dictionary of {seq : representation} - make sure to save on lab folder may be heavy

#
# from esm.models.esmc import ESMC
# from esm.sdk.api import ESMProtein, LogitsConfig
#
# protein = ESMProtein(sequence="AAAAA")
# client = ESMC.from_pretrained("esmc_300m").to("cpu") # or "cuda"
# protein_tensor = client.encode(protein)
# logits_output = client.logits(
#    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
# )
# print(logits_output.logits, logits_output.embeddings)
#
