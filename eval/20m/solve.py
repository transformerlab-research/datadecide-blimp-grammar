"""MUTABLE — the 25-recipe x 90M x 3-seed BLiMP sweep."""
import os

PARADIGMS = os.environ.get("PARADIGMS", ",".join([
    "anaphor_gender_agreement",
    "determiner_noun_agreement_1",   # saturated at .99+; kept for completeness,
                                     # EXCLUDED from significance testing downstream
    "npi_present_1",
    "wh_questions_object_gap",
    "irregular_past_participle_verbs",
    "animate_subject_passive",
])).split(",")

RECIPES = [
    "c4", "dclm-baseline", "dclm-baseline-qc-10p", "dclm-baseline-qc-20p",
    "dclm-baseline-qc-7p-fw2", "dclm-baseline-qc-7p-fw3",
    "dclm-baseline-qc-fw-10p", "dclm-baseline-qc-fw-3p",
    "dclm-baseline-25p-dolma1.7-75p", "dclm-baseline-50p-dolma1.7-50p",
    "dclm-baseline-75p-dolma1.7-25p", "dolma1_6plus", "dolma1_7",
    "dolma1_7-no-flan", "dolma1_7-no-reddit", "dolma1_7-no-code",
    "dolma1_7-no-math-code", "falcon", "falcon-and-cc", "falcon-and-cc-qc-10p",
    "falcon-and-cc-qc-20p", "falcon-and-cc-qc-orig-10p",
    "falcon-and-cc-qc-tulu-10p", "fineweb-edu", "fineweb-pro",
]
SIZE   = os.environ.get("SIZE", "20M")
SEEDS  = ["default", "small-aux-2", "small-aux-3"]
N_ITEMS = int(os.environ.get("N_ITEMS", "1000"))
