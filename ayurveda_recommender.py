"""
Ayurvedic Formulation Recommender System
=========================================
A TF-IDF + Cosine Similarity based retrieval system for classical
Ayurvedic formulations, with optional Sentence-BERT embeddings.

Author: Claude
"""

import re
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# 1.  NLP UTILITIES  (no external corpora needed — fully self-contained)
# ---------------------------------------------------------------------------

# Curated English stopwords (covers medical/symptom context well)
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "each", "few", "more", "most",
    "other", "some", "such", "than", "then", "that", "this", "these", "those",
    "it", "its", "itself", "as", "if", "while", "because", "although",
    "during", "after", "before", "above", "below", "between", "into", "through",
    "which", "who", "whom", "when", "where", "why", "how", "all", "any",
    "also", "very", "just", "only", "also", "too", "even", "still", "both",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they", "their",
    "them", "his", "her", "what", "there",
}

# Simple rule-based suffix lemmatizer (covers common medical English forms)
LEMMA_RULES = [
    (r"nesses$", "ness"),
    (r"ations$", "ation"),
    (r"ments$", "ment"),
    (r"ities$", "ity"),
    (r"ings$", "ing"),
    (r"ness$", ""),
    (r"tion$", "te"),
    (r"ment$", ""),
    (r"ity$", ""),
    (r"ful$", ""),
    (r"less$", ""),
    (r"ing$", ""),
    (r"ied$", "y"),
    (r"ies$", "y"),
    (r"ed$", ""),
    (r"er$", ""),
    (r"est$", ""),
    (r"s$", ""),
]


def simple_lemmatize(word: str) -> str:
    """Apply rule-based suffix stripping (lightweight lemmatization)."""
    if len(word) <= 4:
        return word
    for pattern, replacement in LEMMA_RULES:
        new = re.sub(pattern, replacement, word)
        if new != word and len(new) >= 3:
            return new
    return word


def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
      1. Lowercase
      2. Remove punctuation & digits
      3. Tokenize on whitespace
      4. Remove stopwords
      5. Lemmatize each token
    Returns a single cleaned string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Lowercase
    text = text.lower()

    # Remove punctuation / digits — keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords, short tokens, and lemmatize
    tokens = [
        simple_lemmatize(tok)
        for tok in tokens
        if tok not in STOPWORDS and len(tok) > 2
    ]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# 2.  DATA LOADING & PREPROCESSING
# ---------------------------------------------------------------------------

def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV and create preprocessed text columns used for vectorization.

    Adds:
      - 'clean_symptoms'    : preprocessed 'symptoms' column
      - 'clean_description' : preprocessed 'description' column
      - 'combined_text'     : symptoms (weight x2) + description for TF-IDF
    """
    df = pd.read_csv(csv_path)

    # Fill NaN with empty string to avoid type errors
    for col in ["symptoms", "description", "formulation_name",
                "ingredients", "dosage", "category",
                "product_link", "doctor_type"]:
        df[col] = df[col].fillna("").astype(str)

    df["clean_symptoms"]    = df["symptoms"].apply(preprocess_text)
    df["clean_description"] = df["description"].apply(preprocess_text)

    # Give symptoms 2× weight in the combined corpus
    df["combined_text"] = (
        df["clean_symptoms"] + " " + df["clean_symptoms"] + " " +
        df["clean_description"]
    )

    return df


# ---------------------------------------------------------------------------
# 3.  TFIDF VECTORIZER  (primary model)
# ---------------------------------------------------------------------------

def build_tfidf_model(df: pd.DataFrame):
    """
    Fit a TF-IDF vectorizer on the combined corpus.
    Returns (vectorizer, symptom_matrix, description_matrix).
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        max_df=0.95,          # ignore terms in >95 % of docs
        min_df=1,             # keep rare Ayurvedic terms
        sublinear_tf=True,    # apply log normalisation to term freq
    )

    # Fit on combined text; keep separate matrices for weighted scoring
    vectorizer.fit(df["combined_text"])

    symptom_matrix     = vectorizer.transform(df["clean_symptoms"])
    description_matrix = vectorizer.transform(df["clean_description"])

    return vectorizer, symptom_matrix, description_matrix


# ---------------------------------------------------------------------------
# 4.  OPTIONAL — SENTENCE-BERT  (semantic embeddings)
# ---------------------------------------------------------------------------

def build_sbert_model(df: pd.DataFrame):
    """
    Build sentence embeddings using Sentence-BERT (all-MiniLM-L6-v2).
    Requires: pip install sentence-transformers
    Returns (model, symptom_embeddings, description_embeddings).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. "
            "Run: pip install sentence-transformers"
        )

    model = SentenceTransformer("all-MiniLM-L6-v2")

    symptom_emb     = model.encode(
        df["clean_symptoms"].tolist(), show_progress_bar=True
    )
    description_emb = model.encode(
        df["clean_description"].tolist(), show_progress_bar=True
    )

    return model, symptom_emb, description_emb


# ---------------------------------------------------------------------------
# 5.  RECOMMENDATION ENGINE
# ---------------------------------------------------------------------------

class AyurvedaRecommender:
    """
    Similarity-based retrieval for Ayurvedic formulations.

    Scoring:
      final_score = α × sim(symptoms) + (1-α) × sim(description)

    where α controls how much weight goes to symptom similarity.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        symptom_weight: float = 0.7,
        use_sbert: bool = False,
    ):
        self.df              = df.reset_index(drop=True)
        self.symptom_weight  = symptom_weight
        self.use_sbert       = use_sbert

        # --- TF-IDF (always built) ---
        print("⚙  Building TF-IDF model …")
        (self.tfidf_vec,
         self.tfidf_sym_mat,
         self.tfidf_desc_mat) = build_tfidf_model(df)
        print("✓  TF-IDF ready.")

        # --- Sentence-BERT (optional) ---
        if use_sbert:
            print("⚙  Building Sentence-BERT model …")
            (self.sbert_model,
             self.sbert_sym_emb,
             self.sbert_desc_emb) = build_sbert_model(df)
            print("✓  SBERT ready.")

    # ------------------------------------------------------------------
    def _tfidf_scores(self, query: str) -> np.ndarray:
        """Return weighted cosine similarity scores using TF-IDF."""
        clean_q = preprocess_text(query)
        q_vec   = self.tfidf_vec.transform([clean_q])

        sym_scores  = cosine_similarity(q_vec, self.tfidf_sym_mat)[0]
        desc_scores = cosine_similarity(q_vec, self.tfidf_desc_mat)[0]

        return (self.symptom_weight  * sym_scores +
                (1 - self.symptom_weight) * desc_scores)

    def _sbert_scores(self, query: str) -> np.ndarray:
        """Return weighted cosine similarity scores using SBERT."""
        clean_q = preprocess_text(query)
        q_emb   = self.sbert_model.encode([clean_q])

        sym_scores  = cosine_similarity(q_emb, self.sbert_sym_emb)[0]
        desc_scores = cosine_similarity(q_emb, self.sbert_desc_emb)[0]

        return (self.symptom_weight  * sym_scores +
                (1 - self.symptom_weight) * desc_scores)

    # ------------------------------------------------------------------
    def recommend_formulations(
        self,
        user_input: str,
        top_n: int = 5,
        use_sbert: Optional[bool] = None,
    ) -> List[Dict]:
        """
        Return top-N most relevant Ayurvedic formulations.

        Parameters
        ----------
        user_input : str
            Free-form symptom description, e.g. "bloating nausea stomach pain".
        top_n : int
            Number of results to return (default 5).
        use_sbert : bool | None
            Override instance-level setting for this call.

        Returns
        -------
        List of dicts with keys:
            rank, formulation_name, symptoms, doctor_type, category,
            description, ingredients, dosage, product_link, score
        """
        if not user_input or not user_input.strip():
            return []

        _use_sbert = self.use_sbert if use_sbert is None else use_sbert

        # Compute scores
        if _use_sbert and hasattr(self, "sbert_model"):
            scores = self._sbert_scores(user_input)
        else:
            scores = self._tfidf_scores(user_input)

        # Sort descending and pick top_n
        top_indices = np.argsort(scores)[::-1][:top_n]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
                row = self.df.iloc[idx]

                product_link = row["product_link"]
                if not product_link:
                    product_link = f"https://www.google.com/search?q={row['formulation_name']} ayurveda buy"

                # ✅ FIX DOSHA HERE
                dosha = row.get("dosha", "")

                if isinstance(dosha, str):
                    dosha = [d.strip() for d in dosha.split(",") if d.strip()]

                results.append({
                    "rank"             : rank,
                    "formulation_name" : row["formulation_name"],
                    "symptoms"         : row["symptoms"],
                    "doctor_type"      : row["doctor_type"],
                    "category"         : row["category"],
                    "description"      : row["description"],
                    "ingredients"      : row["ingredients"],
                    "dosage"           : row["dosage"],
                    "dosha"            : dosha,  
                    "product_link"     : product_link,
                    "score"            : round(float(scores[idx]), 4),
                })
                
        return results


# ---------------------------------------------------------------------------
# 6.  CONVENIENCE FUNCTION (as specified)
# ---------------------------------------------------------------------------

# Module-level singleton — initialised lazily on first call
_recommender: Optional[AyurvedaRecommender] = None
_CSV_PATH = "herbal_formulation.csv"


def _get_recommender(csv_path: str = _CSV_PATH) -> AyurvedaRecommender:
    global _recommender
    if _recommender is None:
        df = load_and_preprocess(csv_path)
        _recommender = AyurvedaRecommender(df)
    return _recommender


def recommend_formulations(user_input: str, top_n: int = 5) -> List[Dict]:
    """
    Public API: return top-N Ayurvedic formulations for given symptoms.

    Usage
    -----
    >>> results = recommend_formulations("bloating, nausea, stomach pain")
    >>> for r in results:
    ...     print(r["rank"], r["formulation_name"], r["score"])
    """
    rec = _get_recommender()
    return rec.recommend_formulations(user_input, top_n=top_n)


# ---------------------------------------------------------------------------
# 7.  TEST CASES
# ---------------------------------------------------------------------------

def run_tests(csv_path: str = _CSV_PATH):
    """Run a battery of test queries and pretty-print results."""
    df  = load_and_preprocess(csv_path)
    rec = AyurvedaRecommender(df)

    test_queries = [
        "bloating, nausea, stomach pain",
        "stress anxiety memory loss insomnia",
        "joint pain inflammation arthritis",
        "cough cold respiratory congestion",
        "skin rash eczema itching",
        "fever headache body ache weakness",
        "hair loss dandruff scalp problems",
        "diabetes high blood sugar frequent urination",
    ]

    print("\n" + "=" * 70)
    print("  AYURVEDIC FORMULATION RECOMMENDER — TEST RESULTS")
    print("=" * 70)

    for query in test_queries:
        print(f"\n🔍  Query: \"{query}\"")
        print("-" * 60)
        results = rec.recommend_formulations(query, top_n=5)
        if not results:
            print("  No results found.")
        for r in results:
            dt = r["doctor_type"] or "General Physician"
            print(
                f"  #{r['rank']}  {r['formulation_name']:<35}"
                f"  score={r['score']:.3f}  [{r['category']}]  {dt}"
            )

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else _CSV_PATH
    run_tests(csv_path)
