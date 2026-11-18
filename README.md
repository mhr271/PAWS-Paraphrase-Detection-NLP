PAWS Paraphrase Detection | NLP & Deep Learning
A Python-based Natural Language Processing (NLP) project that explores sentence-pair similarity and paraphrase detection using linguistic analysis, classical machine learning, Sentence-BERT, and Transformer fine-tuning.

Developed as part of COMP3420 – Artificial Intelligence for Text and Vision at Macquarie University.
Project Overview
The goal is to identify whether two sentences express the same meaning, even when their wording or word order differs. The project uses the PAWS (Paraphrase Adversaries from Word Scrambling) dataset, which contains deliberately challenging sentence pairs with substantial word overlap.

Labels: 1 = paraphrase; 0 = non-paraphrase.

The Wikipedia-based labeled_final dataset contains 49,401 training pairs, 8,000 validation pairs, and 8,000 test pairs. The corpus-level analyses use the full training set; the modelling tasks use reproducible subsets of 5,000 training pairs and 1,000 test pairs (random_seed=42).
Technologies Used
- Python & Jupyter Notebook — implementation and reproducible experiments
- NLTK — tokenisation, universal part-of-speech tagging, and n-grams
- NumPy & pandas — numerical processing and result tables
- Matplotlib — visualisation
- scikit-learn — TF-IDF, Gaussian Naive Bayes, cross-validation, and evaluation metrics
- PyTorch — neural-network optimisation
- Sentence Transformers — Siamese Sentence-BERT embeddings and retrieval
- Hugging Face Transformers & Datasets — PAWS loading and DistilBERT fine-tuning
Coding / Implementation
1. Part-of-Speech Analysis
Tokenised the sentence columns and tagged them with NLTK's universal POS tagset. Measured the frequency of nouns, verbs, and adjectives, and identified common nouns separately for paraphrase and non-paraphrase pairs.
2. N-gram Frequency and TF-IDF
Extracted unigrams, bigrams, and trigrams; compared normalized frequencies and total TF-IDF weights across classes. Computed shared n-gram proportions for sentence pairs and visualised their distributions.
3. Gaussian Naive Bayes Classification
Created three numeric pair-level features: shared words, shared bigrams, and absolute sentence-length difference. Trained a Gaussian Naive Bayes model and evaluated it with stratified five-fold cross-validation and a held-out test subset.
4. Siamese Sentence-BERT Retrieval
Fine-tuned a shared Sentence-BERT encoder (all-MiniLM-L6-v2) using contrastive loss. Encoded a candidate pool of test sentences, ranked candidates by cosine similarity, and displayed the five best matches for three example queries.
5. DistilBERT Fine-Tuning and Attention
Fine-tuned distilbert-base-uncased for sentence-pair classification over 4 epochs with a 3e-5 learning rate. Evaluated the classifier and compared final-layer, head-averaged attention from the original and fine-tuned models for a correctly classified pair.
Outputs and Results
Dataset
Split	Sentence pairs	Positive label rate
Training	49,401	0.442
Test	8,000	0.442


Part-of-speech distribution
Sentence column	Nouns	Verbs	Adjectives
Sentence 1	35.4073%	11.0428%	5.8117%
Sentence 2	35.4154%	11.0564%	5.7911%


Mean shared n-gram proportions
N-gram	Paraphrase	Non-paraphrase
Unigram	0.8858	0.8845
Bigram	0.6094	0.5008
Trigram	0.4656	0.3143


Word overlap alone offers little separation between classes, while ordered bigrams and trigrams show larger differences.
Naive Bayes: five-fold cross-validation
Metric	Mean	Standard deviation
Accuracy	0.5766	0.0091
Precision	0.5328	0.0221
Recall	0.2805	0.0229


Test-set classification comparison
Metric	Gaussian Naive Bayes	Fine-tuned DistilBERT
Accuracy	56.50%	62.20%
Precision	49.37%	57.38%
Recall	27.08%	48.61%
F1-score	Not reported	52.63%


On the 1,000-pair test subset, the Naive Bayes model produced 120 false positives and 315 false negatives, illustrating the limitations of relying on a few surface-overlap features.
Sentence-BERT retrieval
The Siamese retrieval model finished training with a reported average loss of 0.0283. In the three illustrated queries, the top-ranked candidates had cosine similarities of 0.9996, 0.8473, and 0.9145, respectively. These are example retrieval scores, not overall retrieval accuracy.
DistilBERT training
Epoch	Training loss
1	0.6860
2	0.6673
3	0.5749
4	0.3949


The attention comparison showed that the highest-attention token was punctuation (.) for the three selected content words under both models. When punctuation was excluded, some preferred token alignments changed after fine-tuning, illustrating why attention interpretations require care.
Key Takeaways
- Word overlap is not enough: paraphrase and non-paraphrase pairs can contain nearly identical words.
- Word order matters: bigram and trigram overlap separated the classes more clearly than unigram overlap.
- Contextual modelling helped: fine-tuned DistilBERT outperformed the three-feature Naive Bayes baseline on the same test subset.
- Retrieval is a different task: high cosine similarity scores for example Sentence-BERT queries should not be interpreted as classification accuracy.
