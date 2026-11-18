
# PAWS Paraphrase Detection | NLP & Deep Learning

A Python-based Natural Language Processing (NLP) project that explores sentence-pair similarity and paraphrase detection using linguistic analysis, classical machine learning, Sentence-BERT, and Transformer fine-tuning.

Developed as part of **COMP3420 – Artificial Intelligence for Text and Vision** at Macquarie University.

## Project Overview

The goal is to identify whether two sentences express the same meaning, even when their wording or word order differs.

The project uses the **PAWS (Paraphrase Adversaries from Word Scrambling)** dataset, which contains deliberately challenging sentence pairs with substantial word overlap.

**Labels:**
- `1` = Paraphrase
- `0` = Non-paraphrase

The Wikipedia-based `labeled_final` dataset contains:
- 49,401 training pairs
- 8,000 validation pairs
- 8,000 test pairs

Corpus-level analyses use the full training set. Machine learning and deep learning tasks use reproducible subsets of 5,000 training pairs and 1,000 test pairs with a random seed of 42.

## Technologies Used

- **Python & Jupyter Notebook** - Implementation and experiments
- **NLTK** - Tokenisation, POS tagging, and n-grams
- **NumPy & Pandas** - Data processing and analysis
- **Matplotlib** - Data visualisation
- **Scikit-learn** - TF-IDF, Naive Bayes, cross-validation, and evaluation
- **PyTorch** - Neural network training
- **Sentence Transformers** - Siamese Sentence-BERT embeddings and retrieval
- **Hugging Face Transformers** - DistilBERT fine-tuning
- **Hugging Face Datasets** - Loading the PAWS dataset

## Coding / Implementation

### 1. Part-of-Speech Analysis

Used NLTK to tokenise sentences and assign universal POS tags.

The implementation:
- Identifies nouns, verbs, and adjectives.
- Calculates their frequency and percentage distributions.
- Compares grammatical composition across sentence columns.
- Extracts the five most common nouns for paraphrase and non-paraphrase pairs.

### 2. N-gram Frequency and TF-IDF Analysis

Extracted unigrams, bigrams, and trigrams to analyse word overlap and word order.

The implementation:
- Calculates normalised n-gram frequencies.
- Identifies the most frequent n-grams for each class.
- Computes shared n-gram proportions between sentence pairs.
- Uses TF-IDF to identify informative terms.
- Visualises overlap distributions.

### 3. Gaussian Naive Bayes Classification

Developed a baseline paraphrase classifier using numeric features extracted from sentence pairs.

Features included:
- Number of shared words
- Number of shared bigrams
- Absolute difference in sentence lengths

The model was trained using Gaussian Naive Bayes and evaluated using stratified 5-fold cross-validation.

Performance metrics included accuracy, precision, recall, and a confusion matrix.

### 4. Siamese Sentence-BERT Retrieval

Implemented a Siamese neural network using the `all-MiniLM-L6-v2` Sentence-BERT model.

The implementation:
- Uses shared encoder weights for both sentences.
- Trains using contrastive loss.
- Generates semantic sentence embeddings.
- Calculates cosine similarity between queries and candidates.
- Ranks candidate sentences based on similarity.
- Displays the top five retrieved sentences for three example queries.

### 5. DistilBERT Fine-Tuning and Attention Analysis

Fine-tuned the pre-trained `distilbert-base-uncased` model for binary sentence-pair classification.

Training configuration:
- Epochs: 4
- Learning rate: 3e-5
- Training samples: 5,000
- Test samples: 1,000

The implementation:
- Tokenises sentence pairs.
- Fine-tunes DistilBERT for paraphrase classification.
- Evaluates predictions using classification metrics.
- Extracts final-layer attention weights.
- Compares attention patterns before and after fine-tuning.

## Outputs and Results

### 1. Dataset Statistics

| Dataset | Sentence Pairs | Positive Rate |
|---|---:|---:|
| Training | 49,401 | 0.442 |
| Test | 8,000 | 0.442 |

### 2. Part-of-Speech Distribution

| Sentence Column | Nouns | Verbs | Adjectives |
|---|---:|---:|---:|
| Sentence 1 | 35.4073% | 11.0428% | 5.8117% |
| Sentence 2 | 35.4154% | 11.0564% | 5.7911% |

Both sentence columns have very similar grammatical distributions.

Nouns represent the largest proportion among the three analysed POS categories.

### 3. N-gram Overlap Results

| N-gram | Paraphrase | Non-paraphrase |
|---|---:|---:|
| Unigram | 0.8858 | 0.8845 |
| Bigram | 0.6094 | 0.5008 |
| Trigram | 0.4656 | 0.3143 |

**Observation:**

Unigram overlap is nearly identical for both classes.

Bigrams and trigrams show larger differences, suggesting that word order is important for distinguishing paraphrases.

### 4. Naive Bayes Cross-Validation Results

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Accuracy | 0.5766 | 0.0091 |
| Precision | 0.5328 | 0.0221 |
| Recall | 0.2805 | 0.0229 |

The model achieved moderate accuracy but relatively low recall.

This suggests that simple word-overlap features struggle to identify many genuine paraphrase pairs.

### 5. Model Performance Comparison

| Metric | Gaussian Naive Bayes | Fine-tuned DistilBERT |
|---|---:|---:|
| Accuracy | 56.50% | 62.20% |
| Precision | 49.37% | 57.38% |
| Recall | 27.08% | 48.61% |
| F1-score | Not reported | 52.63% |

**Observation:**

Fine-tuned DistilBERT outperformed the Naive Bayes baseline across the reported comparable metrics.

This suggests that contextual Transformer representations are more effective than a small set of manually engineered overlap features for this task.

### 6. Naive Bayes Confusion Matrix Analysis

The Naive Bayes model produced:

- False Positives: 120
- False Negatives: 315

False negatives were the dominant error type.

A likely reason is that PAWS contains sentence pairs with complex structural differences, making simple overlap-based features insufficient.

### 7. Siamese Sentence-BERT Retrieval Results

The Siamese model completed training with a reported average loss of:

**0.0283**

Top-ranked cosine similarity scores for three example queries:

| Query | Top Cosine Similarity |
|---|---:|
| Query 1 | 0.9996 |
| Query 2 | 0.8473 |
| Query 3 | 0.9145 |

These results demonstrate how the trained encoder can rank candidate sentences using semantic similarity.

These scores represent individual retrieval examples rather than overall retrieval accuracy.

### 8. DistilBERT Training Results

| Epoch | Training Loss |
|---|---:|
| 1 | 0.6860 |
| 2 | 0.6673 |
| 3 | 0.5749 |
| 4 | 0.3949 |

Training loss decreased throughout the four epochs.

This indicates that the model learned patterns from the training data, although training loss alone does not establish generalisation performance.

### 9. Transformer Attention Analysis

Compared final-layer attention weights from the original pre-trained DistilBERT and the fine-tuned model.

For three selected content words, punctuation received the highest attention under both models.

When punctuation was excluded, some preferred word alignments changed following fine-tuning.

This highlights both the changes associated with model adaptation and the limitations of interpreting raw attention weights.

## Key Findings

1. **Word overlap alone is insufficient.** PAWS contains paraphrase and non-paraphrase pairs with nearly identical words.

2. **Word order matters.** Bigrams and trigrams provided more useful separation between classes than unigrams.

3. **Deep learning improved classification.** Fine-tuned DistilBERT achieved better test performance than the feature-based Naive Bayes baseline.

4. **Sentence embeddings support retrieval.** Sentence-BERT generated embeddings that could be ranked using cosine similarity.

5. **Attention requires careful interpretation.** Fine-tuning changed some token alignments, but attention weights should not be treated as definitive explanations of model predictions.

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/PAWS-Paraphrase-Detection-NLP.git
cd PAWS-Paraphrase-Detection-NLP
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the Notebook

Open `PAWS_Text_Processing.ipynb` using Jupyter Notebook, JupyterLab, or Google Colab.

### 4. Run the Code

Execute the notebook cells in order.

The notebook downloads the PAWS dataset, required NLTK resources, and pre-trained model weights.

Neural network training may take additional time depending on available hardware.

## Repository Structure

```text
PAWS-Paraphrase-Detection-NLP/
├── README.md
├── PAWS_Text_Processing.ipynb
├── requirements.txt
└── outputs/
```

The optional `outputs` directory can contain exported graphs and visualisations.

## Conclusion

This project demonstrates an end-to-end Natural Language Processing workflow, progressing from linguistic feature analysis to classical machine learning and Transformer-based models.

The experiments highlight the challenges of identifying paraphrases when sentences share similar words but differ in their structure and meaning.

The results demonstrate the value of contextual representations and deep learning for sentence-pair classification, while also exploring semantic retrieval and Transformer attention.

## Author

**Humaidi Rahman**

## Notes

This project was developed for educational purposes.

Reported results are taken from the notebook's saved outputs. Results may vary when rerunning the experiments depending on software versions, randomisation, and hardware.

