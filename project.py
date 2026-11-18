!pip -q install datasets    # Install the datasets package to access the dataset
# add the packages you used, and specify the version you installed

from datasets import load_dataset
import numpy as np

# %%
# 1) Load PAWS (Wikipedia-based, human-labelled subset)
paws = load_dataset("google-research-datasets/paws", "labeled_final")

# PAWS provides a labelled test split, so no substitution is needed
train_ds = paws["train"]
test_ds  = paws["test"]

s1_tr = list(train_ds["sentence1"])
s2_tr = list(train_ds["sentence2"])
y_tr  = np.array(train_ds["label"])

s1_te = list(test_ds["sentence1"])
s2_te = list(test_ds["sentence2"])
y_te  = np.array(test_ds["label"])

print(f"Train: {len(s1_tr)} pairs, positive rate {y_tr.mean():.3f}")
print(f"Test:  {len(s1_te)} pairs, positive rate {y_te.mean():.3f}")

# %%
from importlib.metadata import version

packages = [
    'datasets',
    'numpy',
    'nltk',
    'pandas',
    'matplotlib',
    'scikit-learn',
    'torch',
    'transformers',
    'sentence-transformers'
]

for package in packages:
    print(f"{package}: {version(package)}")

# %%
# Write your code below

## Task 1.1

import nltk
from collections import Counter 

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('universal_tagset')


def tag_sentences(sentences):
    tagged_results = []

    for text in sentences:
        tokenized = [
            nltk.word_tokenize(sentence)
            for sentence in nltk.sent_tokenize(text)
        ]

        tagged = nltk.pos_tag_sents(
            tokenized, tagset='universal'
        )

        tagged_results.append(
            [item for sentence in tagged for item in sentence]
        )

    return tagged_results
# Tag each sentence once and reuse the results for both analyses

tagged_s1 = tag_sentences(s1_tr)
tagged_s2 = tag_sentences(s2_tr)


def pos_distribution(tagged_sentences):
    all_tags = []

    for sentence in tagged_sentences:
        for word, tag in sentence:
            all_tags.append(tag)


    tag_counts = Counter(all_tags)
    total_tokens =len(all_tags)

    for tag in ['NOUN' , 'VERB' , 'ADJ']:
        count = tag_counts[tag]
        percentage = count / total_tokens*100
        print(f"{tag} :{count} ({percentage:.4f}%)")

print("\nSentence 1")
pos_distribution(tagged_s1)

print("\nSentence 2")
pos_distribution(tagged_s2)

## Task 1.2


import pandas as pd 

def top_nouns(tagged_sentences, labels, target_label):
    nouns = []

# Collect nouns only from pairs with the required label

    for sentence, label in zip(tagged_sentences, labels):
        if label == target_label:
            for word, tag in sentence:
                if tag == 'NOUN':
                    nouns.append(word.lower())

    noun_count = Counter(nouns)
    return noun_count.most_common(5)

s1_paraphrase = top_nouns(tagged_s1, y_tr, 1)
s1_nonparaphrase = top_nouns(tagged_s1, y_tr, 0)

s2_paraphrase = top_nouns(tagged_s2, y_tr, 1)
s2_nonparaphrase = top_nouns(tagged_s2, y_tr, 0)



results = pd.DataFrame({
    'Sentence 1 - Paraphrase': s1_paraphrase,
    'Sentence 1 - Non-paraphrase': s1_nonparaphrase,
    'Sentence 2 - Paraphrase': s2_paraphrase,
    'Sentence 2 - Non-paraphrase': s2_nonparaphrase
})

results.index = range(1, 6)
results.index.name = 'Rank'

display(results)



# %%
# Write your code below

## Task 2.1 

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from nltk.util import ngrams

def get_ngrams(words, n):
    return list(ngrams(words, n))

# Tokenize the training sentences once to avoid repeated processing


tokens_s1 = [nltk.word_tokenize(text.lower()) for text in s1_tr]
tokens_s2 = [nltk.word_tokenize(text.lower()) for text in s2_tr]


def analyse_ngrams(n, target_label):
    counts = Counter()
    shared_proportions = []

    for s1, s2, label in zip(tokens_s1, tokens_s2, y_tr):
        if label == target_label:
            grams1 = get_ngrams(s1, n)
            grams2 = get_ngrams(s2, n)

            counts.update(grams1)
            counts.update(grams2)

            set1 = set(grams1)
            set2 = set(grams2)

# Calculate shared n-gram proportion using intersection and union

            union = set1 | set2
            intersection = set1 & set2

            proportion = len(intersection) / len(union) if union else 0
            shared_proportions.append(proportion)

    total = sum(counts.values())

    top10 = [
        (gram, round(count / total, 4))
        for gram, count in counts.most_common(10)
    ]

    return top10, shared_proportions


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, n in enumerate([1, 2, 3]):

    para_top, para_shared = analyse_ngrams(n, 1)
    non_top, non_shared = analyse_ngrams(n, 0)

    results = pd.DataFrame({
        'Paraphrase': para_top,
        'Non-paraphrase': non_top
    })

    results.index = range(1, 11)
    results.index.name = 'Rank'

    print(f"\nTop 10 {n}-grams by Frequency")
    display(results)

    para_mean = np.mean(para_shared)
    non_mean = np.mean(non_shared)

    print(f"Paraphrase mean: {para_mean:.4f}")
    print(f"Non-paraphrase mean: {non_mean:.4f}")

    axes[i].hist(
        para_shared, bins=20, alpha=0.5,
        density=True, label='Paraphrase'
    )

    axes[i].hist(
        non_shared, bins=20, alpha=0.5,
        density=True, label='Non-paraphrase'
    )

    axes[i].set_title(f'{n}-gram')
    axes[i].set_xlabel('Shared N-gram Proportion')
    axes[i].set_ylabel('Density')
    axes[i].set_xlim(0, 1)

handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='upper center',
           bbox_to_anchor=(0.5, 1.02), ncol=2)

fig.suptitle('Shared N-gram Proportions by Class', y=1.13)

plt.tight_layout()
plt.show()

## Task 2.2 


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# Combine both sentence columns and preserve their class labels

documents = tokens_s1 + tokens_s2
labels = np.concatenate([y_tr, y_tr])

for n in [1, 2, 3]:

# Calculate TF-IDF weights for unigrams, bigrams and trigrams

    vectorizer = TfidfVectorizer(
        tokenizer=lambda tokens: tokens,
        preprocessor=lambda tokens: tokens,
        token_pattern=None,
        lowercase=False,
        ngram_range=(n, n)
)

    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    para_weights = np.asarray(
        tfidf_matrix[labels == 1].sum(axis=0)
    ).flatten()

    non_weights = np.asarray(
        tfidf_matrix[labels == 0].sum(axis=0)
    ).flatten()

 
    para_top = np.argsort(para_weights)[::-1][:10]
    non_top = np.argsort(non_weights)[::-1][:10]

    results = pd.DataFrame({
        'Paraphrase N-gram': feature_names[para_top],
        'Paraphrase TF-IDF': para_weights[para_top],
        'Non-paraphrase N-gram': feature_names[non_top],
        'Non-paraphrase TF-IDF': non_weights[non_top]
    })

    results.index = range(1, 11)
    results.index.name = 'Rank'

    print(f"\nTop 10 TF-IDF {n}-grams")
    display(results.round(4))

    

# %%
# From now on, you are allowed to use a subset of the dataset which requires less computing resources.
# Note that you have to use the same subset for the following coding tasks, which ensures fairness
# when comparing performance across different models.

RANDOM_SEED = 42
N_TRAIN = 5000
N_TEST  = 1000

# Training subset
train_subset = train_ds.shuffle(seed=RANDOM_SEED).select(range(N_TRAIN))

# Test subset
test_subset  = test_ds.shuffle(seed=RANDOM_SEED).select(range(N_TEST))

# %%
# Write your code below

from nltk.util import bigrams
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate , StratifiedKFold
from sklearn.metrics import accuracy_score , precision_score , recall_score
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay

## Task 3.1

def pair_features(s1,s2):
    words1 = nltk.word_tokenize(s1.lower())
    words2 = nltk.word_tokenize(s2.lower())

# Extract three numerical features from each sentence pair

    shared_words = len(set(words1) & set(words2))
    shared_bigrams = len(set(bigrams(words1)) & set(bigrams(words2)))
    length_difference = abs(len(words1)-len(words2))

    return [shared_words,shared_bigrams,length_difference]

X_train = np.array([
    pair_features(s1,s2)
    for s1, s2 in zip(train_subset["sentence1"] , train_subset["sentence2"])

])

y_train = np.array(train_subset["label"])

X_test = np.array([
    pair_features(s1,s2)
    for s1, s2 in zip(test_subset["sentence1"], test_subset["sentence2"])
])

y_test = np.array(test_subset["label"])

nb_model = GaussianNB()
nb_model.fit(X_train,y_train)


## Task3.2

# Evaluate Naive Bayes using five-fold stratified cross-validation

cv = StratifiedKFold(n_splits=5 , shuffle=True,random_state=RANDOM_SEED)

scores = cross_validate(
    GaussianNB(), X_train , y_train , cv=cv,
    scoring = ['accuracy','precision','recall']
)

cv_results = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall'],
    'Mean': [
        scores['test_accuracy'].mean(),
        scores['test_precision'].mean(),
        scores['test_recall'].mean()
    ],
    'Standard Deviation': [
        scores['test_accuracy'].std(),
        scores['test_precision'].std(),
        scores['test_recall'].std()
    ]
})
display(cv_results.round(4))
y_pred = nb_model.predict(X_test)

test_results = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall'],
    'Test Score': [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred)
    ]
})

display(test_results.round(4))

## Task 3.3 

# Examine false positives and false negatives on the test subset

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Non-paraphrase', 'Paraphrase']
).plot(cmap='Blues', values_format='d')

plt.title('Naive Bayes Confusion Matrix')
plt.show()

print("False Positives:", cm[0, 1])
print("False Negatives:", cm[1, 0])


# %%
# Write your code below

# Task 4.1

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, losses

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

contrastive_loss = losses.ContrastiveLoss(model=model)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

train_pairs = list(zip(
    train_subset['sentence1'],
    train_subset['sentence2'],
    train_subset['label']
))

model.train()
total_loss = 0

for start in range(0, len(train_pairs), 16):
    batch = train_pairs[start:start + 16]

    s1 = [pair[0] for pair in batch]
    s2 = [pair[1] for pair in batch]

    labels = torch.tensor(
        [pair[2] for pair in batch],
        dtype=torch.float32,
        device=model.device
    )

    input1 = model.tokenize(s1)
    input2 = model.tokenize(s2)

    input1 = {
        k: v.to(model.device) if torch.is_tensor(v) else v
        for k, v in input1.items()
    }

    input2 = {
        k: v.to(model.device) if torch.is_tensor(v) else v
        for k, v in input2.items()
    }

    optimizer.zero_grad()

    # Both sentence inputs use the same Sentence-BERT model,
    # so the two Siamese branches share identical encoder weights
    
    loss = contrastive_loss([input1, input2], labels)

    loss.backward()
    optimizer.step()

    total_loss += loss.item()

average_loss = total_loss / len(range(0, len(train_pairs), 16))
print(f"Training Loss: {average_loss:.4f}")


# Task 4.2 

model.eval()

candidate_pool = list(test_subset['sentence2'])

# Encode the candidate pool once and reuse it for all queries


candidate_embeddings = model.encode(
    candidate_pool,
    convert_to_tensor=True,
    normalize_embeddings=True
)

positive_indices = [
    i for i, label in enumerate(test_subset['label'])
    if label == 1
][:3]

pd.set_option('display.max_colwidth', None)

for query_number, index in enumerate(positive_indices, start=1):

    query = test_subset['sentence1'][index]

    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
# Rank candidate sentences by cosine similarity to the query

    scores = F.cosine_similarity(
        query_embedding.unsqueeze(0),
        candidate_embeddings
    )

    top_scores, top_indices = torch.topk(scores, 5)

    results = pd.DataFrame({
        'Rank': range(1, 6),
        'Candidate Sentence': [
            candidate_pool[i] for i in top_indices.tolist()
        ],
        'Cosine Similarity': top_scores.tolist()
    })

    print(f"\nQuery {query_number}: {query}")
    display(results.round(4))

# %%
# Write your code below

# Task 5.1
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score

# Set random seeds for reproducible training
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device(
    'mps' if torch.backends.mps.is_available() else 'cpu'
)

# Tokenize the sentence pairs for training and testing

train_inputs = tokenizer(
    list(train_subset['sentence1']),
    list(train_subset['sentence2']),
    padding=True, truncation=True,
    max_length=128, return_tensors='pt'
)

test_inputs = tokenizer(
    list(test_subset['sentence1']),
    list(test_subset['sentence2']),
    padding=True, truncation=True,
    max_length=128, return_tensors='pt'
)

train_data = TensorDataset(
    train_inputs['input_ids'],
    train_inputs['attention_mask'],
    torch.tensor(train_subset['label'])
)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2,
    attn_implementation='eager'
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

for epoch in range(4):
    model.train()
    total_loss = 0

    for input_ids, attention_mask, labels in train_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Calculate classification loss and update model weights

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        outputs.loss.backward()
        optimizer.step()

        total_loss += outputs.loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/4 - Loss: {average_loss:.4f}")

# Task 5.2
# Evaluate the fine-tuned model without updating its weights

model.eval()
predictions = []

with torch.no_grad():
    for start in range(0, len(test_subset), 16):
        end = start + 16

        outputs = model(
            input_ids=test_inputs['input_ids'][start:end].to(device),
            attention_mask=test_inputs['attention_mask'][start:end].to(device)
        )

        predictions.extend(
            outputs.logits.argmax(dim=1).cpu().tolist()
        )

true_labels = list(test_subset['label'])

# Calculate the required evaluation metrics

test_results = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
    'Score': [
        accuracy_score(true_labels, predictions),
        precision_score(true_labels, predictions, zero_division=0),
        recall_score(true_labels, predictions, zero_division=0),
        f1_score(true_labels, predictions, zero_division=0)
    ]
})

display(test_results.round(4))


# Task 5.3

# Select a correctly classified sentence pair

correct_index = next(
    i for i in range(len(predictions))
    if predictions[i] == true_labels[i]
)

sentence1 = test_subset['sentence1'][correct_index]
sentence2 = test_subset['sentence2'][correct_index]

words1 = nltk.word_tokenize(sentence1)
words2 = nltk.word_tokenize(sentence2)

inputs = tokenizer(
    words1, words2,
    is_split_into_words=True,
    truncation=True, max_length=128,
    return_tensors='pt'
)

sequence_ids = inputs.sequence_ids()
word_ids = inputs.word_ids()

# Load the original pretrained model for comparison

inputs = {k: v.to(device) for k, v in inputs.items()}

original_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2,
    attn_implementation='eager'
).to(device)

original_model.eval()

# Extract final-layer attention and average across heads

with torch.no_grad():
    original_attention = original_model(
        **inputs, output_attentions=True
    ).attentions[-1].mean(dim=1)[0].cpu()

    fine_tuned_attention = model(
        **inputs, output_attentions=True
    ).attentions[-1].mean(dim=1)[0].cpu()

print("Sentence 1:", sentence1)
print("Sentence 2:", sentence2)


# Task 5.4

tokens = tokenizer.convert_ids_to_tokens(
    inputs['input_ids'][0].cpu().tolist()
)

tagged_words = nltk.pos_tag(words1, tagset='universal')

# Select three nouns as content words from sentence1

content_words = ['Panaon', 'Island', 'Philippines']

content_indices = [
    next(
        i for i, (word, tag) in enumerate(tagged_words)
        if word == target and tag == 'NOUN'
    )
    for target in content_words
]

# Identify the tokens belonging to sentence2

sentence2_indices = [
    i for i, seq in enumerate(sequence_ids)
    if seq == 1
]

alignment_results = []

for word_index in content_indices:
    source_index = next(
        i for i, wid in enumerate(word_ids)
        if sequence_ids[i] == 0 and wid == word_index
    )

    for model_label, attention in [
        ('Original', original_attention),
        ('Fine-tuned', fine_tuned_attention)
    ]:

        # Find the highest-attention token in sentence2

        best_index = max(
            sentence2_indices,
            key=lambda j: attention[source_index, j].item()
        )
        # Exclude punctuation to examine word-to-word alignment

        valid_indices = [
            j for j in sentence2_indices
            if any(char.isalnum() for char in tokens[j])
    ]

        best_word_index = max(
            valid_indices,
            key=lambda j: attention[source_index, j].item()
)

        # Store the alignment results and attention weights

        alignment_results.append({
            'Content Word': words1[word_index],
            'Model': model_label,
            'Highest-Attention Token': tokens[best_index],
            'Attention Weight': round(attention[source_index, best_index].item(), 4),
            'Top Non-Punctuation Token': tokens[best_word_index],
            'Non-Punctuation Weight': round(attention[source_index, best_word_index].item(), 4)
})
alignment_table = pd.DataFrame(alignment_results)
display(alignment_table)
