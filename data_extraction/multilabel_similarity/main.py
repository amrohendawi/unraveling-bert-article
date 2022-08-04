import pandas as pd
import re, emoji
from flair.trainers import ModelTrainer
import itertools
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
import numpy as np
from sklearn.metrics import pairwise_distances



def load_df(text_path, label_path):
    with open(text_path, 'rt') as fi:
        texts = fi.read().strip().split('\n')
    text_dfs = pd.Series(data=texts, name='text', dtype='str')
    labels_dfs = pd.read_csv(label_path, names=['label'], index_col=False).label
    ret_df = pd.concat([text_dfs, labels_dfs], axis=1)
    return ret_df


train_df = load_df('../tweeteval/datasets/sentiment/train_text.txt', '../tweeteval/datasets/sentiment/train_labels.txt').head(150)
val_df = load_df('../tweeteval/datasets/sentiment/val_text.txt', '../tweeteval/datasets/sentiment/val_labels.txt').head(150)
test_df = load_df('../tweeteval/datasets/sentiment/test_text.txt', '../tweeteval/datasets/sentiment/test_labels.txt').head(150)


def encode_urls(row):
    row.text = re.sub(
        r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))",
        "HTTPURL", row.text)
    return row


def encode_mentions_hashtags(row):
    row.text = row.text.replace('@', ' @')
    row.text = re.sub(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)", "@USER", row.text)
    row.text = row.text.replace('#', ' ')
    return row


def encode_emojis(row):
    row.text = emoji.demojize(row.text)
    return row


def remove_extra_spaces(row):
    row.text = ' '.join(row.text.split())
    return row


def lower_text(row):
    row.text = row.text.lower()
    return row


def preprocess_data_df(df):
    df = df.apply(encode_urls, axis=1)
    df = df.apply(encode_mentions_hashtags, axis=1)
    df = df.apply(encode_emojis, axis=1)
    df = df.apply(remove_extra_spaces, axis=1)
    df = df.apply(lower_text, axis=1)
    return df


train_df = preprocess_data_df(train_df)
train_df['label'] = '__label__' + train_df['label'].astype(str)
train_df = train_df[train_df.columns[::-1]]
train_df.to_csv("./csv_small/train.csv", sep='\t', index=False, header=False)

val_df = preprocess_data_df(val_df)
val_df['label'] = '__label__' + val_df['label'].astype(str)
val_df = val_df[val_df.columns[::-1]]
val_df.to_csv("./csv_small/dev.csv", sep='\t', index=False, header=False)


test_df = preprocess_data_df(test_df)
test_df['label'] = '__label__' + test_df['label'].astype(str)
test_df = test_df[test_df.columns[::-1]]
test_df.to_csv("./csv_small/test.csv", sep='\t', index=False, header=False)



# load corpus containing training, test and dev data and if CSV has a header, you can skip it
corpus: Corpus = CSVClassificationCorpus("./csv_small/", label_type="sentiment",
                                         column_name_map={0: "label_status", 1: "text"},
                                         delimiter='\t', train_file="train.csv", dev_file="dev.csv",
                                         test_file="test.csv")


label_dict_csv = corpus.make_label_dictionary("sentiment")

document_embeddings = TransformerDocumentEmbeddings('bert-base-uncased', fine_tune=True)

classifier = TextClassifier(document_embeddings, label_type="sentiment", label_dictionary=label_dict_csv)

trainer = ModelTrainer(classifier, corpus)

trainer.fine_tune("./models/", learning_rate=0.00005, mini_batch_size=5, max_epochs=4,
                  train_with_dev=False, monitor_train=True, monitor_test=True)


def get_embeddings(corpus, transformer, fine_tuned=False):

    if fine_tuned:
        embedder = TextClassifier.load(transformer).document_embeddings
    else:
        embedder = TransformerDocumentEmbeddings(transformer)

    all_cor = corpus.get_all_sentences()

    res = {int(i.split("label__")[1]): [] for i in corpus.get_label_distribution()}

    for i in range(len(all_cor)):
        sent = all_cor[i]
        embedder.embed(sent)
        res[int(all_cor[i].annotation_layers['sentiment'][0].value.split("__")[-1])].append(sent)

    min_extracted_sents = min([len(res[i]) for i in list(res.keys())])

    for i in res:
        res[i] = np.array([sent.embedding.detach().numpy() for sent in res[i]])[:min_extracted_sents]

    return res


def get_distance_matrix(class_embeddings):
    for res_comb in sorted([(i, i) for i in range(len(class_embeddings))] + list(
            itertools.combinations(range(len(class_embeddings)), 2)), key=lambda x: x[0] + x[1]):
        class_o = class_embeddings[res_comb[0]]
        class_i = class_embeddings[res_comb[1]]

        pdistances_euc = pairwise_distances(class_o, class_i, n_jobs=-1)
        mean_dist_euc = pdistances_euc.mean()

        print("Euclidean distance between classes {} and {}: {}".format(res_comb[0], res_comb[1], mean_dist_euc))

    for res_comb in sorted([(i, i) for i in range(len(class_embeddings))] + list(
            itertools.combinations(range(len(class_embeddings)), 2)), key=lambda x: x[0] + x[1]):
        class_o = class_embeddings[res_comb[0]]
        class_i = class_embeddings[res_comb[1]]

        pdistances_cosine = pairwise_distances(class_o, class_i, metric='cosine', n_jobs=-1)
        mean_dist_cosine = pdistances_cosine.mean()
        print("Cosine distance between classes {} and {}: {}".format(res_comb[0], res_comb[1], mean_dist_cosine))


embeddings_bef_ft = get_embeddings(corpus, "bert-base-uncased", fine_tuned=False)

embeddings_aft_ft = get_embeddings(corpus, "models/final-model.pt", fine_tuned=True)

print("Mean distance Matrix before Fine-tuning:\n")
get_distance_matrix(embeddings_bef_ft)
print("-----------------------------\n\n")

print("Mean distance Matrix after Fine-tuning:\n")
get_distance_matrix(embeddings_aft_ft)
print("-----------------------------\n\n")
