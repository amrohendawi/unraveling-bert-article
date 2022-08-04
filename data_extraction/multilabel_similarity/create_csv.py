import pandas as pd
import re, emoji


def load_df(text_path, label_path):
    with open(text_path, 'rt') as fi:
        texts = fi.read().strip().split('\n')
    text_dfs = pd.Series(data=texts, name='text', dtype='str')
    labels_dfs = pd.read_csv(label_path, names=['label'], index_col=False).label
    ret_df = pd.concat([text_dfs, labels_dfs], axis=1)
    return ret_df


train_df = load_df('../tweeteval/datasets/sentiment/train_text.txt', '../tweeteval/datasets/sentiment/train_labels.txt').head(2000)
val_df = load_df('../tweeteval/datasets/sentiment/val_text.txt', '../tweeteval/datasets/sentiment/val_labels.txt').head(2000)
test_df = load_df('../tweeteval/datasets/sentiment/test_text.txt', '../tweeteval/datasets/sentiment/test_labels.txt').head(2000)


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


