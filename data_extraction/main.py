import pandas as pd
from openTSNE import TSNE
#from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertForSequenceClassification, BertTokenizerFast, AdamW, logging
logging.set_verbosity_error()
import torch
pd.options.display.max_colwidth = 1000
pd.set_option('display.expand_frame_repr', False)
import re, emoji
import os

torch.manual_seed(42)



"""
srun -K --ntasks=1 --gpus-per-task=1 --cpus-per-gpu=12 --mem=64G -p A100 --container-mounts=/netscratch/$USER:/netscratch/$USER,/home/$USER:/home/$USER,/ds:/ds:ro,`pwd`:`pwd`   --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.04-py3.sqsh   --container-workdir=`pwd`   --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" ./run_binary_doc_class.sh
"""

experiment_name = 'tsne_bert_base_cased_sentiment'

def load_df(text_path, label_path):
    with open(text_path, 'rt') as fi:
        texts = fi.read().strip().split('\n')
    text_dfs = pd.Series(data=texts, name='text', dtype='str')
    labels_dfs = pd.read_csv(label_path, names=['label'], index_col=False).label
    ret_df = pd.concat([text_dfs, labels_dfs], axis=1)
    return ret_df


train_df = load_df('./tweeteval/datasets/sentiment/train_text.txt', './tweeteval/datasets/sentiment/train_labels.txt').head(100)
val_df = load_df('./tweeteval/datasets/sentiment/val_text.txt', './tweeteval/datasets/sentiment/val_labels.txt').head(100)
test_df = load_df('./tweeteval/datasets/sentiment/test_text.txt', './tweeteval/datasets/sentiment/test_labels.txt').head(100)


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
val_df = preprocess_data_df(val_df)
test_df = preprocess_data_df(test_df)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


def get_bert_encoded_data_in_batches(df, batch_size=0, max_seq_length=50):
    global tokenizer
    data = [(row.text, row.label,) for _, row in df.iterrows()]
    sampler = torch.utils.data.sampler.SequentialSampler(data)
    batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                  batch_size=batch_size if batch_size > 0 else len(data),
                                                  drop_last=False)
    for batch in batch_sampler:
        encoded_batch_data = tokenizer.batch_encode_plus([data[i][0] for i in batch], max_length=max_seq_length,
                                                         pad_to_max_length=True, truncation=True)
        seq = torch.tensor(encoded_batch_data['input_ids'])
        mask = torch.tensor(encoded_batch_data['attention_mask'])
        yield (seq, mask), torch.LongTensor([data[i][1] for i in batch])


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# @param {type:"slider", min:0, max:10, step:1}
EPOCHS = 5

# @param [1e-5,5e-5,1e-4] {type:"raw"}
LEARNING_RATE = 0.00005

# @param [8,16,32,64] {type:"raw"}
BATCH_SIZE = 16

# @param [50,100,512] {type:"raw"}
MAX_SEQ_LEN = 50

#dim_reducer = TSNE(n_components=2,
#                   n_iter=1000,
#                   n_jobs=-1,
#                   random_state=42)

dim_reducer = TSNE(n_components=2,
                   perplexity=500,
                   n_iter=1000,
                   n_jobs=-1)

# dim_reducer = PCA(n_components=2)



def visualize_layerwise_embeddings(hidden_states, masks, ys, epoch, title,
                                   all_res,
                                   layers_to_visualize=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]):
    print('visualize_layerwise_embeddings for', title, 'epoch', epoch)
    global dim_reducer

    # !mkdir -p /tmp/plots/{title}

    num_layers = len(layers_to_visualize)
    fig = plt.figure(figsize=(24, (num_layers / 4) * 6))  # each subplot of size 6x6
    ax = [fig.add_subplot(int(num_layers / 4), 4, i + 1) for i in range(num_layers)]
    ys = ys.numpy().reshape(-1)
    for i, layer_i in enumerate(layers_to_visualize):  # range(hidden_states):
        layer_hidden_states = hidden_states[layer_i]
        averaged_layer_hidden_states = torch.div(layer_hidden_states.sum(dim=1), masks.sum(dim=1, keepdim=True))
        layer_dim_reduced_vectors = dim_reducer.fit(averaged_layer_hidden_states.numpy())
        df = pd.DataFrame.from_dict(
            {'epoch': [epoch] * len(ys),
             'layer': [layer_i+1] * len(ys),
             'x': layer_dim_reduced_vectors[:, 0],
             'y': layer_dim_reduced_vectors[:, 1],
             'label': ys})

        df.label = df.label.astype(int)
        if title == 'train_data':
            all_res.append(df)

        sns.scatterplot(data=df, x='x', y='y', hue='label', ax=ax[i])
        fig.suptitle(f"{title}: epoch {epoch}")
        ax[i].set_title(f"layer {layer_i + 1}")

    os.makedirs(f'./plots/{title}/', exist_ok=True)
    plt.savefig(f'./plots/{title}/{epoch}.png', format='png', pad_inches=0)


model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).to(device)


# create empty dataframe to store results
#all_res = pd.DataFrame(columns=['epoch', 'layer', 'x', 'y', 'label'])
all_res = []

loss_function = torch.nn.NLLLoss()
optimizer = AdamW(lr=LEARNING_RATE, params=model.parameters())
# = PlotLosses()
for epoch in range(EPOCHS + 1):
    print(f'epoch = {epoch}')
    logs = {}
    if epoch:  # do not train on 0th epoch, only visualize on it
        model.train(True)  # toggle model in train mode
        train_correct_preds, train_total_preds, train_total_loss = 0, 0, 0.0
        for x, y in get_bert_encoded_data_in_batches(train_df, BATCH_SIZE, MAX_SEQ_LEN):
            model.zero_grad()
            sent_ids, masks = x
            sent_ids = sent_ids.to(device)
            masks = masks.to(device)
            y = y.to(device)
            model_out = model(sent_ids, masks, return_dict=True)
            log_probs = torch.nn.functional.log_softmax(model_out.logits, dim=1)
            loss = loss_function(log_probs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping to prevent exploding gradients
            optimizer.step()

    model.train(False)  # toggle model in eval mode
    with torch.no_grad():
        train_correct_preds, train_total_preds, train_total_loss = 0, 0, 0.0
        train_masks, train_ys = torch.zeros(0, MAX_SEQ_LEN), torch.zeros(0, 1)
        train_hidden_states = None
        for x, y in get_bert_encoded_data_in_batches(train_df, BATCH_SIZE, MAX_SEQ_LEN):
            sent_ids, masks = x
            sent_ids = sent_ids.to(device)
            masks = masks.to(device)
            y = y.to(device)
            model_out = model(sent_ids, masks, output_hidden_states=True, return_dict=True)
            log_probs = torch.nn.functional.log_softmax(model_out.logits, dim=1)
            loss = loss_function(log_probs, y)
            hidden_states = model_out.hidden_states[1:]

            train_total_loss += (loss.detach() * y.shape[0])
            train_preds = torch.argmax(log_probs, dim=1)
            train_correct_preds += (train_preds == y).float().sum()
            train_total_preds += train_preds.shape[0]

            train_masks = torch.cat([train_masks, masks.cpu()])
            train_ys = torch.cat([train_ys, y.cpu().view(-1, 1)])

            if type(train_hidden_states) == type(None):
                train_hidden_states = tuple(layer_hidden_states.cpu() for layer_hidden_states in hidden_states)
            else:
                train_hidden_states = tuple(torch.cat([layer_hidden_state_all, layer_hidden_state_batch.cpu()]) for
                                            layer_hidden_state_all, layer_hidden_state_batch in
                                            zip(train_hidden_states, hidden_states))

        visualize_layerwise_embeddings(train_hidden_states, train_masks, train_ys, epoch, 'train_data', all_res)

        train_acc = train_correct_preds.float() / train_total_preds
        train_loss = train_total_loss / train_total_preds
        logs['loss'] = train_loss.item()
        logs['acc'] = train_acc.item()
        #
        val_correct_preds, val_total_preds, val_total_loss = 0, 0, 0.0
        val_masks, val_ys = torch.zeros(0, MAX_SEQ_LEN), torch.zeros(0, 1)
        val_hidden_states = None
        for x, y in get_bert_encoded_data_in_batches(val_df, BATCH_SIZE, MAX_SEQ_LEN):
            sent_ids, masks = x
            sent_ids = sent_ids.to(device)
            masks = masks.to(device)
            y = y.to(device)
            model_out = model(sent_ids, masks, output_hidden_states=True, return_dict=True)
            log_probs = torch.nn.functional.log_softmax(model_out.logits, dim=1)
            loss = loss_function(log_probs, y)
            hidden_states = model_out.hidden_states[1:]
            # logging logic
            val_total_loss += (loss.detach() * y.shape[0])
            val_preds = torch.argmax(log_probs, dim=1)
            val_correct_preds += (val_preds == y).float().sum()
            val_total_preds += val_preds.shape[0]

            val_masks = torch.cat([val_masks, masks.cpu()])
            val_ys = torch.cat([val_ys, y.cpu().view(-1, 1)])

            if type(val_hidden_states) == type(None):
                val_hidden_states = tuple(layer_hidden_states.cpu() for layer_hidden_states in hidden_states)
            else:
                val_hidden_states = tuple(torch.cat([layer_hidden_state_all, layer_hidden_state_batch.cpu()]) for
                                          layer_hidden_state_all, layer_hidden_state_batch in
                                          zip(val_hidden_states, hidden_states))

        visualize_layerwise_embeddings(val_hidden_states, val_masks, val_ys, epoch, 'val_data', all_res)
        val_acc = val_correct_preds.float() / val_total_preds
        val_loss = val_total_loss / val_total_preds
        logs['val_loss'] = val_loss.item()
        logs['val_acc'] = val_acc.item()
    #if epoch:  # no need to learning-curve plot on 0th epoch
    #    liveloss.update(logs)
    #    liveloss.send()
final_res = pd.concat(all_res, ignore_index=True)
final_res.to_csv(experiment_name+'.csv', index=False)

