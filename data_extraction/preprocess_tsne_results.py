import pandas as pd
import pathlib

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

tsne_dict = {
    "tsne_sentiment": pd.read_csv(DATA_PATH.joinpath("tsne_bert_base_cased_sentiment.csv")),
    "tsne_offensive": pd.read_csv(DATA_PATH.joinpath("tsne_bert_base_cased_offensive.csv")),
    "tsne_hate": pd.read_csv(DATA_PATH.joinpath("tsne_bert_base_cased_hate.csv")),
}

# for every entry in tsne_dict calculate the average and standard deviation of the columns x and y in the dataframe
def get_tsne_stats(df):
    return {
            'avg_x': df['x'].mean(),
            'avg_y': df['y'].mean(),
            'std_x': df['x'].std(),
            'std_y': df['y'].std(),
        }

# remove all points that are above the mean + 2*std
def remove_outliers(df, stats):
    df = df.drop(df[(abs(df['x']) > 1.5*(abs(stats['avg_x']) + stats['std_x'])) | (abs(df['y']) > 1.5*(abs(stats['avg_y']) + stats['std_y']))].index)
    # multiply the y axis by the difference std[x]/std[y]
    df['y'] = df['y'] * (stats['std_x'] / stats['std_y'])
    return df

# ensure equal distribution of the data by the label column
def equal_distribution(df, label):
    # count the number of occurences of each label in the label column
    labels_occurences = df[label].value_counts()
    # split the dataframe into dataframes by the label column
    print(labels_occurences)
    df_by_label = {}
    for l in labels_occurences.index:
        print(l)
        df_by_label[l] = df[df[label] == l]
    # truncate each dataframe to the minimum labels_occurences
    for l in df_by_label:
        min = labels_occurences.min()
        df_by_label[l] = df_by_label[l].sample(n=min//2)
    # concatenate the dataframes
    df = pd.concat(df_by_label.values())
    return df

# call remove_outliers function on all dataframes in the tsne_dict
for key, value in tsne_dict.items():
    # split every tsne_dict by epoch, then by layer
    epoch_dict = {}
    for epoch in value['epoch'].unique():
        epoch_dict[epoch] = value[value['epoch'] == epoch]
        # split the dataframe by layer
        layer_dict = {}
        for layer in epoch_dict[epoch]['layer'].unique():
            layer_dict[layer] = epoch_dict[epoch][epoch_dict[epoch]['layer'] == layer]
            # remove outliers
            # print layer_dict type
            print(type(layer_dict[layer]))
            # print layer_dict first 5 rows
            print(layer_dict[layer].head())
            layer_dict[layer] = remove_outliers(layer_dict[layer], get_tsne_stats(layer_dict[layer]))
            # equalize the distribution of the data by the label column
            layer_dict[layer] = equal_distribution(layer_dict[layer], 'label')
        
        # concatenate the dataframes
        epoch_dict[epoch] = pd.concat(layer_dict.values())
    
    # concatenate the dataframes
    tsne_dict[key] = pd.concat(epoch_dict.values())
    
# save the new dataframes to csv files
def save_to_csv(dict, folder):
    # if the folder isnt created yet, create it
    if not DATA_PATH.joinpath(folder).exists():
        DATA_PATH.joinpath(folder).mkdir()
        
    for key, value in dict.items():
        # save dataframes without index column
        value.to_csv(DATA_PATH.joinpath(folder + "/" + key + ".csv"), index=False)

# call save_to_csv on all tsne_dict dataframes
save_to_csv(tsne_dict, "preprocess_tsne_data")