import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = open("./tsne_bert_base_cased_sentiment.csv").read().split("\n")[1:][:-1]

epoch_data = [data[0:18000], data[18000:18000*2], data[18000*2:18000*3],
              data[18000*3:18000*4], data[18000*4:18000*5], data[18000*5:18000*6]]

epoch_idx = 0
for epoch in epoch_data:
    #fig = plt.figure(figsize=(24, (4 / 2) * 4))  # each subplot of size 6x6
    #ax = [fig.add_subplot(int(4 / 2), 2, i + 1) for i in range(4)]
    #fig1, ax1 = plt.subplots(nrows=2, ncols=3)

    idx = 0
    for i in [0, 9, 10, 11]:
        curr_layer = epoch[i*1500:1500*(i+1)]
        epochs = list(map(lambda x: x.split(",")[0], curr_layer))
        layer = list(map(lambda x: x.split(",")[1], curr_layer))
        x = np.array(list(map(lambda x: x.split(",")[2], curr_layer)), dtype=float)
        y = np.array(list(map(lambda x: x.split(",")[3], curr_layer)), dtype=float)
        labels = list(map(lambda x: x.split(",")[4], curr_layer))
        df = pd.DataFrame.from_dict(
            {'epoch': epochs,
             'layer': layer,
             'x': x,
             'y': y,
             'label': labels})
        df.label = df.label.astype(int)

        #sns.scatterplot(data=df, x='x', y='y', hue='label', ax=ax[idx])
        #plat = sns.color_palette('colorblind')
        sns.scatterplot(data=df, x='x', y='y', hue='label', style='label', palette='dark')
        plt.legend(["negative", "neutral", "positive"])
        plt.savefig("epoch_"+str(epoch_idx)+"layer_"+str(i+1)+".pdf")
        #plt.savefig("epoch_"+str(epoch_idx)+"llayer_"+str(i+1)+".png")
        plt.title("Epoch: "+str(epoch_idx)+" Layer: "+str(i+1))
        plt.show()
        plt.clf()
        #ax[idx].set_title(f"layer "+str(layer[0]))
        #idx += 1
        #fig.suptitle(f"train_data: epoch {epochs[0]}")

    #plt.savefig("epoch_"+str(epoch_idx)+".pdf")
    #plt.show()
    epoch_idx += 1

    print()


print()