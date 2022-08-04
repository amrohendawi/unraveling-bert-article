import itertools

from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
import numpy as np
from sklearn.metrics import pairwise_distances


def get_embeddings(corpus, transformer, fine_tuned=False):

    if fine_tuned:
        #embedder = TextClassifier.load(transformer).document_embeddings
        embedder = TextClassifier.load(transformer).document_embeddings

        exit()
        print()
    else:
        embedder = TransformerDocumentEmbeddings(transformer, layers="all", layer_mean=True)


    #min_class_occ = min([i[1] for i in list(corpus.get_label_distribution().items())])

    all_cor = corpus.get_all_sentences()
    a = all_cor[0]
    embedder.embed(a)

    print()

    res = {int(i.split("label__")[1]): [] for i in corpus.get_label_distribution()}
    #res_labels = {int(i.split("label__")[1]): 0 for i in corpus.get_label_distribution()}

    for i in range(len(all_cor)):
        sent = all_cor[i]
        embedder.embed(sent)
        res[int(all_cor[i].annotation_layers['sentiment'][0].value.split("__")[-1])].append(sent)
        print()
        #res_labels[int(all_cor[i].annotation_layers['sentiment'][0].value.split("__")[-1])] += 1

        #temp_check = [True if res_labels[j] == min_class_occ else False for j in res_labels]
        #if len(list(set(temp_check))) == 1:
        #    if temp_check[0]:
        #        break

    min_extracted_sents = min([len(res[i]) for i in list(res.keys())])

    for i in res:
        res[i] = np.array([sent.embedding.detach().numpy() for sent in res[i]])[:min_extracted_sents]

    return res


corpus: Corpus = CSVClassificationCorpus("./csv_small/",
                                         label_type="sentiment",
                                         column_name_map={0: "label_status", 1: "text"},
                                         delimiter='\t', train_file="train.csv",
                                         dev_file="dev.csv",
                                         test_file="test.csv")


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


#embedder = TextClassifier.load("models/final-model.pt").document_embeddings


#embeddings_bef_ft = get_embeddings(corpus, "bert-base-uncased", fine_tuned=False)

embeddings_aft_ft = get_embeddings(corpus, "models/final-model.pt", fine_tuned=True)

print("Mean distance Matrix before Fine-tuning:\n")
get_distance_matrix(embeddings_bef_ft)
print("-----------------------------\n\n")

print("Mean distance Matrix after Fine-tuning:\n")
get_distance_matrix(embeddings_aft_ft)
print("-----------------------------\n\n")





