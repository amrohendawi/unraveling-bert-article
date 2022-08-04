from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus


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



