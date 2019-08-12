import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, Lambda, Dropout, Activation, SpatialDropout1D, Reshape, GlobalAveragePooling1D, merge, Flatten, Bidirectional, CuDNNGRU, add, Conv1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Model
import model_zoo
import trainer



train_file = r'C:\Users\cheny\Downloads\data\train.csv'
test_file = r'C:\Users\cheny\Downloads\data\test.csv'
embedding_file = r'C:\Users\cheny\Downloads\data\glove.6B.300d.txt'

max_length = 400
max_nb_words = 100000
embedding_dim = 300

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

list_training_sentence = train['comment_text'].fillna('no comment').values
list_test_sentence = test['comment_text'].fillna('no comment').values
toxic_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_train = train[toxic_labels].values

comment = []
for text in list_training_sentence:
    comment.append(text)
test_comment = []
for text in list_test_sentence:
    test_comment.append(text)

tokenizer = Tokenizer(num_words=max_nb_words)
tokenizer.fit_on_texts(comment + test_comment)
sequences = tokenizer.texts_to_sequences(comment)
test_sequences = tokenizer.texts_to_sequences(test_comment)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data = pad_sequences(sequences, maxlen=max_length)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', y_train.shape)

test_data = pad_sequences(test_sequences, maxlen=max_length)
print('Shape of test_data tensor:', test_data.shape)

print('Preparing embedding matrix')
embeddings_index = {}
f = open(embedding_file, encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coeffs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coeffs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

word_index = tokenizer.word_index
num_words = min(max_nb_words, len(word_index))
embedding_matrix = np.zeros((num_words, embedding_dim))
null_count = 0
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i >= max_nb_words:
        continue
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        null_count += 1
print('Null word embeddings: %d' % null_count)


def get_model():
    return model_zoo.get_av_rnn(num_words, embedding_dim, embedding_matrix, max_length, out_size=6)

keras_model_trainer = trainer.KerasModelTrainer(model_stamp='kmax_text_cnn', epoch_num=50, learning_rate=1e-3)

models, val_loss, total_auc, fold_predictions = keras_model_trainer.train_folds(data, y_train, fold_count=10, batch_size=256, get_model_func=get_model)
print("Overall val-loss:", val_loss, "AUC", total_auc)

train_fold_preditcions = np.concatenate(fold_predictions, axis=0)
training_auc = roc_auc_score(y_train[:-1], train_fold_preditcions)
print("Training AUC", training_auc)

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
submit_path_prefix = "results/rnn/nds/fasttext-SC2-nds-randomNoisy-capNet-" + str(max_nb_words) + "-RST-lp-ct-" + str(max_length)

print("Predicting testing results...")
test_predicts_list = []
for fold_id, model in enumerate(models):
    test_predicts = model.predict(test_data, batch_size=256, verbose=1)
    test_predicts_list.append(test_predicts)
    np.save("predict_path/", test_predicts)

test_predicts = np.zeros(test_predicts_list[0].shape)
for fold_predict in test_predicts_list:
    test_predicts += fold_predict
test_predicts /= len(test_predicts_list)

test_ids = test["id"].values
test_ids = test_ids.reshape((len(test_ids), 1))

test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
test_predicts["id"] = test_ids
test_predicts = test_predicts[["id"] + CLASSES]
submit_path = submit_path_prefix + "-L{:4f}-A{:4f}.csv".format(val_loss, total_auc)
test_predicts.to_csv(submit_path, index=False)

print("Predicting training results...")

train_ids = train["id"].values
train_ids = train_ids.reshape((len(train_ids), 1))

train_predicts = pd.DataFrame(data=train_fold_preditcions, columns=CLASSES)
train_predicts["id"] = train_ids
train_predicts = train_predicts[["id"] + CLASSES]
submit_path = submit_path_prefix + "-Train-L{:4f}-A{:4f}.csv".format(val_loss, training_auc)
train_predicts.to_csv(submit_path, index=False)

