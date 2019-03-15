# import the relevant packages
import os
import numpy as np
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras_tqdm import TQDMCallback

# read data, total 50000 records 
df = pd.DataFrame()
df = pd.read_csv('movie_data.csv',encoding='utf-8')
print(df.shape)

# split training and testing set 
X_train = df.loc[:24999, 'review'].values
y_train = df.loc[:24999, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

review_lines = list()
lines = df['review'].values.tolist()

# preprocessing 
for line in lines:
	tokens = word_tokenize(line)
	# convert lowercase
	tokens = [w.lower() for w in tokens]
	# remove punctuation
	table = str.maketrans('','',string.punctuation)
	stripped = [w.translate(table) for w in tokens]
	# remove remaining tokens which are not alphabetic
	words = [word for word in stripped if word.isalpha()]
	# filter out stopwords
	stop_words = set(stopwords.words('english'))
	words = [w for w in words if not w in stop_words]
	review_lines.append(words)
print(len(review_lines))

# # train word2vec
# model = gensim.models.Word2Vec(sentences=review_lines, size=100, window=5, workers=4, min_count=1)

# # vocab size
# words = list(model.wv.vocab)
# print('Vocabulary size: %d' % len(words))

# # save the word vector  
# filename = 'imdb_embed_word2vec.txt'
# model.wv.save_word2vec_format(filename, binary=False)

# load word2vec vector  
embeddings_index = {}
f = open(os.path.join('', 'imdb_embed_word2vec.txt'), encoding='utf-8')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:])
	embeddings_index[word] = coefs
f.close()

# prepare data before being processed in neural network 
total_reviews = X_train + X_test
max_length = max([len(s.split()) for s in total_reviews])
EMBEDDING_DIM = 100

tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(review_lines)
sequences = tokenizer_obj.texts_to_sequences(review_lines)

word_index = tokenizer_obj.word_index
print("Found %s unique tokens." % len(word_index))

review_pad = pad_sequences(sequences, maxlen=max_length)
sentiment = df['sentiment'].values
print("Shape of review tensor: ", review_pad.shape)
print("Shape of sentiment tensor: ", sentiment.shape)

num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word,i in word_index.items():
	if i > num_words:
		continue
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

print(num_words)

# define the neural network's model 
model = Sequential()
embedding_layer = Embedding(num_words, EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix),
	input_length = max_length, trainable=False)
model.add(embedding_layer)
model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# compile the network 
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# print the model summary 
model.summary()

# define the validation split 
VALIDATION_SPLIT = 0.2
indices = np.arange(review_pad.shape[0])
np.random.shuffle(indices)
review_pad = review_pad[indices]
sentiment = sentiment[indices]
num_validation_samples = int(VALIDATION_SPLIT * review_pad.shape[0])

X_train_pad = review_pad[:-num_validation_samples]
y_train = sentiment[:-num_validation_samples]
X_test_pad = review_pad[-num_validation_samples:]
y_test = sentiment[-num_validation_samples:]

# train the network 
model.fit(X_train_pad, y_train, batch_size=128, epochs=3, validation_data=(X_test_pad, y_test), verbose=2, callbacks=[TQDMCallback()])
model.save('modelGRU.h5')