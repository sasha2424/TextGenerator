# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = file = open(filename, encoding="utf8").read()
raw_text = raw_text.lower()
word_text = raw_text.split(" ")
# create mapping of unique chars to integers
words = sorted(list(set(word_text)))
word_to_int = dict((c, i) for i, c in enumerate(words))
int_to_word = dict((i,c) for i, c in enumerate(words))
# summarize the loaded data
n_words = len(word_text)
n_vocab = len(words)
print("Total Characters: ", n_words)
print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_words - seq_length, 1):
	seq_in = word_text[i:i + seq_length]
	seq_out = word_text[i + seq_length]
	dataX.append([word_to_int[word] for word in seq_in])
	dataY.append(word_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
# load the network weights
filename = "weights-improvement-18-6.1468.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Seed:")
print ("\"", ''.join([(int_to_word[value] + " ") for value in pattern]), "\"")
# generate characters
text = ''.join([int_to_word[value] for value in pattern]) + " "
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_word[index]
	text = text + result + " "
	seq_in = [int_to_word[value] for value in pattern]
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
text_file = open("Output.txt", "w")
text_file.write(text)
text_file.close()
print("Done")
