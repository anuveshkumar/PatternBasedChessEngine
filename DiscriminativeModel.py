file = open('dataset.txt', 'r')
games = []
outcomes = []
for line in file.readlines():
    x = line.strip('\n').split(' ')
    games.append(x[:-1])
    outcomes.append(int(x[-1]))

#adding context to dataset


for i in range(len(games)):
    for j in range(0, len(games[i])):
        if j % 2 == 1:
            games[i][j] = '~' + games[i][j]
        else:
            games[i][j] = '@' + games[i][j]

file.close()

# preprocessing
maxlen = 50
print(maxlen)
import pandas as pd
import pickle

data = pd.DataFrame({'game': games, 'outcome': outcomes})
X, y = (data['game'].values, data['outcome'].values)
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
tk = Tokenizer()
tk.fit_on_texts(X)
f = open('tokenized.pkl', 'wb')
pickle.dump(tk, f)
X_seq = tk.texts_to_sequences(X)

# X_pad = pad_sequences(X_seq, maxlen=maxlen, padding='post', truncating='post')
# print(X_seq[0])

# Generating the dataset
import numpy as np
import random
no_games = 0

X_padded = []
Y_padded = []
while no_games < 100:
    this_game = random.randint(0, len(X_seq)-1)
    game = X_seq[this_game]
    # move = random.choice(range(maxlen, len(game) - 1))
    # game_strip = game[move-maxlen: move]
    move = random.choice(range(0, len(game) - 1))     #maxlen

    if move - maxlen < 0:
        game_strip = [0] * maxlen
        for i in range(move + 1):
            game_strip[maxlen - i - 1] = game[move - i]
    else:
        game_strip = game[move - maxlen:move]
    print(str(game_strip) + ' ' + str(y[this_game]))
    X_padded.append(game_strip)
    Y_padded.append(y[this_game])
    no_games += 1


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(np.array(X_padded), np.array(Y_padded), test_size = 0.25, random_state = 1)


batch_size = 64

# Generating the Model
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, CuDNNLSTM, CuDNNGRU
vocabulary_size = len(tk.word_counts.keys())+1
max_words = maxlen
embedding_size = 64
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(CuDNNLSTM(64, return_sequences=True))
model.add(CuDNNLSTM(64, return_sequences=True))
model.add(CuDNNLSTM(64, return_sequences=True))
model.add(CuDNNLSTM(64, return_sequences=True))
model.add(CuDNNLSTM(64, return_sequences=True))
model.add(CuDNNLSTM(64, return_sequences=False))
# model.add(CuDNNLSTM(128, return_sequences=True))
# model.add(CuDNNLSTM(128, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Validation & Early Stopping
from keras.callbacks import ModelCheckpoint, EarlyStopping

filepath = 'discriminate_weights_good1mil50_64_64_6x64.hdf5'
early_stop_checkpoint = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2, min_delta=0)
save_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
callbacks_list = [early_stop_checkpoint, save_checkpoint]

# Training the Model
foo = model.fit(X_train, y_train, validation_split=0.05, batch_size=batch_size, epochs=1, callbacks=callbacks_list)
print(foo)
# Testing the Model
scores = model.evaluate(X_test, y_test, batch_size=128, verbose=0)
print("Test accuracy", scores[1])
