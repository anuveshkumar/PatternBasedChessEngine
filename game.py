import pickle
maxlen = 50
weight_path = 'discriminate_weights_good1mil50_64_64_6x64.hdf5'
from keras_preprocessing.text import Tokenizer
f = open('tokenized.pkl', 'rb')
tk = pickle.load(f)

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, CuDNNLSTM
import numpy as np

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
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.load_weights(weight_path)

file = open('random_agent_data.txt', 'w')
def find_best_sequence(move_list):
    # print(move_list)
    test = np.array(move_list)
    # print(test)
    score = model.predict(test)
    # print(score)
    best_index = np.argmax(score)
    # print(move_list[best_index])
    return best_index, move_list[best_index]

def find_worst_sequence(move_list):
    # print(move_list)
    test = np.array(move_list)
    # print(test)
    score = model.predict(test)
    # print(score)
    best_index = np.argmin(score)
    # print(move_list[best_index])
    return best_index, move_list[best_index]


AI = 0
Random = 0
import chess
import random
import re
import time
for x in range(100):
    game_sequence = ''
    board = chess.Board()
    # print(board)
    end = 0
    past_moves = [0] * maxlen # maxlen, no. of moves to consider from past
    AIiswb = random.choice(range(2))
    chanceAI = AIiswb

    while not end:
        legal_moves = re.sub(',', '', str(board.legal_moves).split('(')[1][:-2]).split(' ')
        # print(legal_moves)
        past_moves = past_moves[1:]
        # print(past_moves)
        current_move = ''

        if chanceAI == 1:
            possible_seq = []
            if AIiswb == 1:     # white
                for i in range(len(legal_moves)):
                    a = '@' + legal_moves[i].lower()
                    if a in tk.word_index.keys():
                        possible_seq.append(past_moves + [tk.word_index[a]])
                best_move_index, past_moves = find_best_sequence(possible_seq)
            else:               # black
                for i in range(len(legal_moves)):
                    a = '~' + legal_moves[i].lower()
                    if a in tk.word_index.keys():
                        possible_seq.append(past_moves + [tk.word_index[a]])
                best_move_index, past_moves = find_worst_sequence(possible_seq)
            best_move = legal_moves[best_move_index]
            # print("ChessAI: " + best_move)


        else:
            """
            # AI
            possible_seq = []
            for i in range(len(legal_moves)):
                a = '~' + legal_moves[i].lower() in tk.word_index.keys()
                if a:
                    possible_seq.append(past_moves + [tk.word_index['~' + legal_moves[i].lower()]])
            best_move_index, past_moves = find_best_sequence(possible_seq)
            best_move = legal_moves[best_move_index]
            """

            """
            # Human User
            user_move = input("User: Enter move") 
            while user_move not in legal_moves:
                user_move = input("User: Enter a valid move")
            best_move = user_move
            past_moves.append(tk.word_index['~' + legal_moves[i].lower()])
            """

            #Random Agent
            # gotit = False
            # while gotit is False:
            #     best_move = random.choice(legal_moves)
            #     a = '@' + best_move.lower()
            #     if a in tk.word_index.keys():
            #         gotit = True
            #         break
            #     else:
            #         continue
            # past_moves.append(tk.word_index[a])
            best_move = random.choice(legal_moves)
            if AIiswb == 0:  # Random is white
                a = '@' + best_move.lower()
            else:           # Random is black
                a = '~' + best_move.lower()
            if a not in tk.word_index.keys():
                past_moves.append(0)
            else:
                past_moves.append(tk.word_index[a])
            # print("Random: " + best_move)

        game_sequence += best_move + ' '
        board.push_san(best_move)  # find exact function
        # print(board)

        if board.is_game_over():
            if chanceAI == 1:
                end = 1
                AI += 1
                print(str(x) + ". Game Over: AI Wins " + str(AI))
                outcome = 1
            else:
                end = 1
                Random += 1
                print(str(x) + ". Game Over: Random Win " + str(Random))
                outcome = 0
            file.write(game_sequence + str(outcome) + '\n')
            break
        chanceAI = 1 - chanceAI  # sexy as f


print("AI : ", AI)
print("Random: ", Random)