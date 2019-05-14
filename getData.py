import re


def get_general_data(stored_game):
    outcomes = []
    games = []
    for i in range(len(stored_game)):
        if stored_game[i][-1] == '0' or stored_game[i][-1] == '1': # or stored_game[i][-1] == '1':
            game = re.sub(r'\d*\. ?', "", stored_game[i][:-3].rsplit(' ', 1)[0])
            outcome = stored_game[i][-3:].split('-')[0]
            outcomes.append(outcome.split('-')[0])
            games.append(game)

    file = open('dataset.txt', 'w+')
    for i in range(len(games)):
        file.write(games[i] + ' ' + outcomes[i] + '\n')

    file.close()

    return(len(games))

def get_white_data(stored_game):
    games_white = []
    for i in range(len(stored_game)):
        if stored_game[i][-1] == '0': # or stored_game[i][-1] == '1':
            game = re.sub(r'\d*\. ?', "", stored_game[i][:-3].rsplit(' ', 1)[0])
            games_white.append(game)

    file_white = open('dataset_white.txt', 'w+')
    for i in range(len(games_white)):
        file_white.write(games_white[i] + '\n')

    file.close()

    return (len(games_white))


def get_black_data(stored_game):
    games_black = []
    for i in range(len(stored_game)):
        if stored_game[i][-1] == '1':
            game = re.sub(r'\d*\. ?', "", stored_game[i][:-3].rsplit(' ', 1)[0])
            games_black.append(game)

    file_black = open('dataset_black.txt', 'w+')
    for i in range(len(games_black)):
        file_black.write(games_black[i] + '\n')
    file.close()

    return (len(games_black))


if __name__ == '__main__':
    file = open('CCRL.pgn', 'r')
    game = ''
    stored_game = []
    for i in range(12500000):
        line = file.readline()
        if line[0] != '\n' and line[0] != '[':  # found game sequence
            game = line
            while True:
                forseq = file.readline()
                if forseq[0] != '[':
                    game += forseq
                else:
                    break
            stored_game.append(game.replace('\n', ' ')[:-2])
            game = ''

    file.close()
    # get_black_data(stored_game)
    # print(get_white_data(stored_game))
    print(get_general_data(stored_game))