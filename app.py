from flask import Flask, request
import gameclasses
from gameclasses import state_compression

import numpy as np

import tensorflow as tf
from tensorflow import keras

def alpha_zero_loss(y_true, y_pred):
    squared_difference = (y_true[:,256] - y_pred[:,256])**2
    cce = keras.losses.CategoricalCrossentropy()
    cce_loss = cce(y_true[:,:256],y_pred[:,:256])
    return squared_difference+cce_loss

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
model_directory = 'iter.h5'
reconstructed_model = keras.models.load_model(model_directory,custom_objects={ 'alpha_zero_loss': alpha_zero_loss })


def predict(compressed_state):
    prediction = reconstructed_model.predict(np.expand_dims(compressed_state,axis=0))[0]
    return prediction[:256]

@app.route('/',methods = ['POST'])
def handle_request():
    RequestContent = request.get_json()

    legal_moves = RequestContent['legal moves']
    game_state = RequestContent['game state']


    legal_moves_indices = []
    for i in range(len(legal_moves)):
        move = ((legal_moves[i][0][0],legal_moves[i][0][1]),(legal_moves[i][1][0],legal_moves[i][1][1]))
        legal_moves_indices.append(gameclasses.move_to_index[move])

    compressed_game_state = state_compression(game_state)
    p = predict(compressed_game_state)
    p_adjusted = np.copy(p)
    for i in range(len(p_adjusted)): #make illegal moves have 0 probability
        if i not in legal_moves_indices:
            p_adjusted[i] = 0
            p_adjusted = p_adjusted/np.sum(p_adjusted)
    
    most_likely_winning_move = gameclasses.general_moves[np.argmax(p_adjusted)]

    return [[most_likely_winning_move[0][0],most_likely_winning_move[0][1]],[most_likely_winning_move[1][0],most_likely_winning_move[1][1]]]

@app.route('/',methods = ['GET'])
def handle_get():
    return "Hello World"



#if __name__ == "__main__":
    #model_iteration = sys.argv[1]
    #model_directory = 'models\\iter'+model_iteration+'.h5'
    
    #app.run(host="0.0.0.0", port=1000, threaded=True)