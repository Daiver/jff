import keras.backend as K

def weightedBinaryCrossentropy(weights):
    nb_cl = len(weights)
    def f(y_true, y_pred):
        final_mask = y_true * weights[1] + (1.0 - y_true) * weights[0]
        return K.binary_crossentropy(y_true, y_pred) * final_mask
    return f

