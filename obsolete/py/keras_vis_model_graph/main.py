import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import keras

#model = keras.applications.ResNet50(True)
model = keras.applications.VGG16(True)
#print model.summary()
from keras.utils import plot_model
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
plot_model(model, to_file='model2.png', show_shapes=True, show_layer_names=True)
