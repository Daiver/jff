import numpy as np
import cv2
import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

target_size = (128, 128)
#target_size = (224, 224)

#model = ResNet50(input_shape=(target_size[0], target_size[1], 3), weights='imagenet')
model = keras.applications.mobilenet.MobileNet(input_shape=(target_size[0], target_size[1], 3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None)

print 'n params', model.count_params()

#img_path = 'elephant.jpg'
#img_path = '/home/daiver/c2VzcDQrsVI.jpg'
img_path = '14586818_Alt01.jpg'

#img = image.load_img(img_path, target_size=target_size)
#x = image.img_to_array(img)

x = cv2.imread(img_path)
x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
width, height, channels = x.shape
targetDim = height if width > height else width
x = x[:targetDim, :targetDim]
x = cv2.resize(x, target_size).astype(np.float32)

x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
for x in decode_predictions(preds, top=10)[0]:
    print x
