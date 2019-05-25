import os
if __name__ == '__main__':
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    #os.environ["CUDA_VISIBLE_DEVICES"] = ""
    pass

import numpy as np
from matplotlib import pyplot as plt

from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_activation, get_num_filters

import keras


if __name__ == '__main__':
    #modelName = "/home/daiver/coding/jff/py/keras_shape_reg/checkpoints/2017-05-19 12:47:16.562989_ep_149_train_l_0.00872_test_l_0.02065.h5"
    modelName = "/home/daiver/coding/jff/py/keras_shape_reg/checkpoints/2017-05-18 16:43:27.041387_ep_9499_train_l_0.00009_test_l_0.00706.h5"
    model = keras.models.load_model(modelName)
    print model.layers[0].name#conv1
    print model.layers[3].name#conv2
    print model.layers[5].name#conv3
    print model.layers[8].name#conv4
    print model.layers[10].name#conv5
    print model.layers[13].name#conv6
    print model.layers[15].name#conv7
    print model.layers[18].name#conv8
    print model.layers[20].name#conv8
    print model.layers[25].name#conv9
    #exit()
    
    layer_idx = 25
    num_filters = get_num_filters(model.layers[layer_idx])
    print num_filters
    filters = np.arange(get_num_filters(model.layers[layer_idx]))[:32]
    vis_images = []
    for idx in filters:
	img = visualize_activation(model, layer_idx, filter_indices=idx) 
	img = utils.draw_text(img, str(idx))
	vis_images.append(img)

    # Generate stitched image palette with 8 cols.
    stitched = utils.stitch_images(vis_images, cols=8)    
    plt.axis('off')
    plt.imshow(stitched)
    plt.show()
