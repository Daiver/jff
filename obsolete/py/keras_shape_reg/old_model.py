
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
def mkPretrainedVGG16Model(nFeats):
    #target_size=(224, 224)
    #baseModel = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    #baseModel = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    baseModel = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    #baseModel = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

    input_img = Input(shape=input_shape)
    x = input_img
    x = baseModel(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(nFeats, activation='linear')(x)

    model = Model(inputs=input_img, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in baseModel.layers:
        layer.trainable = False
    return model

def mkSharedByViewsModel(nFeats, input_shape):
    nInputDims = input_shape[2]
    nViews = nInputDims / 3
    inp = Input(shape=input_shape)
    darkNet = mkDarknetLight(nFeats, (input_shape[0], input_shape[1], 3), includeTop=False)

    x1 = Lambda(lambda x: x[:, :, :, 0:3])(inp)
    x2 = Lambda(lambda x: x[:, :, :, 3:6])(inp)
    x3 = Lambda(lambda x: x[:, :, :, 6:9])(inp)
    x1 = darkNet(x1)
    x2 = darkNet(x2)
    x3 = darkNet(x3)
    x = Concatenate()([x1, x2, x3])
    #x = GlobalAveragePooling2D()(x)
    x = AveragePooling2D(strides=(2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dropout(0.1)(x)
    x = Dense(nFeats, activation='linear')(x)

    model = Model(inputs=inp, outputs=x)
    return model


