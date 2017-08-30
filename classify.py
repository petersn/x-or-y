#!/usr/bin/python2
"""
classify.py

Example usage of a model built with build_model.py
"""

import json
import PIL.Image
import numpy
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

BLUE  = "\033[94m"
GREEN = "\033[92m"
RED   = "\033[91m"
ENDC  = "\033[0m"

# This MUST be the same as is used in build_model.
input_resolution = [32, 32]

class Classifier:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)
        img_prep = ImagePreprocessing()
        img_prep.add_featurewise_zero_center()
        img_prep.add_featurewise_stdnorm()
        img_aug = ImageAugmentation()
        img_aug.add_random_flip_leftright()
        img_aug.add_random_rotation(max_angle=25.)
        img_aug.add_random_blur(sigma_max=3.)
        net = input_data(
            shape=[None] + list(input_resolution) + [3],
            data_preprocessing=img_prep,
            data_augmentation=img_aug,
        )
        net = conv_2d(net, 32, 3, activation="relu")
        net = max_pool_2d(net, 2)
        net = conv_2d(net, 64, 3, activation="relu")
        net = conv_2d(net, 64, 3, activation="relu")
        net = max_pool_2d(net, 2)
        net = fully_connected(net, 512, activation="relu")
        net = dropout(net, 0.5)
        net = fully_connected(net, len(self.config["classes"]), activation="softmax")
        self.net = regression(net,
            optimizer="adam",
            loss="categorical_crossentropy",
            learning_rate=0.001,
        )
        self.model = tflearn.DNN(self.net)
        self.model.load(self.config["name"] + ".tfl")

    def classify(self, image, argmax=False):
        if isinstance(image, str):
            image = PIL.Image.open(image)
        if image.size != tuple(input_resolution):
            image = image.resize(input_resolution, resample=PIL.Image.BICUBIC)
        arr = numpy.array(image) / 255.0
        prediction = self.model.predict([arr])
        if argmax:
            return numpy.argmax(prediction[0])
        return dict(zip(self.config["classes"], prediction[0]))

if __name__ == "__main__":
    import argparse, pprint
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config.json", help="Configuration file for built model.")
    parser.add_argument("input_image.xyz", help="Image to classify.")
    args = parser.parse_args()
    dnn = Classifier(args.__dict__["model_config.json"])
    result = dnn.classify(args.__dict__["input_image.xyz"])
    print
    print BLUE + "RESULTS:" + ENDC
    sorted_result = sorted(result.items(), key=lambda (cls, prob): -prob)
    for i, (cls, prob) in enumerate(sorted_result):
        print "    #%i: [%6.2f%%] %s" % (i+1, prob * 100.0, cls)

