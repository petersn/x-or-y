#!/usr/bin/python2

import os, json, random, argparse
import PIL.Image
import numpy
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

class Example:
    def __init__(self, arr, cls):
        self.arr, self.cls = arr, cls

# This MUST be the same as is used in classify.
input_resolution = [32, 32]

def build_model(config):
    # Load up all the images.
    examples_by_class = {}
    for directory, cls in config["images"].iteritems():
        if cls not in config["classes"]:
            print
            print "You have an error in your JSON model configuration."
            print "Directory %r declared with class %r, which isn't in the declared classes." % (directory, cls)
            print "You declared the classes: %r" % (config["classes"],)
            exit(1)
        paths = os.listdir(directory)
        print 'Reading %i images from "%s" as class "%s"' % (len(paths), directory, cls)
        for path in paths:
            path = os.path.join(directory, path)
            img = PIL.Image.open(path)
            img = img.resize(input_resolution, resample=PIL.Image.BICUBIC)
            arr = numpy.array(img) / 255.0
            assert arr.shape == tuple(input_resolution) + (3,), "Bad image shape: %r" % (arr.shape,)
            if cls not in examples_by_class:
                examples_by_class[cls] = []
            examples_by_class[cls].append(Example(arr, cls))
    # Shuffle each class.
    for examples in examples_by_class.itervalues():
        random.shuffle(examples)
    # Generate training and test sets.
    training = []
    test = []
    print "Producing training and test sets."
    for cls, examples in examples_by_class.iteritems():
        num_test = int(len(examples) * config["test-proportion"])
        training.extend(examples[num_test:])
        test.extend(examples[:num_test])
        print "%16s: %i images (diving as %i training, %i test)" % (cls, len(examples), len(examples[num_test:]), len(examples[:num_test]))
    print "Totals: %i training images, %i test images." % (len(training), len(test))
    # Build the numpy arrays.
    def to_arrays(examples):
        X = numpy.array([example.arr for example in examples])
        Y = numpy.array([
            [cls == example.cls for cls in config["classes"]]
            for example in examples
        ])
        return X, Y
    training_X, training_Y = to_arrays(training)
    test_X,     test_Y     = to_arrays(test)

    # Do the actual training.
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
    net = fully_connected(net, len(config["classes"]), activation="softmax")
    net = regression(net,
        optimizer="adam",
        loss="categorical_crossentropy",
        learning_rate=0.001,
    )
    model = tflearn.DNN(net)
    print "Beginning training."
    # Determine if we have any test images at all.
    # If not, then don't set the validation_set argument on model.fit()
    if len(test_X):
        kwargs = {"validation_set": (test_X, test_Y)}
    else:
        print "Warning: No test images! Continuing with no validation_set."
        kwargs = {}
    model.fit(training_X, training_Y,
        n_epoch=config["epochs"],
        shuffle=True,
        show_metric=True,
        batch_size=config["batch-size"],
        run_id=config["name"],
        **kwargs
    )
    print "Saving model."
    model.save(config["name"] + ".tfl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config.json", help="Configuration for the model to be built.")
    args = parser.parse_args()
    with open(args.__dict__["model_config.json"]) as f:
        config = json.load(f)
    build_model(config)

