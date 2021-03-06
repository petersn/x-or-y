# x_or_y

Do you ever have a hard time telling things apart?
Have you tried looking at thousands of pictures of trees and cars, but you still sometimes get them confused?
Now you never have to again, because thanks to Deep Learning your computer can sort through these images for you.

`x_or_y` is a brain dead simple tool for building classifiers to distinguish between things.
To use `x_or_y` simply think up a set of things you'd like to distinguish:

    $ x_or_y "tree" "car"
    ... lots and lots of processing...
    $ ./tree_OR_car /absolute/path/picture_of_tree.jpeg
    ... more processing...
    RESULTS:
        #1: [ 98.31%] tree
        #2: [  1.69%] car

That's all there is to it.

## How?

`x_or_y` simply scrapes Bing images (TODO: Check if this is kosher...) for 1,000 images of each class, and then trains a simple convolutional net on them.
Despite the dichotomous name, you may have as many different classes as you want.

TODO: Document all the issues and bugs. For example, currently `makeself` cds into the temp dir before running my script, so paths input to the resultant binary have to be absolute. :/

## Dependencies

This project has a lot of dependencies. For Ubuntu/Debian users:

    # Install the Python dependencies.
    pip install tensorflow
    pip install tflearn
    pip install PIL
    pip install numpy
    
    # Install additional dependencies.
    apt-get install makeself
    apt-get install nodejs

    # Finally, install scraper.js's dependencies.
    # For those unfamiliar with npm, this must be run from
	# the root of the repo, so npm can find package.json.
    npm install

If you cannot install `PIL` via pip and you're on Ubuntu/Debian, then `sudo apt-get install python-pil` might work for you.

If you run `x_to_y` and it crashes while "Scraping images" then you might have forgotten to install `scraper.js`'s dependencies by running `npm install` (installs ~150 MiB of dependencies.).

## License

This entire project is Public Domain -- do whatever you want with it.

## Inspirations

Inspired by (and blatantly rips off the network architecture of) the infamous [r_u_a_bird.py](http://blog.bitfusion.io/2016/08/31/training-a-bird-classifier-with-tensorflow-and-tflearn) which in turn takes the network architecture from a [tflearn example](https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py).
