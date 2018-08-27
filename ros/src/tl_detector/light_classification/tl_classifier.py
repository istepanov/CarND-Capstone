import os
import tarfile
import urllib

from styx_msgs.msg import TrafficLight

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets


CHECKPOINT_URL = 'https://s3-us-west-2.amazonaws.com/istepanov-ml/vgg_16_traffic_lights/model.tar.gz'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VGG_DIR = os.path.join(SCRIPT_DIR, 'vgg_16')
CHECKPOINT = os.path.join(VGG_DIR, 'model')

VGG_MEAN = [123.68, 116.78, 103.94]

CLASSES = [TrafficLight.GREEN, TrafficLight.RED, TrafficLight.YELLOW]


class TLClassifier(object):
    def __init__(self):
        if not os.path.isfile('{}.meta'.format(CHECKPOINT)):
            print('Downloading the checkpoint ...')
            tar_path = os.path.join(VGG_DIR, 'model.tar.gz')
            urllib.urlretrieve(CHECKPOINT_URL, tar_path)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(VGG_DIR)
            os.remove(tar_path)
            print('Download is complete !')

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_image = tf.placeholder(tf.float32, shape=(600, 800, 3))

            image = tf.image.resize_image_with_crop_or_pad(self.input_image, 224, 224)
            image = image - tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])

            images = tf.expand_dims(image, axis=0)

            vgg = tf.contrib.slim.nets.vgg
            with slim.arg_scope(vgg.vgg_arg_scope()):
                logits, _ = vgg.vgg_16(images, num_classes=3, is_training=False)

            saver = tf.train.Saver()

            self.prediction = tf.nn.softmax(logits)

        self.session = tf.Session(graph=self.graph)

        saver.restore(self.session, CHECKPOINT)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        result = self.session.run(self.prediction, feed_dict={self.input_image: image}).tolist()[0]
        class_index = result.index(max(result))

        return CLASSES[class_index]
