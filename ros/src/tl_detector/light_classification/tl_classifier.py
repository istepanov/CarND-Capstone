import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import tarfile
import urllib
import glob

from styx_msgs.msg import TrafficLight
from darkflow.net.build import TFNet


CHECKPOINT_FOLDER = os.path.join(SCRIPT_DIR, 'ckpt')
CHECKPOINT = os.path.join(CHECKPOINT_FOLDER, 'yolov2-tiny-traffic-lights-*')
CHECKPOINT_URL = 'https://s3-us-west-2.amazonaws.com/istepanov-ml/darkflow-traffic-lights/model.tar.gz'


class TLClassifier(object):
    def __init__(self):
        if not glob.glob(CHECKPOINT):
            print('Downloading the checkpoint ...')
            tar_path = os.path.join(CHECKPOINT_FOLDER, 'model.tar.gz')
            urllib.urlretrieve(CHECKPOINT_URL, tar_path)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(CHECKPOINT_FOLDER)
            os.remove(tar_path)
            print('Download is complete !')

        options = {
            'model': os.path.join(SCRIPT_DIR, 'cfg/yolov2-tiny-traffic-lights.cfg'),
            'labels': os.path.join(SCRIPT_DIR, 'labels.txt'),
            'backup': CHECKPOINT_FOLDER,
            'load': -1,
            'threshold': 0.5,
        }
        self.tfnet = TFNet(options)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        result = self.tfnet.return_predict(image)
        if not result:
            return TrafficLight.UNKNOWN

        item = max(result, key=lambda x: x['confidence'])
        if item['label'] == 'red':
            return TrafficLight.RED
        elif item['label'] == 'yellow':
            return TrafficLight.YELLOW
        elif item['label'] == 'green':
            return TrafficLight.GREEN
        else:
            assert False
