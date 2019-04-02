import os
import time
import logging

import cv2
import numpy as np
import tensorflow as tf

from nets import model_train as model
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector


def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


class TextDetectionPredict:
    def __init__(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = "checkpoints_mlt/"
        if not os.path.exists(checkpoint_path):
            raise ValueError("checkpoint_path {} does not exist".format(checkpoint_path))
        self.checkpoint_path = checkpoint_path
        self.input_image = None
        self.input_im_info = None
        self.bbox_pred = None
        self.cls_pred = None
        self.cls_prob = None
        self.session = None
        # self.load_model_from_checkpoint()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session is not None:
            self.session.close()

    def load_model_from_checkpoint(self):

        checkpoint_path = self.checkpoint_path

        with tf.get_default_graph().as_default():
            input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
            input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            bbox_pred, cls_pred, cls_prob = model.model(input_image)

            variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
            saver = tf.train.Saver(variable_averages.variables_to_restore())

            if self.session is None:
                self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            sess = self.session
            with sess.as_default():
                ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
                model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
                print('Restore from {}'.format(model_path))
                saver.restore(sess, model_path)
            print("load model finished....")

            self.input_image = input_image
            self.input_im_info = input_im_info
            self.bbox_pred = bbox_pred
            self.cls_pred = cls_pred
            self.cls_prob = cls_prob

    def predict_by_image_files(self, im_fn_list):

        logger = logging.getLogger(__name__)
        bbox_pred = self.bbox_pred
        cls_prob = self.cls_prob
        input_image = self.input_image
        input_im_info = self.input_im_info

        result_items = []

        sess = self.session
        with sess.as_default():
            for im_fn in im_fn_list:
                start = time.time()
                try:
                    im = cv2.imread(im_fn)[:, :, ::-1]
                except Exception as e:
                    logger.error("Error reading image {}!".format(im_fn), e)
                    continue

                img, (rh, rw) = resize_image(im)
                h, w, c = img.shape
                im_info = np.array([h, w, c]).reshape([1, 3])
                bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                       feed_dict={input_image: [img],
                                                                  input_im_info: im_info})

                textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
                scores = textsegs[:, 0]
                textsegs = textsegs[:, 1:5]

                textdetector = TextDetector(DETECT_MODE='H')
                boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
                boxes = np.array(boxes, dtype=np.int)

                cost_time = (time.time() - start)
                logger.debug("image {} cost time: {:.2f}s".format(im_fn, cost_time))

                arr1 = []
                for i, box in enumerate(boxes):
                    arr1.append(dict(
                        box=box[:8].tolist(),
                        score=scores[i].item()
                    ))
                result_items.append(dict(
                    image_file=im_fn,
                    boxes=arr1))

        return result_items


def test101(model_path):
    import json
    predictor = TextDetectionPredict(model_path)
    predictor.load_model_from_checkpoint()
    print(predictor)
    base_dir = os.path.dirname(os.path.dirname(__file__))
    file_lists = ["006.jpg", "007.jpg", "009.jpg", "010.png"]
    demo_dir = os.path.join(base_dir, "data/demo")
    file_lists = [os.path.join(demo_dir, item) for item in file_lists]
    print(file_lists)
    results = predictor.predict_by_image_files(file_lists)
    print(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    import sys
    test101(sys.argv[1])
