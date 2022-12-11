import time
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from YOLOKiwiFlower.yolov3_tf2.models import YoloV3, YoloV3Tiny
from YOLOKiwiFlower.yolov3_tf2.dataset import transform_images
from YOLOKiwiFlower.yolov3_tf2.utils import draw_outputs
from threading import Thread
from time import sleep



def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    yolo = YoloV3(classes=FLAGS.num_classes)
    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')
    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)
    fps = 0.0
    count = 0

    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count += 1
            if count < 3:
                continue
            else:
                break

        # Resize windows
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)
        
        # Image size (center screen)
        height, width, channels = img.shape  # Esto se puede borrar y dejar la medida fija

        boxes, scores, classes, nums = yolo.predict(img_in)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names, img)

        cv2.imshow('Computer Vision', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

def vc_robotic_arm():
    app.run(main)

