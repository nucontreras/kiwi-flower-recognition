
import threading
from absl import app, flags, logging
from absl.flags import FLAGS
import YOLOKiwiFlower.computer_vision as cv


flags.DEFINE_string('classes', './YOLOKiwiFlower/data/labels/obj.names', 'path to classes file')
flags.DEFINE_string('weights', './YOLOKiwiFlower/weights/yolov3.tf', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', '0', 'path to video file or number for webcam)')
flags.DEFINE_integer('num_classes', 1, 'number of classes in the model')


if __name__ == '__main__':
    try:
        vc_thread = threading.Thread(target = cv.vc_robotic_arm)
        vc_thread.start()

        print("Begin programme\n")

    except SystemExit:
        pass
