from ultralytics import YOLO
import cv2
from datetime import datetime
import os
import json
from aist_systems.utils import _decode

class Watcher2D:
    """
    Class for realtime watching from you camera.
    Where you can choose subjects you are looking at.
    Save logs etc.

    To start using it:
        watcher = aist_systems.watching.Watcher2D()
        watcher.start()
    """
    def __init__(self,
                 yolo_version : str = "yolov8n.pt"):
        """
        :param yolo_version: You can specify version of YOLO (detection model).
        Default = yolov8n.pt
        """
        self.detection_model = YOLO(yolo_version)
        self.classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
        self.model_classes = list(self.classes.keys())
        self.log = {}

    def _data_perf(self, detection_output) -> dict:
        output = detection_output.boxes
        classes = output.cls.tolist()
        bboxes = output.xyxyn.tolist()

        returning_dict = {'classes': classes,
                          'xyxyn_bboxes' : bboxes}
        return returning_dict

    def show_all_classes(self):
        """
        You can look at all available classes and their indexes.
        :return:
        """
        print(*self.classes.items(), sep="\n")

    def show_model_classes(self):
        """
        You can look at classes that detection model will look for.
        :return:
        """
        print(*self.model_classes, sep="\n")

    def choose_classes(self, input_classes : list):
        """
        You can specify classes that detection model will look for.
        :param input_classes: indexes of classes
        (Watch classes and indexes you can with 'show_all_classes' func')
        :return:
        """
        self.model_classes = input_classes

    def save_log(self,
                 path_to_save : str = str(datetime.now()),
                 clear_after_save = False):
        """
        You can write the information about subjects.
        (in 'start' function you can specify to write logs automatically)
        :param path_to_save: Path where logs will be saved
        :param clear_after_save: If you need, you can clear logs from RAM after saving
        :return:
        """
        with open(path_to_save, 'w') as f:
            json.dump(self.log, f)
        if clear_after_save:
            self.log.clear()

    def pred_from_bytes(self,
                        image_bytes : bytes,
                        show : bool = True) -> dict:
        """
        If you have an image in bytes, you can get prediction of the model via this function.
        :param image_bytes: Your image's bytes.
        :param show: If you want to be shown a result of the model - 'True'
        :return: Dict represents classes, bounding boxes, and the time when prediction was made.
        """
        image = _decode(image_bytes=image_bytes)
        output = self.detection_model.predict(image, show=show)[0]
        return self._data_perf(output)

    def start (self,
               show : bool = True,
               cam_index : int = 0,
               write_logs : bool = True,
               save_logs_every : int = 1000):
        """
        Main function of Watcher2D class.
        You can look at detection model's predictions at realtime.
        :param show: True - if you want to look at model results at realtime. False - if you don't.
        :param cam_index: If you have several cameras, you can specify which one you will use.
        :param write_logs: Watcher2D can write information about detection model predictions to JSON files.
        :param save_logs_every: How many times Watcher2D will write predictions to RAM before save it as a file.
        :return:
        """
        saving_lib = str(datetime.now())
        if write_logs:
            os.mkdir(saving_lib)
        camera = cv2.VideoCapture(cam_index)

        while camera.grab():
            response, frame = camera.retrieve()
            if response:
                current_output = self.detection_model.predict(frame, show = show, classes = self.model_classes)[0]
                if write_logs:
                    self.log[str(datetime.now())] = self._data_perf(current_output)
                    # Saving logs
                    if len(self.log.keys()) == save_logs_every:
                        self.save_log(path_to_save=os.path.join(saving_lib, str(datetime.now()) + '.json'),
                                      clear_after_save=True)
        camera.release()
