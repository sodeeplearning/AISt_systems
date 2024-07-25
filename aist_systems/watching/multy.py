import cv2
import aist_systems.watching as watching
from aist_systems.utils import only_digits
from datetime import datetime
import os


class Watcher2D(watching.Watcher2D):
    """
    If you have several cameras, you this class to you them.
    This class is an addition to a Watcher2D
    """
    @staticmethod
    def _data_perf(detection_output,
                   camera_index: int = 0) -> dict:
        output = detection_output.boxes
        classes = output.cls.tolist()
        bboxes = output.xyxyn.tolist()

        returning_dict = {'classes': classes,
                          'xyxyn_bboxes': bboxes,
                          'camera': camera_index}
        return returning_dict

    def single_thread(self,
                      cameras: list[int],
                      show: bool = False,
                      write_logs: bool = True,
                      save_logs_every: int = 500,
                      threshold: float = 0.5,
                      use_cuda=False):
        """Use this function if you have several cameras,
        but you need to use just 1 thread.

        :param cameras: Specify which cameras you will use.
        :param show: True - if you want to look at model results at realtime. False - if you don't.
        :param write_logs: Watcher2D can write information about detection model predictions to JSON files.
        :param save_logs_every: How many times Watcher2D will write predictions to RAM before save it as a file.
        :param use_cuda: if ypu have a GPU, you can specify it in this param.
        :param threshold: you can specify threshold meaning model's confidence.
        :return:
        """
        if use_cuda:
            self.detection_model.cuda()
        saving_lib = only_digits(str(datetime.now()))
        if write_logs:
            os.mkdir(saving_lib)

        devices = [cv2.VideoCapture(current_camera) for current_camera in cameras]

        while True:
            for camera_ind, current_device in zip(cameras, devices):
                response, frame = current_device.read()
                if response:
                    current_output = self.detection_model.predict(frame,
                                                                  show=show,
                                                                  classes=self.model_classes,
                                                                  conf=threshold)[0]
                    if write_logs:
                        self.log[str(datetime.now())] = self._data_perf(current_output, camera_index=camera_ind)
                        # Saving logs
                        if len(self.log.keys()) == save_logs_every:
                            self.save_log(
                                path_to_save=os.path.join(saving_lib, only_digits(str(datetime.now())) + '.json'),
                                clear_after_save=True)

    def multy_thread(self):
        pass
