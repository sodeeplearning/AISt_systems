import aist_systems.face as face
import cv2
from datetime import datetime
import os


class Recognizer(face.Recognizer):
    """
    Recognizer for working with several cameras
    """
    def single_thread(self,cameras: list[int],
                      threshold: float = 0.7,
                      write_logs: bool = False,
                      write_logs_every: int = 500,
                      print_logs: bool = True):
        """
        If you have some cameras, you can use them all in this func.
        This function works on only 1 thread,
        So everything works sequentially.

        :param cameras: Specify cameras' indexes.
        :param threshold: confidence threshold.
        :param write_logs: 'True' if you want to write logs.
        :param write_logs_every: How many times logs will be stored in RAM before saving.
        :param print_logs: 'True' if you want to see logs on your console.
        :return:
        """

        assert self.has_faces, "You didn't add any faces"
        devices = [cv2.VideoCapture(cam_ind) for cam_ind in cameras]
        saving_dir = str(datetime.now())

        if write_logs:
            os.mkdir(saving_dir)

        while True:
            for camera_index, current_device in zip(cameras, devices):
                flag, image = current_device.read()

                batch_boxes, cropped_images = self.mtcnn.detect_box(image)

                if cropped_images is not None:
                    for box, cropped in zip(batch_boxes, cropped_images):
                        wrong_person = False
                        img_embedding = self._encode(cropped.unsqueeze(0))
                        detect_dict = {}
                        for k, v in self.all_people_faces.items():
                            detect_dict[k] = (v - img_embedding).norm().item()
                        min_key = min(detect_dict, key=detect_dict.get)

                        if detect_dict[min_key] >= threshold:
                            min_key = 'Wrong person'
                            wrong_person = True

                        if print_logs:
                            if not wrong_person:
                                print(f"Hi, {min_key} (camera {camera_index})")
                            else:
                                print(f"Wrong person detected! (camera {camera_index})")

                        if write_logs:
                            self.log[str(datetime.now())] = min_key + f" (camera {camera_index})"
                            if len(self.log.keys()) == write_logs_every:
                                self.save_log(path_for_saving=os.path.join(saving_dir, str(datetime.now()) + '.json'),
                                              clear_after_saving=True)

    def multy_thread(self):
        pass
