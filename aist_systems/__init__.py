"""
    AISt - Artificial Intelligence Security. This is the library for security systems made with AI.

    So far, we have these modules:
        1) face - module for face recognition or face unlocking systems.
        2) utils - module with some stuff for systems.
        3) watching - module for surveillance

    You can look at these libraries in folders with corresponding names
"""
import cv2
import numpy as np
from transformers import pipeline
from aist_systems.utils import pil_image_from_bytes
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output


def test_camera(camera_index: int = 0) -> None:
    """
    Before using neural networks you can test your cameras.
    :param camera_index:
    :return:
    """
    cam = cv2.VideoCapture(camera_index)
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("Testing camera, 'Esc' to close", frame)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
    cam.release()
    cv2.destroyAllWindows()


depth_output_type = np.ndarray


class DepthEstimator:
    """Class made for depth estimation."""
    def __init__(self):
        """Init func of DepthEstimator."""
        self.model = pipeline(task="depth-estimation")

    @staticmethod
    def _show_results(output: depth_output_type):
        """Show model's output."""
        clear_output(True)
        plt.imshow(output)
        plt.show()

    @staticmethod
    def _output_perform(model_output: dict) -> depth_output_type:
        """Get performed output"""
        return model_output['predicted_depth'].detach().cpu().permute((1, 2, 0)).numpy()

    def from_pil_image(self, pii_image) -> depth_output_type:
        """Get depth map from PIL Image."""
        return self._output_perform(self.model(pii_image))

    def from_bytes(self, image_bytes: bytes) -> depth_output_type:
        """Get depth map from bytes."""
        return self._output_perform(self.model(pil_image_from_bytes(image_bytes)))

    def from_path(self, path: str) -> depth_output_type:
        """Get depth map from path"""
        return self._output_perform(self.model(Image.open(path)))

    def from_ndarray(self, array: np.ndarray) -> depth_output_type:
        """Get depth map from np.ndarray."""
        return self._output_perform(self.model(Image.fromarray(array)))

    def from_camera(self, cam_ind: int = 0):
        """Get depth map from a camera device (single object)"""
        camera = cv2.VideoCapture(cam_ind)
        while camera.grab():
            flag, frame = camera.retrieve()
            if flag:
                return self.from_ndarray(frame)
        raise "Failed to capture an image!"

    def from_camera_stream(
            self,
            cam_ind: int = 0,
            single_object: bool = False,
            show: bool = True,
            save_data: bool = False,
            max_iter: int = None
            ) -> None | list[depth_output_type]:
        """Get depth map from camera device.

        :param cam_ind: If you have several cameras, you can choose which one you will use.
        :param single_object: If you need just one shot from your camera - True, else - False.
        :param show: 'True' if you want to look at results.
        :param save_data: 'True' if you need to save data to a list.
        :param max_iter: Max num of iterations.
        :return: if you chose to save data with results, you will get a list of results.
        """
        camera = cv2.VideoCapture(cam_ind)
        data_list = []
        current_iter = 0

        while camera.grab():
            flag, frame = camera.retrieve()

            if flag:
                depth_map = self.from_ndarray(frame)
                if show:
                    self._show_results(output=depth_map)
                if save_data:
                    data_list.append(depth_map)

            if single_object:
                break

            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break

            if max_iter is not None:
                current_iter += 1
                if current_iter == max_iter:
                    break

        if save_data:
            return data_list
