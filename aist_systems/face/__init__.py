import os
import cv2
import requests
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from types import MethodType
from time import sleep
from aist_systems.utils import load, save, decode, get_hash
from datetime import datetime
import json


def _detect_box(self, img, save_path=None):
    # Detect faces
    batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
    # Select faces
    if not self.keep_all:
        batch_boxes, batch_probs, batch_points = self.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=self.selection_method
        )
    # Extract faces
    faces = self.extract(img, batch_boxes, save_path)
    return batch_boxes, faces


class Recognizer:
    """
    Class made for face recognition. To use it:
        face_recognizer = aist_systems.face.Recognizer()
        face_recognizer.add_face()
        face_recognizer.launch()

    If you used it and want to save your config with your saved face:
        face_recognizer.save_core(<path for saving>)

    If you want to load the config:
        face_recognizer.load_core(<path for loading>)

    Or if you have an url of core file:
        face_recognizer.load_core_from_url(<url to core file>)

    Hope that You will enjoy using our lib :)
    """
    def __init__(self, path_to_dict: str = None):
        """
        Initializing func.
        :param path_to_dict: path to your config if you used it before. You can always load it later.
        """
        self.log = {}
        self.has_faces = False

        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.mtcnn = MTCNN(
            image_size=224, keep_all=True, thresholds=[0.4, 0.5, 0.5], min_face_size=60
        )
        self.mtcnn.detect_box = MethodType(_detect_box, self.mtcnn)

        if path_to_dict is not None:
            self.all_people_faces = load(path_to_dict)
        else:
            self.load_core_from_url(url="https://storage.yandexcloud.net/facecore/core.pkl",
                                    name_for_file="core")

    def _encode(self, img):
        res = self.resnet(torch.Tensor(img))
        return res

    def save_log(self,
                 path_for_saving: str,
                 clear_after_saving: bool = False):
        """
        You can save information about persons that recognition model detected
        :param path_for_saving: path for saving file
        :param clear_after_saving: if you need, you can clear log files from RAM after saving.
        :return:
        """
        with open(path_for_saving, 'w') as f:
            json.dump(self.log, f)

        if clear_after_saving:
            self.log.clear()

    def _take_photo(self,
                    cam: int = 0):
        cam = cv2.VideoCapture(cam)
        is_taken = False
        received_image = None

        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("Press 'Space' to take a photo, 'Esc' to close", frame)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                received_image = frame
                sleep(0.5)
                is_taken = True
                break

        cam.release()
        cv2.destroyAllWindows()
        if is_taken:
            while True:
                cv2.imshow("Press 'Space' to reshoot, 'Esc' to confirm", received_image)
                k = cv2.waitKey(1)
                if k % 256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    cv2.destroyAllWindows()
                    break
                elif k % 256 == 32:
                    # SPACE pressed
                    return self._take_photo()
            print("You took a photo of your face")
            return received_image
        return None

    def load_core(self, path_to_core: str):
        """
        If you used Recognizer before,
        you can load your config file via this function specifying path to your config file.
        If you have an url address of your config file. You should use 'Recognizer.load_core_from_url'.
        :param path_to_core: path to your config file
        :return:
        """
        loaded_object = load(path_to_core)
        if type(loaded_object) is dict:
            self.all_people_faces = loaded_object
        else:
            print(f"The object can't be converted to a core (Need dict, got {type(loaded_object)})")

    def load_core_from_url(self,
                           url: str,
                           name_for_file: str):
        """
        If you used Recognizer before and saved your config file on a web page,
        you can load it via this function specifying direct link to this file.
        :param url: url (direct link) to your config file
        :param name_for_file: path to loading core.
        :return:
        """
        with open(f"{name_for_file}.pkl", 'wb') as f:
            f.write(requests.get(url).content)
        self.all_people_faces = load(f"{name_for_file}.pkl")

    def save_core(self, path):
        """
        If you added some faces and need to save them to use it next time,
        you can do it via this function.
        :param path: path where file will be saved.
        :return:
        """
        save(self.all_people_faces, path)

    def add_face(self,
                 name=None,
                 camera_index: int = 0):
        """
        Adding face to recognize it. When yoy will launch this function,
        you will have a window with your camera filming you.

        To make a photo just press 'SPACE'.
        Or if you have to exit press 'ESC'.

        After shooting a photo you have an opportunity to reshoot it pressing 'SPACE'
        If you won't make a photo, nothing will be added to a Recognizer.

        :param name: You can add a name to the user. Default: User <N>
        :param camera_index: If you have several cameras, you can specify which one you will use.
        :return:
        """
        if name is None:
            name = f"User {len(self.all_people_faces.keys())}"

        received_image = self._take_photo(cam=camera_index)
        if type(received_image) is not type(None):
            batch_boxes, cropped_image = self.mtcnn.detect_box(received_image)
            if cropped_image is not None:
                img_embedding = self._encode(cropped_image)
                self.all_people_faces[name] = img_embedding
                self.has_faces = True
            else:
                print("Failed to add face")

    def add_face_from_image(self,
                            path_to_image: str,
                            name: str = None):
        """
        If you have no opportunities to take a photo of your face,
        You can load the image specifying path to it.
        :param path_to_image: path to the image of your face.
        :param name: if you want to add a name to the user, you can specify it.
        Default : User <N>
        :return:
        """
        if name is None:
            name = f"User {len(self.all_people_faces.keys())}"

        current_image = cv2.imread(path_to_image)
        cropped_image = self.mtcnn(current_image)
        if cropped_image is not None:
            self.all_people_faces[name] = self._encode(cropped_image).squeeze()

    def predict_from_bytes(self,
                           image_bytes: bytes,
                           threshold: float = 0.7) -> str:
        """
        If you have an image in bytes, you can get prediction of the model via this function.
        :param image_bytes: Image in bytes.
        :param threshold: Confidence threshold. Less = more confidence
        :return: Returns the most similar User to the image,
        Or 'Wrong person' if detected someone wrong,
        Or returns 'No one was detected'.
        """
        image = decode(image_bytes=image_bytes)
        batch_boxes, cropped_images = self.mtcnn.detect_box(image)
        min_key = "No one was detected"

        if cropped_images is not None:
            for box, cropped in zip(batch_boxes, cropped_images):
                img_embedding = self._encode(cropped.unsqueeze(0))
                detect_dict = {}
                for k, v in self.all_people_faces.items():
                    detect_dict[k] = (v - img_embedding).norm().item()
                min_key = min(detect_dict, key=detect_dict.get)
                if detect_dict[min_key] >= threshold:
                    min_key = 'Wrong person'
        return min_key

    def launch(self,
               cam: int = 0,
               threshold: float = 0.7,
               stop_when_rec: bool = False,
               write_logs: bool = False,
               write_logs_every: int = 500):
        """
        Launch recognizer.
        :param cam: if you have several cameras, you can specify which one you will use.
        :param threshold: confidence threshold: less = more strict
        :param stop_when_rec: if Recognizer detected right person, it can stop using camera.
        :param write_logs: If you want to save detection model's predictions with its time, choose True
        :param write_logs_every: How many times model have to save her predictions to the RAM,
        before the log will be saved as a file.
        :return:
        """
        assert self.has_faces, "You didn't add any faces"
        vdo = cv2.VideoCapture(cam)
        saving_dir = str(datetime.now())
        if write_logs:
            os.mkdir(saving_dir)

        while vdo.grab():
            _, img0 = vdo.retrieve()
            batch_boxes, cropped_images = self.mtcnn.detect_box(img0)

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

                    if not wrong_person:
                        print(f"Hi, {min_key}")
                        if stop_when_rec:
                            break
                    else:
                        print("Wrong person detected!")

                    if write_logs:
                        self.log[str(datetime.now())] = min_key
                        if len(self.log.keys()) == write_logs_every:
                            self.save_log(path_for_saving=os.path.join(saving_dir, str(datetime.now()) + '.json'),
                                          clear_after_saving=True)
        vdo.release()


class Unlocker(Recognizer):
    """
    Class made for security systems.
    This is face unlocker.

    To start working with it:
        face_unlocker = aist_systems.face.Unlocker()
        face_unlocker.add_face()
        face_unlocker.unlock()

    If you need you can set password to your unlocker via instructions shown bellow:
        1) Launch:
            aist_systems.utils.get_hash(<your password>)
        2) Copy output's string
        3) face_unlocker.set_password(<output's string>)

        Then, if your face unlocker says that can't recognize face, you can enter your password.

    If you added some faces, you can save the config file to use them later:
        face_unlocker.save_core(<path where the config will be saved>)

    If you used Unlocker or Recognizer before and have a config file you can load it:
        face_unlocker.load_core(<path to core>)

    If you saved your config file on a web page, you can load it:
        face_unlocker.load_core_from_url(<url to the core>)
    """
    def __init__(self):
        """
        Init function of an Unlocker.
        """
        super().__init__()

        self.has_password = False
        self._password = None
        self._hash_method = None

    def launch(self,
               cam: int = 0,
               threshold: float = 0.7,
               num_of_attempts: int = 10,
               **kwargs) -> bool:
        """
        Face recognition part of Unlocker. If you need whole unlock-system use Unlocker.unlock()
        :param cam: if you have several cameras, you can choose which one you will use.
        :param threshold: confidence threshold: less = more strict
        :param num_of_attempts: If camera detects face but can't recognize it, the counter of unknown faces increase.
        You can choose the highest value of this counter
        :return: returns True or False.
        True - if recognized and the access is open.
        False - faces aren't recognized but detected many times.
        """
        vdo = cv2.VideoCapture(cam)
        wrong_person_detects = 0
        assert self.has_faces, "You didn't add any faces"

        while vdo.grab():
            _, img0 = vdo.retrieve()
            batch_boxes, cropped_images = self.mtcnn.detect_box(img0)

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
                        wrong_person_detects += 1
                        wrong_person = True
                        if wrong_person_detects == num_of_attempts:
                            print("Too much attempts!")
                            return False

                    if not wrong_person:
                        print(f"Hi, {min_key}")
                        return True
                    else:
                        print("Wrong person detected!")

    def set_password(self,
                     hash_object: str,
                     hash_method: str = 'sha256'):
        """
        If face recognizer didn't recognize the face, you can enter the password.

        If you need you can set password to your unlocker via instructions shown bellow:
            1) Launch:
                aist_systems.utils.get_hash(<your password>)
            2) Copy output's string
            3) face_unlocker.set_password(<output's string>)

            Then, if your face unlocker says that can't recognize face, you can enter your password.
        :param hash_object: Hash string of your password
        :param hash_method: If you change hash method from default (sha256),
        you have to specify which method you use.
        :return:
        """
        self.has_password = True
        self._password = hash_object
        self._hash_method = hash_method

    def _password_checker(self):

        input_object = input("Enter your password: \n")
        hashed_object = get_hash(hash_object=input_object,
                                 hash_method=self._hash_method)
        if hashed_object == self._password:
            return True
        return False

    def unlock(self,
               cam: int = 0,
               threshold: float = 0.7,
               num_of_attempts: int = 10,
               password_attempts: int = 3) -> bool:
        """
        Main function of Unlocker. Works with your cameras online.
        :param cam: if you have several cameras, you can choose which one you will use.
        :param threshold: confidence threshold: less = more strict
        :param num_of_attempts: If camera detects face but can't recognize it, the counter of unknown faces increase.
        You can choose the highest value of this counter
        :param password_attempts: how many times you can enter wrong passwords
        :return: True or False.
        True - if face recognized or password was correct.
        False - if unlocker didn't give access to a system.
        """
        launch_output = self.launch(cam=cam,
                                    threshold=threshold,
                                    num_of_attempts=num_of_attempts)
        if launch_output:
            return True
        if self.has_password:
            password_checking_output = self._password_checker()
            current_attempt = 1

            if not password_checking_output:
                while not password_checking_output and current_attempt < password_attempts:
                    left_attempts = password_attempts - current_attempt
                    if left_attempts > 1:
                        print(f"You have {left_attempts} attempts")
                    else:
                        print("You have last one attempt")
                    current_attempt += 1
                    password_checking_output = self._password_checker()
                    if password_checking_output:
                        return True
            else:
                return True
        return False
