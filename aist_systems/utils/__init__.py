import hashlib
import pickle
import cv2
import numpy as np
from PIL import Image


def get_hash(hash_object: str,
             hash_method: str = 'sha256') -> str:
    """
    A function to get a hash code of your object.

    For examole:
    If you want to make a password verification, you can't write your password just in code, because it's unsafe
    Instead of that you can do this:
        1) Launch a file with coda shown bellow:
            print(aist_systems.get_hash(<your_password>))
        2) Copy the output string
        3) To verify is your input password correct:
            if aist_systems(<Input_password>) == <copied string>:
                print('Correct!')


    :param hash_object: an object to get hash code
    :param hash_method: method of hash coding. So far, available hash methods:
        'sha256', 'sha224', 'sha384', 'sha512', 'md5'
    :return: Hash code of the object
    """
    available_hashes = ['sha256', 'sha224', 'sha384', 'sha512', 'md5']
    assert hash_method in available_hashes, f"Hash method '{hash_method}' is not available"

    mapping = {'sha256': hashlib.sha256,
               'sha224': hashlib.sha224,
               'sha384': hashlib.sha384,
               'sha512': hashlib.sha512,
               'md5': hashlib.md5}

    hash_func = mapping[hash_method]
    return hash_func(hash_object.encode('utf-8')).hexdigest()


def save(saving_object, path):
    with open(path, 'wb') as f:
        pickle.dump(saving_object, f)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def decode(image_bytes: bytes) -> np.array:
    return cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)


def only_digits(string: str) -> str:
    answer_massive = [i for i in string if i.isdigit()]
    return "".join(answer_massive)


def pil_image_from_bytes(image_bytes: bytes):
    cv_image = decode(image_bytes)
    return Image.fromarray(cv_image)
