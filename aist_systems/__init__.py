"""
    AISt - Artificial Intelligence Security. This is the library for security systems made with AI.

    So far, we have these modules:
        1) face - module for face recognition or face unlocking systems.
        2) utils - module with some stuff for systems.
        3) watching - module for surveillance

    You can look at these libraries in folders with corresponding names
"""
import cv2

def test_camera(camera_index : int = 0):
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
