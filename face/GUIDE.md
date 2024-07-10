# FACE - module made for security systems for working with faces
## Recognizer
If you need just face recognition without any additional security tasks, you can use Recognizer car
```python
face_recognizer = aist_systems.face.Recognizer()
face_recognizer.add_face()
face_recognizer.launch()
```
If you used it and want to save your config with your saved face:
```python
face_recognizer.save_core(<path_for_saving>)
```

If you want to load the config:
```python
face_recognizer.load_core(<path_for_loading>)
```
Or if you have an url of core file:
```python
face_recognizer.load_core_from_url(<url_to_core_file>)
```
## Unlocker
Class made for security systems.
This is face unlocker.

### To start working with it:
```python
face_unlocker = aist_systems.face.Unlocker()
face_unlocker.add_face()
face_unlocker.unlock()
```
### Setting password
If you need you can set password to your unlocker via instructions shown bellow:

    1) Launch:
        aist_systems.utils.get_hash(<your password>)
    2) Copy output's string
    3) face_unlocker.set_password(<output's string>)

Then, if your face unlocker says that can't recognize face, you can enter your password.

### Saving and loading cores
If you added some faces, you can save the config file to use them later:
```python
face_unlocker.save_core(<path where the config will be saved>)
```
If you used Unlocker or Recognizer before and have a config file you can load it:
```python
face_unlocker.load_core(<path to core>)
```
If you saved your config file on a web page, you can load it:
```python
face_unlocker.load_core_from_url(<url to the core>)
```
    