# WATCHING - module for realtime surveillance
## Watcher2D
Class for realtime watching from you camera.
Where you can choose subjects you are looking at.
Save logs etc.

### To start using it:
```python
watcher = aist_systems.watching.Watcher2D()
watcher.start()
```
 To specify what classes do you need:
 ```python
watcher.choose_classes([0, 1, 2]) # for example: people, bicycles, cars
```