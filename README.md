"detect.tflite" and "labelmap.txt" are the trained tensorflow lite models that i trained on my computer

How i intend for my object detection to work-
there is 1 usb web camera source, and during the day it is a regular webcam and during the night it goes into nightvision mode (IR camera)
in order to make the object detection model detect black and white feed from the IR camera i added an augmentation step with black/white images
