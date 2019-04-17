# Drowsiness-detection

First, we’ll setup a camera that monitors a stream for faces. If a face is found, we apply facial landmark detection and extract the eye regions. Then we can compute the eye aspect ratio to determine if the eyes are closed. If the eye aspect ratio indicates that the eyes have been closed for a sufficiently long enough amount of time, we’ll sound an alarm to wake up the driver.

## Dependencies ## 

* OpenCV
* Playsound
* Dlib
* Scipy
