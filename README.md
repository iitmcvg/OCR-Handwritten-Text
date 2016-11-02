# Handwritten Optical Character Recognition
This is a **ongoing** project that aims to perform OCR on NIST Special Datbase 19 (handwritten characters) containing 62 classes (uppercase, lowercase and digits) using Tensorflow and Python

### Source Programs
* read_and_train.py: Reading and training from the `test` folder, which contains a small subset of the entire training data
* read3.py: Performs what read_and_train does, with a decaying learning rate
* Boundaries.m and TextSegmentation: Segmenting each letter of the read document into individual letters

### Dependencies
* OpenCV Python Bindings
* Numpy
* Tensorflow
