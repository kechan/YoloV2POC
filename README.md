# YoloV2POC
Running full Yolo V2 on iPad at 1280x720. A rough proof of concept.

1) Pre-trained model weights from Coursera Deep Learning Specialization's Course 4 on Convolutional Network.
2) Custom layer for tf.space_to_depth and final activations written with Apple Accelerate (this is probably suboptimal, see below).

Since iOS 12, Apple has new framework support such that you don't have to implement your own filtering and non-max suppression post 
processing. For now, I think this is true for model created with "Create ML". Since the pre-trained model is from Keras, I have yet to 
figure out this out. The custom layers are written using Accelerate Framework. Presumably, written them in lower level code or Metal 
Performance Shader will speed things up. Currently, it takes roughly 250-300ms to complete an inference, which is too slow for the purpose
of live stream and tracking for moving objects.
