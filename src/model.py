import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = "../models/3.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)    # loads model file into memory
interpreter.allocate_tensors()  # preps memory for inputs/outputs

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]['index'] # model input slot
output_index = output_details[0]['index'] # model output slot


# resizes webcam frame and normalizes it

def preprocess_frame(frame):

    # resize
    frame = cv2.resize(frame, (192, 192))

    # uint8 to float32
    frame = frame.astype(np.float32)

    

    # add batch dimension, model expects (1, 192, 192, 3)
    # currently its only (192, 192, 3)
    frame = np.expand_dims(frame, axis = 0)

    return frame
