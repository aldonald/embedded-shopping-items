import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
import logging
import requests
import json
import time

token = 'Token 87e7a6cc416dbc0630a598bf921acefa5469b75e'
# Set up camera constants
IM_WIDTH = 1280
IM_HEIGHT = 720
#IM_WIDTH = 640    Use smaller resolution for
#IM_HEIGHT = 480   slightly faster framerate

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

# Shopping items
SHOPPING_LIST = os.path.join(CWD_PATH, 'variables', 'shopping_items.txt')


## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)
        
def run_camera():    
    shopping_items_file = open(SHOPPING_LIST, "r")
    list_of_items = shopping_items_file.read().splitlines()
    shopping_items_file.close()
    new_list_of_items = []
    add_to_shopping_list = []
    
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    font = cv2.FONT_HERSHEY_SIMPLEX

    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)
    
    start_time = time.time()
    
    try:
        for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
            t1 = cv2.getTickCount()
                
            # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
            # i.e. a single-column array, where each item in the column has the pixel RGB value
            frame = np.copy(frame1.array)
            frame.setflags(write=1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_expanded = np.expand_dims(frame_rgb, axis=0)

            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: frame_expanded})
            if 55 in classes:
                logging.warning("Orange detected")
                if "orange" not in new_list_of_items:
                    new_list_of_items.append("orange")
                
            if 52 in classes:
                logging.warning("Banana detected")
                if "banana" not in new_list_of_items:
                    new_list_of_items.append("banana")

            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc = 1/time1

            rawCapture.truncate(0)
            time_lapsed = time.time() - start_time
            if time_lapsed > 60:
                break
        
        # Now to handle the data captured
        logging.warning(f"Old list: {list_of_items}")
        logging.warning(f"New list: {new_list_of_items}")
        
        for item in list_of_items:
            if item not in new_list_of_items:
                add_to_shopping_list.append(item)
                
        logging.warning(f"Shopping list: {add_to_shopping_list}")
        
        for item in add_to_shopping_list:
            if item == "banana":
                price = "0.90"
            if item == "orange":
                price = "1.36"
            data = {
                "data": {
                    "type": "ShoppingItem",
                    "attributes": {
                         "name": item.capitalize(),
                         "price": price
                    }
                }
            }
            json_data = json.dumps(data)
            logging.warning(data)
            response = requests.post(
                'https://tranquil-lowlands-73758.herokuapp.com/api/pi_add_item/',
                data=json_data,
                headers={"Authorization": token, "Content-Type": "application/vnd.api+json"}
            )
            logging.warning(f"Response received: {response.json()}")
        
        shopping_items_file = open(SHOPPING_LIST, "w")
        shopping_items_file.write('\n'.join(new_list_of_items))
        shopping_items_file.close()
            
        camera.close()

        cv2.destroyAllWindows()
        
        
    except KeyboardInterrupt:
        logging.warning("Interrupted")
        logging.warning(list_of_items)
        logging.warning(new_list_of_items)
        for item in list_of_items:
            if item not in new_list_of_items:
                add_to_shopping_list.append(item)
        list_of_items = new_list_of_items
        logging.warning(add_to_shopping_list)
        for item in add_to_shopping_list:
            if item == "banana":
                price = "0.90"
            if item == "orange":
                price = "1.36"
            data = {
                "data": {
                    "type": "ShoppingItem",
                    "attributes": {
                         "name": item.capitalize(),
                         "price": price
                    }
                }
            }
            json_data = json.dumps(data)
            logging.warning(data)
            response = requests.post(
                'https://tranquil-lowlands-73758.herokuapp.com/api/pi_add_item/',
                data=json_data,
                headers={"Authorization": token, "Content-Type": "application/vnd.api+json"}
            )
            logging.warning(response.json())
        
        shopping_items_file = open(SHOPPING_LIST, "w")
        shopping_items_file.write('\n'.join(list_of_items))
        shopping_items_file.close()
            
        camera.close()

        cv2.destroyAllWindows()

