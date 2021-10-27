import pika
import time
from redis import Redis
import base64
import numpy as np
import cv2
import sys
import signal
import json 
import random
import os
import uuid
from dotenv import load_dotenv
from pathlib import Path  
import tensorflow as tf
import logging
from mtcnn import MTCNN
import graypy


def json_load(body):
    """
        This function to decode json data.
        :param body : (byte), encoded json data
        :returns : Json Data
    """
    json_data = body.decode()
    return json.loads(json_data)

def redis_get(redis,json_data):
    """
        This function to check if frame exists in redis and get it as tobyte type.
        :param redis : redis connection
        :param json_data : (json), frame json data
        :returns frame : (tobyte) | None
    """
    if redis.exists(json_data['frame_uuid'] + "_data"):
        return  redis.get(json_data['frame_uuid'] + "_data")
    return None

def get_frame(frame_as_array):
    """
        This function to convert frame type from tobyte to uint8 and reshape it.
        :param frame_as_array : (array), frame (1-dimensional) array
        :returns frame : (uint8)
    """
    jpg_as_np = np.frombuffer(frame_as_array, dtype=np.uint8)
    frame = jpg_as_np.reshape(480,848,3)
    return frame


def do_detection(json_data,original_frame):
    """
        This function to detect faces and prepare dict of croped faces.
        :param json_data : (json), frame json data.
        :param original_frame : (uint8), frame.
        :returns dict of faces : (json).
    """
    confidence_t = 0.99
    faces_coordinate_dict = {}
    faces_counter = 0

    json_data = {
        "frame_uuid":  json_data['frame_uuid'],
        "camera_id" : json_data['camera_id'],
        "device_id" : json_data['device_id'],
        "captured_at": json_data['captured_at'],
        "coordinates": [ ]
        }

    frame_rgb = original_frame
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
    detector = MTCNN(min_face_size=40)
    faces = detector.detect_faces(frame_bgr)

    for face in faces:
        if face['confidence'] < confidence_t:
            continue
        
        x, y, width, height = face['box']
        croped_face = frame_rgb[y:y+height+1,x:x+width+1].copy()
        face_uuid = str(uuid.uuid4())
        final_coordinate = (x,y,width,height)
        faces_coordinate_dict[faces_counter] = {'detected_img_coordinate':final_coordinate,
                                                 "face_arr":croped_face, 
                                                 "success_rate":face["confidence"], 
                                                 "face_uuid":face_uuid
                                                }
        faces_counter += 1


    return faces_coordinate_dict
