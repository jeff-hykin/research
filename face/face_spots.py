import sys
import os
from os.path import isabs, isfile, isdir, join, dirname, basename, exists, splitext
from os import remove, getcwd, makedirs, listdir, rename, rmdir
from shutil import move
import dlib
import glob
import numpy as np

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor( join(dirname(__file__), "./shape_predictor_68_face_landmarks.dat") )

nuber_of_face_features = 68

def bounds_to_points(max_x, max_y, min_x, min_y):
    return (min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, min_y)

def bounding_box(array_of_points):
    """
    the input needs to be an array with the first column being x values, and the second column being y values
    """
    max_x = -float('Inf')
    max_y = -float('Inf')
    min_x = -float('Inf')
    min_y = -float('Inf')
    for each in array_of_points:
        if max_x < each[0]:
            max_x = each[0]
        if max_y < each[1]:
            max_y = each[1]
        if min_x > each[0]:
            min_x = each[0]
        if min_y > each[1]:
            min_y = each[1]
    return max_x, max_y, min_x, min_y

class Face():
    def __init__(self, shape):
        global nuber_of_face_features
        # create the empty array
        self.as_array = np.empty((nuber_of_face_features, 2), dtype=np.int32)
        # store the face as an array
        for each_part_index in range(shape.num_parts):
            point = shape.part(each_part_index)
            as_array[each_part_index][0] = point.x
            as_array[each_part_index][1] = point.y
        # calculate the bounding boxes
        self.chin_curve_bounds    = bounding_box(self.chin_curve())
        self.left_eyebrow_bounds  = bounding_box(self.left_eyebrow())
        self.right_eyebrow_bounds = bounding_box(self.right_eyebrow())
        self.nose_bounds          = bounding_box(self.nose())
        self.left_eye_bounds      = bounding_box(self.left_eye())
        self.right_eye_bounds     = bounding_box(self.right_eye())
        self.mouth_bounds         = bounding_box(self.mouth())
        # calculate the face bounding box
        max_x = max(chin_curve_bounds[0], left_eyebrow_bounds[0], right_eyebrow_bounds[0], nose_bounds[0], left_eye_bounds[0], right_eye_bounds[0], mouth_bounds[0])
        max_y = max(chin_curve_bounds[1], left_eyebrow_bounds[1], right_eyebrow_bounds[1], nose_bounds[1], left_eye_bounds[1], right_eye_bounds[1], mouth_bounds[1])
        min_x = max(chin_curve_bounds[2], left_eyebrow_bounds[2], right_eyebrow_bounds[2], nose_bounds[2], left_eye_bounds[2], right_eye_bounds[2], mouth_bounds[2])
        min_y = max(chin_curve_bounds[3], left_eyebrow_bounds[3], right_eyebrow_bounds[3], nose_bounds[3], left_eye_bounds[3], right_eye_bounds[3], mouth_bounds[3])
        self.bounds = ( max_x, max_y, min_x, min_y )

    def as_array(self):
        return self.as_array
    
    # 
    # Facial parts
    #
    def chin_curve(self):
        return self.as_array[0:16]
    def left_eyebrow(self):
        return self.as_array[17:21]
    def right_eyebrow(self):
        return self.as_array[22:26]
    def nose(self):
        return self.as_array[27:35]
    def left_eye(self):
        return self.as_array[36:41]
    def right_eye(self):
        return self.as_array[42:47]
    def mouth(self):
        return self.as_array[48:67]

    #
    # bounding boxes
    # 
    def bounding_box(self):
        return bounds_to_points(self.bounds)
    def bounding_box(self):
        return bounds_to_points(self.bounds)
    def chin_curve_bounding_box(self):
        return bounds_to_points(self.chin_curve_bounds)
    def left_eyebrow_bounding_box(self):
        return bounds_to_points(self.left_eyebrow_bounds)
    def right_eyebrow_bounding_box(self):
        return bounds_to_points(self.right_eyebrow_bounds)
    def nose_bounding_box(self):
        return bounds_to_points(self.nose_bounds)
    def left_eye_bounding_box(self):
        return bounds_to_points(self.left_eye_bounds)
    def right_eye_bounding_box(self):
        return bounds_to_points(self.right_eye_bounds)
    def mouth_bounding_box(self):
        return bounds_to_points(self.mouth_bounds)

def faces_for(jpg_image_path):
    global detector
    global predictor
    
    # load up the image
    img = dlib.load_rgb_image(jpg_image_path)
    
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    # initialize by the number of faces
    faces = [None]*len(dets)
    for index, d in enumerate(dets):
        faces[index] = Face(predictor(img, d))
    
    return faces

def vector_points_for(jpg_image_path):
    global detector
    global predictor
    
    # load up the image
    img = dlib.load_rgb_image(jpg_image_path)
    
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    
    # initialize by the number of faces
    faces = [None]*len(dets)
    for index, d in enumerate(dets):
        shape = predictor(img, d)
        # copy over all 68 facial features/vertexs/points
        faces[index] = [ shape.part(each_part_index) for each_part_index in range(shape.num_parts) ]
    
    return faces
