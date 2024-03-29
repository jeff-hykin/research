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
    min_x = float('Inf')
    min_y = float('Inf')
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
    def __init__(self, shape, img):
        global nuber_of_face_features
        self.img = img
        # create the empty array
        self.as_array = np.empty((nuber_of_face_features, 2), dtype=np.int32)
        # store the face as an array
        for each_part_index in range(shape.num_parts):
            point = shape.part(each_part_index)
            self.as_array[each_part_index][0] = point.x
            self.as_array[each_part_index][1] = point.y
        # calculate the bounding boxes
        self.chin_curve_bounds    = bounding_box(self.chin_curve())
        self.left_eyebrow_bounds  = bounding_box(self.left_eyebrow())
        self.right_eyebrow_bounds = bounding_box(self.right_eyebrow())
        self.nose_bounds          = bounding_box(self.nose())
        self.left_eye_bounds      = bounding_box(self.left_eye())
        self.right_eye_bounds     = bounding_box(self.right_eye())
        self.mouth_bounds         = bounding_box(self.mouth())
        # calculate the face bounding box
        max_x = max(self.chin_curve_bounds[0], self.left_eyebrow_bounds[0], self.right_eyebrow_bounds[0], self.nose_bounds[0], self.left_eye_bounds[0], self.right_eye_bounds[0], self.mouth_bounds[0])
        max_y = max(self.chin_curve_bounds[1], self.left_eyebrow_bounds[1], self.right_eyebrow_bounds[1], self.nose_bounds[1], self.left_eye_bounds[1], self.right_eye_bounds[1], self.mouth_bounds[1])
        min_x = min(self.chin_curve_bounds[2], self.left_eyebrow_bounds[2], self.right_eyebrow_bounds[2], self.nose_bounds[2], self.left_eye_bounds[2], self.right_eye_bounds[2], self.mouth_bounds[2])
        min_y = min(self.chin_curve_bounds[3], self.left_eyebrow_bounds[3], self.right_eyebrow_bounds[3], self.nose_bounds[3], self.left_eye_bounds[3], self.right_eye_bounds[3], self.mouth_bounds[3])
        self.bounds = ( max_x, max_y, min_x, min_y )
    
    def bounded_by(self, bounds, padding):
        height = self.img.shape[0]
        x_max = bounds[0] + int(padding * height)
        y_max = bounds[1] + int(padding * height)
        x_min = bounds[2] - int(padding * height)
        y_min = bounds[3] - int(padding * height)
        # dont let the indices go negative
        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        return self.img[ y_min:y_max, x_min:x_max]
    
    #
    # Facial parts
    #
    # see: https://miro.medium.com/max/828/1*96UT-D8uSXjlnyvs9DZTog.png
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
        return bounds_to_points(*self.bounds)
    def chin_curve_bounding_box(self):
        return bounds_to_points(*self.chin_curve_bounds)
    def left_eyebrow_bounding_box(self):
        return bounds_to_points(*self.left_eyebrow_bounds)
    def right_eyebrow_bounding_box(self):
        return bounds_to_points(*self.right_eyebrow_bounds)
    def nose_bounding_box(self):
        return bounds_to_points(*self.nose_bounds)
    def left_eye_bounding_box(self):
        return bounds_to_points(*self.left_eye_bounds)
    def right_eye_bounding_box(self):
        return bounds_to_points(*self.right_eye_bounds)
    def mouth_bounding_box(self):
        return bounds_to_points(*self.mouth_bounds)
    
    #
    # Save options
    #
    def save_to(self, image_path, padding):
        """padding is a percentage of the height"""
        dlib.save_image(self.bounded_by(self.bounds, padding), image_path)
    def save_chin_curve_to(self, image_path, padding):
        """padding is a percentage of the height"""
        dlib.save_image(self.bounded_by(self.chin_curve_bounds, padding), image_path)
    def save_left_eyebrow_to(self, image_path, padding):
        """padding is a percentage of the height"""
        dlib.save_image(self.bounded_by(self.left_eyebrow_bounds, padding), image_path)
    def save_right_eyebrow_to(self, image_path, padding):
        """padding is a percentage of the height"""
        dlib.save_image(self.bounded_by(self.right_eyebrow_bounds, padding), image_path)
    def save_nose_to(self, image_path, padding):
        """padding is a percentage of the height"""
        dlib.save_image(self.bounded_by(self.nose_bounds, padding), image_path)
    def save_left_eye_to(self, image_path, padding):
        """padding is a percentage of the height"""
        dlib.save_image(self.bounded_by(self.left_eye_bounds, padding), image_path)
    def save_right_eye_to(self, image_path, padding):
        """padding is a percentage of the height"""
        dlib.save_image(self.bounded_by(self.right_eye_bounds, padding), image_path)
    def save_mouth_to(self, image_path, padding):
        """padding is a percentage of the height"""
        dlib.save_image(self.bounded_by(self.mouth_bounds, padding), image_path)


def faces_for(img):
    global detector
    global predictor

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    # initialize by the number of faces
    faces = [None]*len(dets)
    for index, d in enumerate(dets):
        faces[index] = Face(predictor(img, d), img)

    return faces

def aligned_faces_for(img, size=320, padding=0.25):
    images = get_aligned_face_images(img, size, padding)
    faces = [None]*len(images)
    for each_index, each_img in enumerate(images):
        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(each_img, 1)
        # initialize by the number of faces
        faces = [None]*len(dets)
        for d in dets:
            faces[each_index] = Face(predictor(each_img, d), each_img)
    return faces


def get_aligned_face_images(img, size=320, padding=0.25):
    global detector
    global predictor

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)

    # if no faces return an empty list
    if len(dets) == 0:
        return []

    # Find the 5 face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(predictor(img, detection))

    # returns a list of images
    return dlib.get_face_chips(img, faces, size=size, padding=padding)


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

def test_example():
    # load up the image
    img = dlib.load_rgb_image("./face/faces/person.jpg")
    faces = aligned_faces_for(img, size=800, padding=0.25)

    faces[0].save_left_eye_to("./face/faces/left_eye.nosync.jpeg", padding=0.05)
    faces[0].save_to("./face/faces/face.nosync.jpeg", padding=0.05)
    
    return faces

faces = test_example()
print("done")