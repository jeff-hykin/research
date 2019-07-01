import sys
import os
from os.path import isabs, isfile, isdir, join, dirname, basename, exists, splitext
from os import remove, getcwd, makedirs, listdir, rename, rmdir
from common_tools import download
from shutil import move
import dlib
import glob

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor( join(dirname(__file__), "./shape_predictor_68_face_landmarks.dat") )

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

