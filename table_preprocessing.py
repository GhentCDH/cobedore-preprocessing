#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:05:27 2023

@author: kthiruko
"""

import os
import time
import json
import numpy as np
from tqdm import trange, tqdm
from os import listdir
from shutil import copyfile,copy2

from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
import layoutparser as lp
import math
from PIL import Image

import cv2
import matplotlib.pyplot as plt

from scipy import signal


def check_dir(path):
    """
    Creates a new directory if the directory does not exist

    Parameters
    ----------
    path : Path to the directory

    Returns
    -------
    None

    """
    if not os.path.exists(path):
        os.mkdir(path)

def rotateImage(image, angle):
    """
    Rotate image to a certain angle

    Parameters
    ----------
    image : color image
    angle : angle value (int)

    Returns
    -------
    result : rotated color image

    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result


def white_balance(image):
    """
    White balancing on color image

    Parameters
    ----------
    image : color image

    Returns
    -------
    result : white balanced color image

    """
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def shadow_removal(image):
    """
    Removes shadows from a color image

    Parameters
    ----------
    image : color image

    Returns
    -------
    shadow_removed_image  : shadow removed color image
    normalized_shadow_removed_image  : nomralized and shadow removed color image

    """
    result_planes = []
    result_norm_planes = []
    for plane in image:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
        
    shadow_removed_image = cv2.merge(result_planes)
    normalized_shadow_removed_image = cv2.merge(result_norm_planes)
    
    return shadow_removed_image, normalized_shadow_removed_image

def gamma_correction(image, gamma):
    """
    Function to perfrom gamma correction on a image

    Parameters
    ----------
    image : grayscale image
    gamma : gamma value (Float)

    Returns
    -------
    gamma_corrected_image

    """
    
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    
    return cv2.LUT(image,gamma_table)
    
    
def find_splits(arr, kind='row', importance=5000):
    """
    Function to find the rows and columns

    Parameters
    ----------
        arr (list): 1D array with sums of rows/columns
        kind (str): 'col' will separate the middle line in two (default: row)

    Returns:
        list of split coordinates
        filtered signal
    """
    
    b, a = signal.butter(3, 0.05)
    filtered = signal.filtfilt(b, a, arr)

    peaks, _ = signal.find_peaks(arr,prominence=importance,distance=70)
    prominences = signal.peak_prominences(arr, peaks)[0]
    

    return peaks, filtered, prominences

def guess_splits(prev_splits, max_length):
    """
    Guess splits based on previously found splits

    Parameters
    ----------
        arr (list): 1D array with sums of rows/columns
        thres (int): threshold for split
        prev_splits (list): list of found splits in de standard way

    Returns:
        list of guessed split coordinates
    """
    
    # Distances between splits
    dist = []
    for i in range(len(prev_splits)-1):
        dist.append(prev_splits[i+1] - prev_splits[i])
        
    # Find outliers with IQR (inter quartile range)
    Q1 = np.percentile(dist, 25, interpolation = 'midpoint')
    Q3 = np.percentile(dist, 75, interpolation = 'midpoint')
    IQR = Q3 - Q1
    
    upper = dist >= (Q3+1.5*IQR)
    lower = dist <= (Q1-1.5*IQR)
    outliers = np.logical_or(upper, lower)
    
    # TODO Add beginning and ending prediction?
    total_splits = np.copy(prev_splits)
    new_splits = []
    if np.any(outliers):
        # Get average distance
        corr_dist = []
        for i in range(len(outliers)):
            if not outliers[i]:
                corr_dist.append(dist[i])
        
        avg_dist = int(sum(corr_dist)/len(corr_dist))
    
        # Base the search area on the outliers
        for i in range(len(outliers)):
            if outliers[i]:
                first_coord = prev_splits[i]
                last_coord = prev_splits[i+1]
                
                curr_coord = first_coord
                # Fill the area with 
                while(curr_coord +avg_dist*1.5 < last_coord):
                    total_splits = np.insert(total_splits, i+1, curr_coord)
                    new_splits.append(curr_coord+avg_dist)
                    
                    curr_coord += avg_dist
            
    return prev_splits, new_splits, total_splits


def visualize(image, rows, columns, row_splits=[], col_splits=[],
              guess_row_splits=[], guess_col_splits=[], show=False ,save=False, path='', fileName='image.jpg'):
    """
    Visualize the image with sums of rows and columns

    Parameters
    ----------
        image                  : color image
        rows (list)            : pixel sum for each row
        columns (list)         : pixel sum for each column
        
        Optional:
        row_splits (list)      : coordinates of splits in rows
        col_splits (list)      : coordinates of splits in columns
        guess_row_splits (list): coordinates of the guessed splits in rows
        guess_col_splits (list): coordinates of the guessed splits in columns
        show (bool)            : show to results on screen
        save (bool)            : save the figure to disk
        path (str)             : path to save the figure if save=True

    Returns:
        None
    """
    
    if show or save:
        fig, ax = plt.subplots(2, 2, figsize=(10,7))
        
        # Plot image with splits
        ax[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for i in row_splits:
            ax[0,0].axhline(y=i, color='red', linestyle='-')
        for i in col_splits:
            ax[0,0].axvline(x=i, color='red', linestyle='-')
            
        for i in guess_row_splits:
            ax[0,0].axhline(y=i, color='green', linestyle='-')
        for i in guess_col_splits:
            ax[0,0].axvline(x=i, color='green', linestyle='-')
        ax[0,0].tick_params(axis='both')
        
        # Plot rows with splits
        # Funky business to rotate the plot
        tmp = []
        tmp = np.linspace(0,len(rows),len(rows))

        ax[0,1].plot(rows, tmp)
        for i in row_splits:
            ax[0,1].axhline(y=i, color='red', linewidth=0.5, linestyle='-')
        for i in guess_row_splits:
            ax[0,1].axhline(y=i, color='green', linewidth=0.5, linestyle='-')
        ax[0,1].set_title('Rows')
        ax[0,1].set_ylabel('Height')
        ax[0,1].tick_params(axis='y')
        ax[0,1].set_ylim(0, image.shape[0])
        ax[0,1].invert_yaxis()
        
        # Plot columns with splits
        ax[1,0].plot(columns)
        for i in col_splits:
            ax[1,0].axvline(x=i, color='red', linewidth=0.5, linestyle='-')
        for i in guess_col_splits:
            ax[1,0].axvline(x=i, color='green', linewidth=0.5, linestyle='-')
        ax[1,0].set_title('Columns')
        ax[1,0].set_xlabel('Width')
        ax[1,0].tick_params(axis='x')
        ax[1,0].set_xlim(0, image.shape[1])

        if save:
            # Draw lines on image with opencv to save
            for i in col_splits:
                image = cv2.line(image, (i,0), (i,image.shape[0]), (0, 0, 255), 9)
            
            for i in guess_col_splits:
                image = cv2.line(image, (i,0), (i,image.shape[0]), (0, 255, 0), 9)

            for i in row_splits:
                image = cv2.line(image, (0,i), (image.shape[1],i), (0, 0, 255), 9)
                
            for i in guess_row_splits:
                image = cv2.line(image, (0,i), (image.shape[1],i), (0, 255, 0), 9)

            cv2.imwrite(path+fileName, image)

            # new_path = path.split('.')[-2] + '_analysis.jpg'
            # plt.savefig(new_path)

        if show:
            plt.show()

        plt.close('all')
        
        
####### Start here 

def process_Table(imagePath, rotate=True, angle=180, save=False, debug=False):
    """
    This function accepts an image path as input and returns the cropped original image, cropped preprocessed image, cropped thresholded image, columns, rows,

    Parameters
    ----------
    imagePath : Path to the the image (str)
    rotate : Boolean, optional -- Used to rotate an image
        The default is True.
    angle : angle value for the image to be rotated , optional
        The default is 180.
    save : Boolean, optional -- Used to save results
        The default is False.
    debug : Boolean, optional -- Used to visualize the results of each step
        The default is False.

    Returns
    -------
    list of images :
        table_crop: color image of the crop of the table
        table_preprocess: grayscale image of preprocessed crop
        table_threshold: thresholded image of table_preprocess
        rows, coloumns : List containing row and column splits

    """
    parentPath = os.path.split(os.path.split(imagePath)[0])[0]
    
    fileName = os.path.split(imagePath)[1]
    
    image = cv2.imread(imagePath)
    
    if debug:
        cv2.imshow("initial", cv2.resize(image, (0, 0), fx=0.4, fy=0.4))
        cv2.waitKey(0)
    
    ## Rotate the image if required
    if rotate is True:
        image = rotateImage(image, angle)
    
    rgb_planes = cv2.split(image)
    
    ##### Shadow removal ---- works well use when required
    
    # # result, result_norm = shadow_removal(rgb_planes)
    
    # # if debug:
    # #     cv2.imshow('shadow removal',  cv2.resize(result, (0, 0), fx=0.4, fy=0.4))
    # #     cv2.waitKey(0)
    # #     cv2.imshow('shadow removal with normalization', cv2.resize(result_norm, (0, 0), fx=0.4, fy=0.4))
    # #     cv2.waitKey(0)
    
    
    #### HSV colour filtering --- page extraction
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    ub = np.array([240, 240, 250])
    lb = np.array([3, 3, 80])
    
    
    mask = cv2.inRange(hsv, lb, ub)
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    areas = [cv2.contourArea(c) for c in contours]
    
    max_index = np.argmax(areas)
    c = contours[max_index]
    
    x, y, w, h = cv2.boundingRect(c)
    
    full_page = hsv[y:y + h, x:x + w]
    
    h,s,v = cv2.split(full_page)
    
    if debug:
        cv2.imshow("Page level crop", cv2.resize(v, (0, 0), fx=0.4, fy=0.4))
        cv2.waitKey(0)
        
    #### Gamma Correction
    image_gamma_correct=gamma_correction(v, 1.5)
    gray_inverted =cv2.bitwise_not(v)
    
    if debug:
        cv2.imshow("gamma corrected", cv2.resize(image_gamma_correct, (0, 0), fx=0.4, fy=0.4))
        cv2.waitKey(0)
    
    
    ####### equalization and Thresholding with blur 
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(1, 1))
    equalized = clahe.apply(image_gamma_correct)
    
    thresh = cv2.adaptiveThreshold(equalized, 255,
    	cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    
    
    thresh_erode = cv2.dilate(thresh, None,iterations=1)
    thresh_erode = cv2.erode(thresh_erode, None,iterations=1)
    blur = cv2.GaussianBlur(thresh_erode, (5,5), sigmaX=33, sigmaY=33)
    
    if debug:
        cv2.imshow("equalized", cv2.resize(equalized, (0, 0), fx=0.4, fy=0.4))
        cv2.waitKey(0)
        cv2.imshow("Adaptive thresholded", cv2.resize(blur, (0, 0), fx=0.4, fy=0.4))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
     
    ###### layout parser plugin for table detection 
    ### Download and install layout parser: https://github.com/Layout-Parser/layout-parser
    
    model = lp.Detectron2LayoutModel(
                config_path ='lp://TableBank/faster_rcnn_R_101_FPN_3x/config', # In model catalog
                label_map   ={0: "Table"}, # In model`label_map`
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8] # Optional
            )
    
    layout = model.detect(Image.fromarray(thresh))
    table_block = lp.Layout([b for b in layout if b.type=='Table'])
    
    ###### Detecting rows and columns in the table
    
    if len(table_block) > 0:
        for block in table_block:
            table_preprocess = (block
                           .pad(left=5, right=5, top=5, bottom=5)
                           .crop_image(gray_inverted))
            table_threshold = (block
                           .pad(left=5, right=5, top=5, bottom=5)
                           .crop_image(blur))
            table_crop = (block
                           .pad(left=5, right=5, top=5, bottom=5)
                           .crop_image(cv2.cvtColor(full_page, cv2.COLOR_HSV2BGR)))
        
            columns = find_splits(np.sum(table_preprocess,axis=0), kind='col', importance=31000)
            rows = find_splits(np.sum(table_preprocess,axis=1), kind='row')
            
            if debug:
                visualize(table_crop, np.sum(table_preprocess,axis=1), np.sum(table_preprocess,axis=0), col_splits=columns[0], row_splits=rows[0], show=debug, path=parentPath+"/table_vis/",fileName=fileName)

            
    else:
        table_preprocess = gray_inverted
        table_threshold = thresh
        table_crop = cv2.cvtColor(full_page, cv2.COLOR_HSV2BGR)
        columns = find_splits(np.sum(table_preprocess,axis=0), kind='col', importance=31000)
        rows = find_splits(np.sum(table_preprocess,axis=1), kind='row')        
        
    if save:
        check_dir(parentPath+"/table_preprocess")
        cv2.imwrite(parentPath+"/table_preprocess/"+fileName, table_preprocess)
        check_dir(parentPath+"/table_threshold")
        cv2.imwrite(parentPath+"/table_threshold/"+fileName, table_threshold)
        check_dir(parentPath+"/table_crop")
        cv2.imwrite(parentPath+"/table_crop/"+fileName, table_crop)
        check_dir(parentPath+"/table_vis")        
        
        visualize(table_crop, np.sum(table_preprocess,axis=1), np.sum(table_preprocess,axis=0), col_splits=columns[0], row_splits=rows[0], save=save, path=parentPath+"/table_vis/",fileName=fileName)
        
    
    return table_crop, table_preprocess, table_threshold, columns, rows


###### Run over all images in the folder and save images

# input_dir = '/home/kthiruko/cdh/data/Gimbi_Plateau_and_Tsibinda/7169/original/'
# problems = '/home/kthiruko/cdh/data/Gimbi_Plateau_and_Tsibinda/7169/problems/'

# listOfFiles = listdir(problems)

# for img in tqdm(sorted(listOfFiles)):
#     input_path = input_dir+img
#     try:
#         if input_path.endswith(".jpg"):
#             table = process_Table(input_path, save=True)
#     except:
#         if not os.path.isdir(problems):
#             os.mkdir(problems)
    
#         copy2(input_dir+img, problems)
#         print(input_dir+"------check")