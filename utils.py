# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 17:35:54 2022

@author: overs
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from shapely.geometry import Polygon
import cv2 as cv

#helper function for plot_with_box
def boxArea(box_pts):
    #box_pts is np array of shape (4,2)
    xmax = np.max(box_pts[:,0])
    xmin = np.min(box_pts[:,0])
    ymax = np.max(box_pts[:,1])
    ymin = np.min(box_pts[:,1])
    print(box_pts)
    print(f'xmax:{xmax}, xmin:{xmin}, ymax:{ymax}, ymin:{ymin}')
    return (xmax - xmin + 1) * (ymax - ymin + 1)

#helper function for plot_with_box, calculates IoU
def intrArea(box_pts, comp_pts):
    poly_target = Polygon(box_pts.reshape(-1, 2))
    poly_test = Polygon(comp_pts.reshape(-1, 2))
    poly_inter = poly_target & poly_test

    area_target = poly_target.area
    area_test = poly_test.area
    area_inter = poly_inter.area

    area_union = area_test + area_target - area_inter
    # Little hack to cope with float precision issues when dealing with polygons:
        #   If intersection area is close enough to target area or GT area, but slighlty >,
        #   then fix it, assuming it is due to rounding issues.
    area_min = min(area_target, area_test)
    if area_min < area_inter and area_min * 1.0000000001 > area_inter:
       area_inter = area_min
       print("Capping area_inter.")

    jaccard_index = area_inter / area_union
    return jaccard_index

#displays image with box over it
def plot_with_box(image_data, bounding_box, compare_box=None):
    fig,ax = plt.subplots(1)
    ax.imshow(image_data)
    # Creating a polygon patch for the changed one
    boxA = patches.Polygon(bounding_box,linewidth=3, edgecolor='y', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(boxA)
    #Creating another polygon patch for the real one
    if compare_box is not None:
        '''boxB = patches.Rectangle((compare_box.xmin, compare_box.ymin),
                                 compare_box.xmax - compare_box.xmin,
                                 compare_box.ymax - compare_box.ymin,
                                 linewidth=2, edgecolor='b', facecolor='none')'''
        boxB = patches.Polygon(compare_box,linewidth=2, edgecolor='b', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(boxB)
        #FOR FINDING INTERSECTION OVER UNION
        
        iou =intrArea(bounding_box,compare_box)
        #By intersection of union I mean intersection over union(IOU) #itself
        print('intersection of union =',iou)
        plt.show()

        

#resizes image so it's a square and is smaller for faster processing
def resize_image_eff(img, bounding_box=None, target_size=None):
    #bounding_box is (4,2) np array of coords
    #target_size is tuple of new size to resize to
    image = img#already image object
    
    height, width, channels= image.shape
    w_pad = 0
    h_pad = 0
    bonus_h_pad = 0
    bonus_w_pad = 0
#the following code helps determining where to pad or is it not necessary for the images we have.
# If the difference between the width and height was odd((height<width)case), we add one pixel on one side
# If the difference between the height and width was odd((height>width)case), then we add one pixel on one side.
#if both of these are not the case, then pads=0, no padding is needed, since the image is already a square itself.
    if width > height:
        pix_diff = (width - height)
        h_pad = pix_diff // 2
        bonus_h_pad = pix_diff % 2
    elif height > width:
        pix_diff = (height - width)
        w_pad = pix_diff // 2
        bonus_w_pad = pix_diff % 2
# When we pad the image to square, we need to adjust all the bounding box values by the amounts we added on the left or top.
#The "bonus" pads are always done on the bottom and right so we can ignore them in terms of the box.
    #top, bottom, left, right
    image = cv.copyMakeBorder(image, h_pad, h_pad+bonus_h_pad, w_pad, w_pad+bonus_w_pad, borderType=0)
    toRet = np.copy(bounding_box)
    unsize = [w_pad, h_pad]
    if bounding_box is not None:
        toRet[:,0] += w_pad
        toRet[:,1] += h_pad
        # We need to also apply the scalr to the bounding box which we used in resizing the image
    if target_size is not None:
        # So, width and height have changed due to the padding resize.
        height, width, channels = image.shape
        image = cv.resize(image, target_size)
        width_scale = target_size[0] / width
        height_scale = target_size[1] / height
        if bounding_box is not None:
            toRet[:,0] *= width_scale
            toRet[:,1] *= height_scale
            unsize.append(width_scale)
            unsize.append(height_scale)
    # The image data is a 3D array such that 3 channels ,RGB of target_size.(RGB values are 0-255)
    if bounding_box is None:
        return image, None
    #can use information in unsize to translate predicted box coordinates for original image
    return (image, toRet, unsize)


def unsize(pred, info):
    #translates predicted coordinates from resized image to original image
    pred = np.copy(pred)
    pred[:,0] /= info[2]#undoes width scale
    pred[:,1] /= info[3]#undoes height scaling
    pred[:,0] -= info[0]#undoes width padding
    pred[:,1] -= info[1]#undoes height padding
    return pred