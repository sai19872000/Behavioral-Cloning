import numpy as np
import cv2
import math

def jitter(img, angle):
    ''' 
    method for adding jitterness to dataset images, including random brightness adjust, and a random vertical shift of the horizon position
    '''
    new_img = img.astype(float)
    # random brightness - the mask bit keeps values from going beyond (0,255)
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (new_img[:,:,0] + value) > 255 
    if value <= 0:
        mask = (new_img[:,:,0] + value) < 0
    new_img[:,:,0] += np.where(mask, 0, value)
    # random shadow - full height, random left/right side, random darkening
    h,w = new_img.shape[0:2]
    mid = np.random.randint(0,w)
    factor = np.random.uniform(0.6,0.8)
    if np.random.rand() > .5:
        new_img[:,0:mid,0] *= factor
    else:
        new_img[:,mid:w,0] *= factor
    # randomly shift horizon
    h,w,_ = new_img.shape
    horizon = 2*h/5
    v_shift = np.random.randint(-h/8,h/8)
    pts1 = np.float32([[0,horizon],[w,horizon],[0,h],[w,h]])
    pts2 = np.float32([[0,horizon+v_shift],[w,horizon+v_shift],[0,h],[w,h]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    new_img = cv2.warpPerspective(new_img,M,(w,h), borderMode=cv2.BORDER_REPLICATE)
    return (new_img.astype(np.uint8), angle)


def crop_scaleImage(image,new_col,new_row):
    
    img_shape = image.shape
    
    image = image[math.floor(img_shape[0]/2.8):img_shape[0]-25, 0:img_shape[1]]
    image = cv2.resize(image,(new_col,new_row), interpolation=cv2.INTER_AREA)    
    return image
