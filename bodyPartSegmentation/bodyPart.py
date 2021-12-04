import os
import argparse
import sys

parser = argparse.ArgumentParser(description='loading eval params')
parser.add_argument('--gpus', metavar='N', type=int, default=1)
parser.add_argument('--model', type=str, default='./weights/model_simulated_RGB_mgpu_scaling_append.0071.h5', help='path to the weights file')
parser.add_argument('--input_folder', type=str, default='../clothesInput', help='path to the folder with test images')
parser.add_argument('--output_folder', type=str, default='../clothesOutput', help='path to the output folder')
parser.add_argument('--max', type=bool, default=True)
parser.add_argument('--average', type=bool, default=False)
parser.add_argument('--scale', action='append', help='<Required> Set flag', required=True)

args = parser.parse_args()

import cv2
import math
import time
import numpy as np
import util as util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from keras.models import load_model
import code
import copy
import scipy.ndimage as sn
from PIL import Image
from tqdm import tqdm
from model_simulated_RGB101 import get_testing_model_resnet101
from human_seg.human_seg_gt import human_seg_combine_argmax


right_part_idx = [2, 3, 4,  8,  9, 10, 14, 16]
left_part_idx =  [5, 6, 7, 11, 12, 13, 15, 17]
human_part = [0,1,2,4,3,6,5,8,7,10,9,12,11,14,13]
human_ori_part = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
seg_num = 15 # current model supports 15 parts only

def recover_flipping_output(oriImg, heatmap_ori_size, paf_ori_size, part_ori_size):

    heatmap_ori_size = heatmap_ori_size[:, ::-1, :]
    heatmap_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    heatmap_flip_size[:,:,left_part_idx] = heatmap_ori_size[:,:,right_part_idx]
    heatmap_flip_size[:,:,right_part_idx] = heatmap_ori_size[:,:,left_part_idx]
    heatmap_flip_size[:,:,0:2] = heatmap_ori_size[:,:,0:2]

    paf_ori_size = paf_ori_size[:, ::-1, :]
    paf_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
    paf_flip_size[:,:,ori_paf_idx] = paf_ori_size[:,:,flip_paf_idx]
    paf_flip_size[:,:,x_paf_idx] = paf_flip_size[:,:,x_paf_idx]*-1

    part_ori_size = part_ori_size[:, ::-1, :]
    part_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 15))
    part_flip_size[:,:,human_ori_part] = part_ori_size[:,:,human_part]
    return heatmap_flip_size, paf_flip_size, part_flip_size

def recover_flipping_output2(oriImg, part_ori_size):

    part_ori_size = part_ori_size[:, ::-1, :]
    part_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 15))
    part_flip_size[:,:,human_ori_part] = part_ori_size[:,:,human_part]
    return part_flip_size

def part_thresholding(seg_argmax):
    background = 0.6
    head = 0.5
    torso = 0.8

    rightfoot = 0.55 
    leftfoot = 0.55
    leftthigh = 0.55
    rightthigh = 0.55
    leftshank = 0.55
    rightshank = 0.55
    rightupperarm = 0.55
    leftupperarm = 0.55
    rightforearm = 0.55
    leftforearm = 0.55
    lefthand = 0.55
    righthand = 0.55
    
    part_th = [background, head, torso, leftupperarm ,rightupperarm, leftforearm, rightforearm, lefthand, righthand, leftthigh, rightthigh, leftshank, rightshank, leftfoot, rightfoot]
    th_mask = np.zeros(seg_argmax.shape)
    for indx in range(15):
        part_prediction = (seg_argmax==indx)
        part_prediction = part_prediction*part_th[indx]
        th_mask += part_prediction

    return th_mask


def process (input_image, params, model_params):
    input_scale = 1.0

    oriImg = cv2.imread(input_image)
    flipImg = cv2.flip(oriImg, 1)
    oriImg = (oriImg / 256.0) - 0.5
    flipImg = (flipImg / 256.0) - 0.5
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]

    seg_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 15))

    segmap_scale1 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale2 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale3 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale4 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))

    segmap_scale5 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale6 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale7 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale8 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))

    for m in range(len(multiplier)):
        scale = multiplier[m]*input_scale
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        pad = [ 0,
                0, 
                (imageToTest.shape[0] - model_params['stride']) % model_params['stride'],
                (imageToTest.shape[1] - model_params['stride']) % model_params['stride']
              ]
        
        imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))

        input_img = imageToTest_padded[np.newaxis, ...]
        
        print( "\tActual size fed into NN: ", input_img.shape)

        output_blobs = model.predict(input_img)
        seg = np.squeeze(output_blobs[2])
        seg = cv2.resize(seg, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        if m==0:
            segmap_scale1 = seg
        elif m==1:
            segmap_scale2 = seg         
        elif m==2:
            segmap_scale3 = seg
        elif m==3:
            segmap_scale4 = seg


    # flipping
    for m in range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv2.resize(flipImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        pad = [ 0,
                0, 
                (imageToTest.shape[0] - model_params['stride']) % model_params['stride'],
                (imageToTest.shape[1] - model_params['stride']) % model_params['stride']
              ]
        
        imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))
        input_img = imageToTest_padded[np.newaxis, ...]
        print( "\tActual size fed into NN: ", input_img.shape)
        output_blobs = model.predict(input_img)

        # extract outputs, resize, and remove padding
        seg = np.squeeze(output_blobs[2])
        seg = cv2.resize(seg, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        seg_recover = recover_flipping_output2(oriImg, seg)

        if m==0:
            segmap_scale5 = seg_recover
        elif m==1:
            segmap_scale6 = seg_recover         
        elif m==2:
            segmap_scale7 = seg_recover
        elif m==3:
            segmap_scale8 = seg_recover

    segmap_a = np.maximum(segmap_scale1,segmap_scale2)
    segmap_b = np.maximum(segmap_scale4,segmap_scale3)
    segmap_c = np.maximum(segmap_scale5,segmap_scale6)
    segmap_d = np.maximum(segmap_scale7,segmap_scale8)
    seg_ori = np.maximum(segmap_a, segmap_b)
    seg_flip = np.maximum(segmap_c, segmap_d)
    seg_avg = np.maximum(seg_ori, seg_flip)

    
    return seg_avg


if __name__ == '__main__':

    args = parser.parse_args()
    keras_weights_file = args.model

    print('start processing...')
    # load model
    model = get_testing_model_resnet101() 
    model.load_weights(keras_weights_file)
    params, model_params = config_reader()

    scale_list = []
    for item in args.scale:
        scale_list.append(float(item))

    params['scale_search'] = scale_list

    # generate image with body parts
    for filename in os.listdir(args.input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            print(args.input_folder+'/'+filename)
            seg = process(args.input_folder+'/'+filename, params, model_params)

            seg_argmax = np.argmax(seg, axis=-1)
            seg_max = np.max(seg, axis=-1)
            th_mask = part_thresholding(seg_argmax)
            seg_max_thres = (seg_max > 0.1).astype(np.uint8)
            seg_argmax *= seg_max_thres
            seg_canvas = human_seg_combine_argmax(seg_argmax)
            
            print(seg_canvas.shape)
            seg_zero = seg_canvas[:,:,2] # i번째 seg
            cur_canvas = cv2.imread(args.input_folder+'/'+filename) # 현재 그림
            
            # rearrange body part
            re_canvas = np.zeros((seg_zero.shape[0], seg_zero.shape[1], 9))
            re_canvas[:,:,0] = seg_canvas[:,:,2]
            re_canvas[:,:,1] = seg_canvas[:,:,4]
            re_canvas[:,:,2] = seg_canvas[:,:,6]
            re_canvas[:,:,3] = seg_canvas[:,:,3]
            re_canvas[:,:,4] = seg_canvas[:,:,5]
            re_canvas[:,:,5] = seg_canvas[:,:,10]
            re_canvas[:,:,6] = seg_canvas[:,:,12]
            re_canvas[:,:,7] = seg_canvas[:,:,9]
            re_canvas[:,:,8] = seg_canvas[:,:,11]

            for k in range(0, 9):
                segment = np.zeros((seg_zero.shape[0], seg_zero.shape[1], 4))
                for i in range(0, seg_zero.shape[0]):
                    for j in range(0, seg_zero.shape[1]):
                        if re_canvas[i][j][k] == 1:
                            segment[i][j][:-1] = cur_canvas[i][j]
                            segment[i][j][3] = 255
                file = '%s/%s'%(args.output_folder,'seg_' + str(k) + '_' + filename)
                print(file)
                cv2.imwrite(file, segment)

                '''
                # 꼭짓점 검출
                segmentGray = cv2.cvtColor(segment.astype('float32'), cv2.COLOR_BGR2GRAY) # 꼭짓점 추출에는 흑백 그림이 필요합니다.
                segmentPoint = cv2.cvtColor(segment.astype('float32'), cv2.COLOR_BGR2RGB) # 나중에 꼭짓점을 여기에 표시합니다.
                segmentGray = np.float32(segmentGray) # 넘파이(numpy)를 사용해서 자료형을 부동소수점으로 바꾸어주어야 합니다.
                result = cv2.cornerHarris(segmentGray, 2, 3, 0.04) # 2, 3, 0.04는 바꿀 수 있는 인자들입니다.
                result = cv2.dilate(result, None, iterations=6) # 꼭짓점을 표시하기 위해 확장 (dilate) 연산을 합니다.
                segmentPoint[result>0.01*result.max()]=[255, 0, 0] # 꼭짓점이 빨간색 점으로 그림에 표시됩니다.
                '''

            # 꼭짓점 검출
            '''
            img = cv2.imread('output/seg_0_clothes.jpg')
            contours, hierarchy = get_corner.getContour(img, visualize=False)
            smoothedImg = cv2.imread("output_point/contour.jpg")
            validHull, validContours = get_corner.getConvexHull(smoothedImg, contours, hierarchy)
            corners = get_corner.getCorners('output_point/contour.jpg')
            '''
            
            '''
            for i in range(0, 9):
                img = cv2.imread('output/seg_{}_clothes.jpg'.format(i))
                contourList, hList = get_corner.getContour(img, 50, "results/temp_contour.jpg")
                _, _, pnts = get_corner.getConvexHull(img, contourList, hList)

                #cImg = cv2.imread("output/seg_{}_clothes.jpg")
                for dot in pnts:
                    cv2.circle(img, (int(dot[0]), int(dot[1])), 7, (255,255,255), 2)
                cv2.imwrite("output/points_{}_clothes.jpg".format(i), img)
            '''
            
            #cv2.imwrite(file, segment)