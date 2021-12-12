from enum import Enum
import numpy as np
import cv2
from detectCorner import detectCorners

class BodyParts(Enum):
    BODY = 0
    RIGHT_BRANCHIAL = 1
    RIGHT_FOREARM = 2
    LEFT_BRANCHIAL = 3
    LEFT_FOREARM = 4
    RIGHT_THIGH = 5
    RIGHT_CALF = 6
    LEFT_THIGH = 7
    LEFT_CALF = 8

class Transform(Enum):
    ThinPlateSpline = 0
    Perspective = 1

def warpClothesSegs(segImgs, humanImg, ordered_clothes_points, ordered_pose_points):
    ################################# hyper parameter #################################
    transformType = Transform.ThinPlateSpline   # warping type
    Nb = 1                                      # height of gaussian pyramid
    ###################################################################################
    
    divider = pow(2, Nb)

    total_parts_num = 9
    pair_num = [8, 4, 4, 4, 4, 4, 4, 4, 4] # number of pairs for each body part
    matchStartNum = 0

    tried_on = humanImg
    M2, N2 = tried_on.shape[:2]

    M2_, N2_ = (M2 // divider) * divider, (N2 // divider) * divider
    tried_on = cv2.resize(tried_on, (N2_, M2_))

    for i in range(total_parts_num):
        # get each clothes segment
        clothes_seg = segImgs[i].astype(np.uint8)
        M1, N1 = clothes_seg.shape[:2]

        # resize it to size of pose image
        clothes = cv2.resize(clothes_seg, (N2_, M2_)) # direction check required
        clothes_points = [(int(point[0]*N2_/N1), int(point[1]*M2_/M1)) for point in ordered_clothes_points]

        if transformType == Transform.ThinPlateSpline:
            # prepare target points, source points, and match indices for transformation
            target_shape = np.array(ordered_pose_points,np.int32).reshape(1,-1,2)
            source_shape = np.array(clothes_points,np.int32).reshape(1,-1,2)
            matches =[]
            for j in range(matchStartNum, matchStartNum + pair_num[i]):
                matches.append(cv2.DMatch(j,j,0))

            # warp clothes with the obtained transformation
            tps = cv2.createThinPlateSplineShapeTransformer()
            tps.estimateTransformation(target_shape, source_shape, matches)
            warped_clothes = tps.warpImage(clothes)
        else: # Perspective warping
            if i == 0:
                M = cv2.getPerspectiveTransform(np.float32([clothes_points[1], clothes_points[3], clothes_points[4], clothes_points[6]]), np.float32([ordered_pose_points[1], ordered_pose_points[3], ordered_pose_points[4], ordered_pose_points[6]]))
            else:
                M = cv2.getPerspectiveTransform(np.float32(clothes_points)[matchStartNum:matchStartNum + pair_num[i]], np.float32(ordered_pose_points)[matchStartNum:matchStartNum + pair_num[i]])

            warped_clothes = cv2.warpPerspective(clothes, M, (N2_, M2_))

        # get masks for blending
        clothes_mask = (warped_clothes[:, :, 3]/255)
        inv_clothes_mask = 1 - clothes_mask
        # for j in range(M2_):
        #     for k in range(N2_):
        #         if warped_clothes[j, k, 3] < 128 :
        #             warped_clothes[j, k, 0:3] = np.array([255, 255, 255])

        # gaussian pyramid for warped clothes
        wc_ge = warped_clothes.copy()
        clothes_gp = [wc_ge]
        for j in range(Nb):
            wc_ge = cv2.pyrDown(wc_ge)
            clothes_gp.append(wc_ge)
        
        # gaussian pyramid for pose
        p_ge = tried_on.copy()
        pose_gp = [p_ge]
        for j in range(Nb):
            p_ge = cv2.pyrDown(p_ge)
            pose_gp.append(p_ge)
        
        # laplacian pyramid for warped clothes
        clothes_lp = [clothes_gp[-1]]
        for j in range(Nb, 0, -1):
            wc_u = cv2.pyrUp(clothes_gp[j])
            wc_le = cv2.subtract(clothes_gp[j-1], wc_u)
            clothes_lp.insert(0, wc_le)
        
        # laplacian pyramid for pose
        pose_lp = [pose_gp[-1]]
        for j in range(Nb, 0, -1):
            p_u = cv2.pyrUp(pose_gp[j])
            p_le = cv2.subtract(pose_gp[j-1], p_u)
            pose_lp.insert(0, p_le)
        
        # combined laplacian pyramid
        combined_lp = []
        for j in range(Nb+1):
            c_e = np.ndarray(pose_lp[j].shape, dtype=np.uint8)
            resize_clothes_mask = cv2.resize(clothes_mask, (pose_lp[j].shape[1], pose_lp[j].shape[0]))
            resize_inv_clothes_mask = cv2.resize(inv_clothes_mask, (pose_lp[j].shape[1], pose_lp[j].shape[0]))
            for k in range(3):
                c_e[:, :, k] = resize_clothes_mask[:, :] * clothes_lp[j][:, :, k] + resize_inv_clothes_mask[:, :] * pose_lp[j][:, :, k]
            combined_lp.append(c_e)
        
        # reconstruction
        tried_on = combined_lp[-1]
        for j in range(Nb-1, -1, -1):
            tried_on = cv2.pyrUp(tried_on)
            tried_on = cv2.add(tried_on, combined_lp[j])

        matchStartNum = matchStartNum + pair_num[i]

    tried_on = cv2.resize(tried_on, (N2, M2))

    return tried_on

import os

if __name__ == '__main__':
    segImgs = []
    clothesSegDir = 'segImage'
    for i in range(0,9):
         img = cv2.imread(os.path.join(clothesSegDir, 'clothes_2_seg_{}.png'.format(i)), cv2.IMREAD_UNCHANGED)
         segImgs.append(np.copy(img))

    humanDir = 'inputImage'
    humanImg = cv2.imread(os.path.join(humanDir, 'human.jpg'), cv2.IMREAD_COLOR)

    clothes_points = []
    corners = []
    for i in range(0, 9):
        #print(i)
        IDX = i
        fName = f"segImage/clothes_2_seg_{IDX}.png"
        img = cv2.imread(fName)
        pnts = detectCorners(img, fName, IDX)
        for j in pnts:
            if isinstance(j, np.ndarray):
                #print('array')
                (a, b) = j
                j = (a, b)
            clothes_points.append(j)
        corners.extend(pnts)
    print(clothes_points)
    print(len(clothes_points))

    human_points = []
    corners = []
    for i in range(0, 9):
        #print(i)
        IDX = i
        fName = f"segImage/human_seg_{IDX}.png"
        img = cv2.imread(fName)
        pnts = detectCorners(img, fName, IDX)
        for j in pnts:
            if isinstance(j, np.ndarray):
                #print('array')
                (a, b) = j
                j = (a, b)
            human_points.append(j)
        corners.extend(pnts)
    print(human_points)
    print(len(human_points))

    '''
    clothes_points = [
        # BODY
        (364, 220),
        (257, 280),
        (283, 330),
        (284, 539),
        (470, 280),
        (450, 306),
        (455, 522),
        (366, 565),
        # RIGHT BRANCHIAL
        (471, 281),
        (451, 307),
        (461, 414),
        (489, 409),
        # RIGHT FOREARM
        (489, 410),
        (461, 415),
        (452, 497),
        (474, 505),
        # LEFT BRANCHIAL
        (256, 281),
        (282, 331),
        (277, 424),
        (239, 423),
        # LEFT FOREARM
        (239, 424),
        (277, 425),
        (277, 548),
        (241, 550),
        # RIGHT THIGH
        (455, 523),
        (366, 566),
        (370, 727),
        (441, 734),
        # RIGHT CALF
        (441, 735),
        (370, 728),
        (376, 886),
        (419, 890),
        # LEFT THIGH
        (284, 540),
        (365, 565),
        (350, 746),
        (283, 756),
        # LEFT CALF
        (283, 757),
        (350, 747),
        (341, 902),
        (292, 907),
    ]

    human_points = [
        # BODY
        (342, 265),
        (276, 292),
        (284, 327),
        (276, 484),
        (404, 291),
        (393, 328),
        (405, 487),
        (343, 519),
        # RIGHT BRANCHIAL
        (405, 292),
        (394, 329),
        (402, 391),
        (432, 390),
        # RIGHT FOREARM
        (432, 391),
        (402, 392),
        (411, 493),
        (425, 479),
        # LEFT BRANCHIAL
        (275, 293),
        (283, 328),
        (280, 398),
        (250, 393),
        # LEFT FOREARM
        (250, 394),
        (280, 399),
        (277, 476),
        (264, 490),
        # RIGHT THIGH
        (405, 488),
        (344, 522),
        (346, 646),
        (397, 656),
        # RIGHT CALF
        (397, 657),
        (346, 647),
        (369, 774),
        (392, 774),
        # LEFT THIGH
        (276, 485),
        (343, 522),
        (342, 654),
        (286, 657),
        # LEFT CALF
        (286, 658),
        (341, 657),
        (309, 769),
        (290, 767),
    ]
    '''
    tried_on = warpClothesSegs(segImgs, humanImg, clothes_points, human_points)

    cv2.imshow('tried_on', tried_on)
    cv2.waitKey(0)
    cv2.destroyAllWindows()