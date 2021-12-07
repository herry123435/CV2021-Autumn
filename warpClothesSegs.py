from enum import Enum
import numpy as np
import cv2

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
    pair_num = [7, 4, 4, 4, 4, 4, 4, 4, 4] # number of pairs for each body part
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
    clothesSegDir = 'clothesOutput'
    for i in range(0,9):
         img = cv2.imread(os.path.join(clothesSegDir, 'seg_{}_clothes.png'.format(i)), cv2.IMREAD_UNCHANGED)
         segImgs.append(np.copy(img))

    humanDir = 'humanInput'
    humanImg = cv2.imread(os.path.join(humanDir, 'humanPose01.jpeg'), cv2.IMREAD_COLOR)

    clothes_points = [
        # BODY
        (317, 188),
        (225, 243),
        (239, 285),
        (237, 426),
        (417, 228),
        (400, 269),
        (398, 423),
        # RIGHT BRANCHIAL
        (417, 228),
        (400, 269),
        (406, 364),
        (451, 361),
        # RIGHT FOREARM
        (451, 361),
        (406, 364),
        (402, 514),
        (451, 510),
        # LEFT BRANCHIAL
        (225, 243),
        (239, 285),
        (236, 376),
        (201, 375),
        # LEFT FOREARM
        (201, 375),
        (236, 376),
        (227, 516),
        (205, 521),
    ]

    human_points = [
        # BODY
        (564, 333),
        (286, 479),
        (381, 605),
        (368, 929),
        (826, 493),
        (745, 607),
        (723, 929),
        # RIGHT BRANCHIAL
        (826, 493),
        (745, 607),
        (763, 812),
        (863, 802),
        # RIGHT FOREARM
        (863, 802),
        (763, 812),
        (807, 1090),
        (862, 1088),
        # LEFT BRANCHIAL
        (286, 479),
        (381, 605),
        (327, 814),
        (227, 801),
        # LEFT FOREARM
        (227, 801),
        (327, 814),
        (292, 1081),
        (234, 1090),
    ]

    tried_on = warpClothesSegs(segImgs, humanImg, clothes_points, human_points)

    cv2.imshow('tried_on', tried_on)
    cv2.waitKey(0)
    cv2.destroyAllWindows()