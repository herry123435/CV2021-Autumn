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

def warpClothesSegs(segImgs, humanImg, ordered_clothes_points, ordered_pose_points):
    # hyper parameter #
    Nb = 2 # height of gaussian pyramid
    divider = pow(2, Nb)

    total_parts_num = 2
    pair_num = [7, 4, 4, 4, 4, 4, 4, 4, 4] # number of pairs for each body part
    matchStartNum = 0

    tried_on = humanImg
    M2, N2 = tried_on.shape[:2]

    M2_, N2_ = (M2 // divider) * divider, (N2 // divider) * divider
    tried_on = cv2.resize(tried_on, (N2_, M2_))
    target_shape = np.array(ordered_pose_points,np.int32).reshape(1,-1,2)

    for i in range(total_parts_num):
        # get each clothes segment
        clothes_seg = segImgs[i].astype(np.uint8)
        M1, N1 = clothes_seg.shape[:2]

        # resize it to size of pose image
        clothes = cv2.resize(clothes_seg, (N2_, M2_)) # direction check required
        clothes_points = [(int(point[0]*N2_/N1), int(point[1]*M2_/M1)) for point in ordered_clothes_points]

        tps = cv2.createThinPlateSplineShapeTransformer()
        
        # prepare target points, source points, and match indices for transformation
        source_shape = np.array(clothes_points,np.int32).reshape(1,-1,2)
        matches =[]
        for i in range(matchStartNum, matchStartNum + pair_num[i]):
            matches.append(cv2.DMatch(i,i,0))

        # warp clothes with the obtained transformation
        tps.estimateTransformation(target_shape, source_shape, matches)
        warped_clothes = tps.warpImage(clothes)

        # get masks for blending
        clothes_mask = (warped_clothes[:, :, 3]/255)
        inv_clothes_mask = 1 - clothes_mask

        # gaussian pyramid for warped clothes
        wc_ge = warped_clothes.copy()
        clothes_gp = [wc_ge]
        for i in range(Nb):
            wc_ge = cv2.pyrDown(wc_ge)
            clothes_gp.append(wc_ge)
        
        # gaussian pyramid for pose
        p_ge = tried_on.copy()
        pose_gp = [p_ge]
        for i in range(Nb):
            p_ge = cv2.pyrDown(p_ge)
            pose_gp.append(p_ge)
        
        # laplacian pyramid for warped clothes
        clothes_lp = [clothes_gp[-1]]
        for i in range(Nb, 0, -1):
            wc_u = cv2.pyrUp(clothes_gp[i])
            wc_le = cv2.subtract(clothes_gp[i-1], wc_u)
            clothes_lp.insert(0, wc_le)
        
        # laplacian pyramid for pose
        pose_lp = [pose_gp[-1]]
        for i in range(Nb, 0, -1):
            p_u = cv2.pyrUp(pose_gp[i])
            p_le = cv2.subtract(pose_gp[i-1], p_u)
            pose_lp.insert(0, p_le)
        
        # combined laplacian pyramid
        combined_lp = []
        for i in range(Nb+1):
            c_e = np.ndarray(pose_lp[i].shape, dtype=np.uint8)
            resize_clothes_mask = cv2.resize(clothes_mask, (pose_lp[i].shape[1], pose_lp[i].shape[0]))
            resize_inv_clothes_mask = cv2.resize(inv_clothes_mask, (pose_lp[i].shape[1], pose_lp[i].shape[0]))
            for j in range(3):
                c_e[:, :, j] = resize_clothes_mask[:, :] * clothes_lp[i][:, :, j] + resize_inv_clothes_mask[:, :] * pose_lp[i][:, :, j]
            combined_lp.append(c_e)
        
        # reconstruction
        tried_on = combined_lp[-1]
        for i in range(Nb-1, -1, -1):
            tried_on = cv2.pyrUp(tried_on)
            tried_on = cv2.add(tried_on, combined_lp[i])

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
        # LEFT BRANCHIAL
        (418, 229),
        (401, 270),
        (406, 364),
        (451, 361),
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
        # LEFT BRANCHIAL
        (827, 494),
        (746, 608),
        (763, 812),
        (863, 802),
    ]

    tried_on = warpClothesSegs(segImgs, humanImg, clothes_points, human_points)

    cv2.imshow('tried_on', tried_on)
    cv2.waitKey(0)
    cv2.destroyAllWindows()