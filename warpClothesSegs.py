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
    Nb = 6 # height of gaussian pyramid

    total_parts_num = 9
    pair_num = [7, 4, 4, 4, 4, 4, 4, 4, 4] # number of pairs for each body part
    tried_on = humanImg
    M2, N2 = tried_on.shape[::2]

    for i in range(total_parts_num):
        # get each clothes segment
        clothes_seg = segImgs[i].astype(np.uint8)
        M1, N1 = clothes_seg.shape[::2]

        # resize it to size of pose image
        clothes = cv2.resize(clothes_seg, (N2, M2)) # direction check required
        clothes_points = [(int(point[0]*N2/N1), int(point[1]*M2/M1)) for point in ordered_clothes_points]

        tps = cv2.createThinPlateSplineShapeTransformer()
        
        # prepare target points, source points, and match indices for transformation
        target_shape = np.array(ordered_pose_points,np.int32).reshape(1,-1,2)
        source_shape = np.array(clothes_points,np.int32).reshape(1,-1,2)
        N = pair_num[i]
        matches =[]
        for i in range(0, N):
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
        for i in range(Nb-1, 0, -1):
            wc_u = cv2.pyrUp(clothes_gp[i])
            wc_le = cv2.subtract(clothes_gp[i-1], wc_u)
            clothes_lp.append(wc_le)
        
        # laplacian pyramid for pose
        pose_lp = [pose_gp[-1]]
        for i in range(Nb-1, 0, -1):
            p_u = cv2.pyrUp(pose_gp[i])
            p_le = cv2.subtract(pose_gp[i-1], p_u)
            pose_lp.append(p_le)
        
        # combined laplacian pyramid
        combined_lp = []
        for i in range(Nb, -1, -1):
            c_e = np.ndarray(pose_lp[i].shape, dtype=np.uint8)
            for j in range(3):
                c_e[:, :, j] = clothes_mask[:, :] * clothes_lp[i][:, :, j] + inv_clothes_mask[:, :] * pose_lp[i][:, :, j]
            combined_lp.append(c_e)
        
        # reconstruction
        tried_on = combined_lp[0]
        for i in range(1, Nb):
            tried_on = cv2.pyrUp(tried_on)
            tried_on = cv2.add(tried_on, combined_lp[i])

    return tried_on
