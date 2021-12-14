import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# input:
#   clothesPoints: list[39][2] (옷에서 추출한 꼭지점들) 
#   clothesJoints: list[15][2] (옷(을 입은 사람)에서 추출한 관절들) 
#   posePoints: list[39][2] (사람에서 추출한 꼭지점들) 
#   poseJoints: list[15][2] (사람에서 추출한 관절들)

# output: (순서대로)
#   orderedClothesPoints: list[39][2] (대응 순서에 맞게 정리한 옷에서 추출한 꼭지점들. 
#                                       여기서도 BodyParts(Enum)의 순서에 맞게 해주셨으면 합니다! 
#                                       맨 처음이 몸통 7개, 그 다음이 오른쪽 상완 4개, ... 이렇게요. ) 
#   orderedPosePoints: list[39][2] (대응 순서에 맞게 정리한 사람에서 추출한 꼭지점들) 

def fm(p, j): #points, joints
    op = np.zeros((39,2))
    
    p = p.tolist()
    p.sort(key=lambda x:x[1])            # y coordinate value가 작은 것(image에서는 위쪽에 있는 점)이 앞에 오도록 정렬
    
    nj = [-1]*(p.shape[0])                    # nearest joint
    nd = [sys.maxsize]*(p.shape[0])           # nearest distance(to nj)
    
    for k in range(p.shape[0]):
        for i in range(14):        # i = joint number in MPI model. j.shape[0] = 15. 그러나 j[14](몸통 중앙)은 고려 대상 아님
            if (np.sum(np.square(j[i]-p[k])) < nd[k]):
                nd[k] = np.sum(np.square(j[i]-p[k]))
                nj[k] = i
    
    for k in range(p.shape[0]):
        if(nj[k] == 1):
            op[0] = p[k]
        if(nj[k] == 2):
            if(op[9] == [0,0].all() and (op[1] == [0,0]).all()):
                for l in range(k+1, p.shape[0]):
                    if (nj[l] == 2):
                        if(p[k][0] <= p[l][0]):
                            op[9] = p[k]
                        else:
                            op[1] = p[k]
                        continue
            if((not(op[9] == [0,0]).all()) and (op[1] == [0,0]).all()):
                op[1] = p[k]
                continue
            if((not(op[1] == [0,0]).all()) and (op[9] == [0,0]).all()):
                op[9] = p[k]
                continue
            
            if(op[10] == [0,0].all() and (op[2] == [0,0]).all()):
                for l in range(k+1, p.shape[0]):
                    if (nj[l] == 2):
                        if(p[k][0] <= p[l][0]):
                            op[10] = p[k]
                        else:
                            op[2] = p[k]
                        continue
            if((not(op[10] == [0,0]).all()) and (op[2] == [0,0]).all()):
                op[2] = p[k]
                continue
            if((not(op[2] == [0,0]).all()) and (op[10] == [0,0]).all()):
                op[10] = p[k]
                continue
        if(nj[k] == 3):
            if(op[13] == [0,0].all() and (op[7] == [0,0]).all()):
                for l in range(k+1, p.shape[0]):
                    if (nj[l] == 3):
                        if(p[k][0] <= p[l][0]):
                            op[13] = p[k]
                        else:
                            op[7] = p[k]
                        continue
            if((not(op[13] == [0,0]).all()) and (op[7] == [0,0]).all()):
                op[7] = p[k]
                continue
            if((not(op[7] == [0,0]).all()) and (op[13] == [0,0]).all()):
                op[13] = p[k]
                continue
            
            if(op[14] == [0,0].all() and (op[8] == [0,0]).all()):
                for l in range(k+1, p.shape[0]):
                    if (nj[l] == 3):
                        if(p[k][0] <= p[l][0]):
                            op[14] = p[k]
                        else:
                            op[8] = p[k]
                        continue
            if((not(op[14] == [0,0]).all()) and (op[8] == [0,0]).all()):
                op[8] = p[k]
                continue
            if((not(op[8] == [0,0]).all()) and (op[14] == [0,0]).all()):
                op[14] = p[k]
                continue
        if(nj[k] == 4):
            if(op[11] == [0,0].all() and (op[12] == [0,0]).all()):
                for l in range(k+1, p.shape[0]):
                    if (nj[l] == 4):
                        if(p[k][0] <= p[l][0]):
                            op[11] = p[k]
                        else:
                            op[12] = p[k]
                        continue
            if((not(op[11] == [0,0]).all()) and (op[12] == [0,0]).all()):
                op[12] = p[k]
                continue
            if((not(op[12] == [0,0]).all()) and (op[11] == [0,0]).all()):
                op[11] = p[k]
                continue
        if(nj[k] == 5):
            if(op[4] == [0,0].all() and (op[15] == [0,0]).all()):
                for l in range(k+1, p.shape[0]):
                    if (nj[l] == 5):
                        if(p[k][0] <= p[l][0]):
                            op[4] = p[k]
                        else:
                            op[15] = p[k]
                        continue
            if((not(op[4] == [0,0]).all()) and (op[15] == [0,0]).all()):
                op[15] = p[k]
                continue
            if((not(op[15] == [0,0]).all()) and (op[4] == [0,0]).all()):
                op[4] = p[k]
                continue
            
            if(op[5] == [0,0].all() and (op[16] == [0,0]).all()):
                for l in range(k+1, p.shape[0]):
                    if (nj[l] == 5):
                        if(p[k][0] <= p[l][0]):
                            op[5] = p[k]
                        else:
                            op[16] = p[k]
                        continue
            if((not(op[5] == [0,0]).all()) and (op[16] == [0,0]).all()):
                op[16] = p[k]
                continue
            if((not(op[16] == [0,0]).all()) and (op[5] == [0,0]).all()):
                op[5] = p[k]
                continue
        if(nj[k] == 6):
            if(op[17] == [0,0].all() and (op[19] == [0,0]).all()):
                for l in range(k+1, p.shape[0]):
                    if (nj[l] == 6):
                        if(p[k][0] <= p[l][0]):
                            op[17] = p[k]
                        else:
                            op[19] = p[k]
                        continue
            if((not(op[17] == [0,0]).all()) and (op[19] == [0,0]).all()):
                op[19] = p[k]
                continue
            if((not(op[19] == [0,0]).all()) and (op[17] == [0,0]).all()):
                op[17] = p[k]
                continue
            
            if(op[18] == [0,0].all() and (op[20] == [0,0]).all()):
                for l in range(k+1, p.shape[0]):
                    if (nj[l] == 6):
                        if(p[k][0] <= p[l][0]):
                            op[18] = p[k]
                        else:
                            op[20] = p[k]
                        continue
            if((not(op[18] == [0,0]).all()) and (op[20] == [0,0]).all()):
                op[20] = p[k]
                continue
            if((not(op[20] == [0,0]).all()) and (op[18] == [0,0]).all()):
                op[18] = p[k]
                continue
        if(nj[k] == 7):
            if(op[22] == [0,0].all() and (op[21] == [0,0]).all()):
                for l in range(k+1, p.shape[0]):
                    if (nj[l] == 7):
                        if(p[k][0] <= p[l][0]):
                            op[22] = p[k]
                        else:
                            op[21] = p[k]
                        continue
            if((not(op[22] == [0,0]).all()) and (op[21] == [0,0]).all()):
                op[21] = p[k]
                continue
            if((not(op[21] == [0,0]).all()) and (op[22] == [0,0]).all()):
                op[22] = p[k]
                continue
        if(nj[k] == 8):
            if(j[8][0] < p[k][0]):
                if((op[25] == [0,0]).all()):
                    op[25] = p[k]
                    continue
                else:
                    op[31] == p[k]
                    continue
            else:
                if((op[3] == [0,0]).all()):
                    op[3] = p[k]
                    continue
                else:
                    op[23] == p[k]
                    continue
        if(nj[k] == 9):
            if(j[9][0] < p[k][0]):
                if((op[26] == [0,0]).all()):
                    op[26] = p[k]
                    continue
                else:
                    op[29] = p[k]
                    continue
            else:
                if((op[24] == [0,0]).all()):
                    op[24] = p[k]
                    continue
                else:
                    op[27] = p[k]
                    continue
        if(nj[k] == 10):
            if(j[10][0] < p[k][0]):
                op[30] = p[k]
                continue
            else:
                op[28] = p[k]
                continue
        if(nj[k] == 11):
            if(j[11][0] > p[k][0]):
                if((op[31] == [0,0]).all()):
                    op[31] = p[k]
                    continue
                else:
                    op[25] == p[k]
                    continue
            else:
                if((op[6] == [0,0]).all()):
                    op[6] = p[k]
                    continue
                else:
                    op[33] == p[k]
                    continue
        if(nj[k] == 12):
            if(j[12][0] < p[k][0]):
                if((op[34] == [0,0]).all()):
                    op[34] = p[k]
                    continue
                else:
                    op[37] = p[k]
                    continue
            else:
                if((op[32] == [0,0]).all()):
                    op[32] = p[k]
                    continue
                else:
                    op[35] = p[k]
                    continue
        if(nj[k] == 13):
            if(j[13][0] < p[k][0]):
                op[38] = p[k]
                continue
            else:
                op[36] = p[k]
                continue
    
    op = op.tolist()
    return op

def findMatchings(clothesPoints, clothesJoints, posePoints, poseJoints):
    
    orderedClothesPoints = fm(clothesPoints, clothesJoints)
    orderedPosePoints = fm(posePoints, poseJoints)
    
    return orderedClothesPoints, orderedPosePoints