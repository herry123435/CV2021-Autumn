import cv2
from matplotlib.pyplot import contour
import utils
import numpy as np
from enum import Enum

BODY = 0
RIGHT_BRANCHIAL = 1
RIGHT_FOREARM = 2
LEFT_BRANCHIAL = 3
LEFT_FOREARM = 4
RIGHT_THIGH = 5
RIGHT_CALF = 6
LEFT_THIGH = 7
LEFT_CALF = 8

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

VERT = 1
HORI = 0

MAN = 0
WOMAN = 1

def parseBody(img, corners, contours, hulls):
    top, left, right, bLeft, bRight = utils.getExtremities(hulls)
    wCtr = (top[0] + (bRight[0] + bLeft[0])/2)/2

    parts = [[] for j in range(9)]

    # BODY (we assume that at least one side - left or right - for each body part is fully found)
    body = utils.nClosestTo(8, top, corners, VERT)
    neck = body[:2]
    if utils.isSimilar(neck[0][1], neck[1][1]):
        parts[BODY].append((neck[0] + neck[1])/2)
    else:
        # TODO for generalization
        print("PROBLEMO")
    shoulders = body[3:5]
    if utils.isSimilar(shoulders[0][1], shoulders[1][1]):
        parts[BODY].append(shoulders[0])
        parts[BODY].append(shoulders[1])
    else:
        print("PROBLEMO")

    # print result
    cImg = cv2.imread("results/contour.jpg")
    for p in parts:
        for dot in p:
            cv2.circle(cImg, (int(dot[0]), int(dot[1])), 7, (255,255,255), 2)
    cv2.imwrite("results/final.jpg", cImg)

    return parts


def parseWoman(img, corners, contours, hulls):
    parts = [[] for j in range(9)]
    top, left, right, bLeft, bRight = utils.getExtremities(hulls)
    wCtr = (top[0] + (bRight[0] + bLeft[0])/2)/2
    # BODY
    parts[BODY].extend([corners[6], corners[13], corners[21], corners[23], corners[12], corners[5]])
    parts[BODY].append((corners[5] + corners[6])/2)
    # RIGHT_BRANCHIAL
    parts[RIGHT_BRANCHIAL].extend([corners[12], corners[18], right, corners[5]])
    # RIGHT_FOREARM
    parts[RIGHT_FOREARM].extend([corners[18], right])
    parts[RIGHT_FOREARM].append([wCtr + (wCtr - corners[22][0]), corners[23][1]+20])
    parts[RIGHT_FOREARM].append(corners[23])
    # LEFT_BRANCHIAL
    parts[LEFT_BRANCHIAL].extend([corners[6], corners[13], corners[19], left])
    # LEFT_FOREARM
    parts[LEFT_FOREARM].extend([corners[19], left, corners[22], corners[21]])
    # RIGHT_THIGH
    parts[RIGHT_THIGH].extend([corners[24], corners[26]])
    parts[RIGHT_THIGH].append([wCtr + (wCtr - corners[22][0]), corners[23][1]+20])
    parts[RIGHT_THIGH].append([corners[23][0], corners[26][1]])
    # RIGHT_CALF
    parts[RIGHT_CALF].append([corners[23][0], corners[26][1]])
    parts[RIGHT_CALF].extend([corners[26], corners[27], corners[28]])
    # LEFT_THIGH
    parts[LEFT_THIGH].extend([corners[21], corners[24], corners[25]])
    parts[LEFT_THIGH].append([corners[6][0], corners[25][1]])
    # LEFT_CALF
    parts[LEFT_CALF].append([corners[6][0], corners[25][1]])
    parts[LEFT_CALF].extend([corners[25], corners[31], corners[30]])

    # print result
    cImg = cv2.imread("results/contour.jpg")
    for p in parts:
        for dot in p:
            cv2.circle(cImg, (int(dot[0]), int(dot[1])), 7, (255,255,255), 2)
    cv2.imwrite("results/final.jpg", cImg)

    return parts


def parseMan(img, corners, contours, hulls):
    parts = [[] for j in range(9)]
    top, left, right, bLeft, bRight = utils.getExtremities(hulls)
    wCtr = (top[0] + (bRight[0] + bLeft[0])/2)/2
    # BODY
    parts[BODY].append((corners[6] + corners[7])/2)
    parts[BODY].extend([corners[10], corners[16], corners[21], corners[22], corners[15], corners[12]])
    # RIGHT_BRANCHIAL
    parts[RIGHT_BRANCHIAL].extend([corners[12], corners[15], corners[20]])
    parts[RIGHT_BRANCHIAL].append(right)
    # RIGHT_FOREARM
    parts[RIGHT_FOREARM].extend([corners[20], right])
    parts[RIGHT_FOREARM].append([wCtr + (wCtr - corners[23][0]), corners[23][1]])
    parts[RIGHT_FOREARM].append(corners[22])
    # LEFT_BRANCHIAL
    parts[LEFT_BRANCHIAL].extend([corners[10], corners[16], corners[19], left])
    # LEFT_FOREARM
    parts[LEFT_FOREARM].extend([corners[19], left, corners[23], corners[21]])
    # RIGHT_THIGH
    parts[RIGHT_THIGH].extend([corners[24], corners[31], corners[29]])
    parts[RIGHT_THIGH].append([wCtr + (wCtr - corners[23][0]), corners[23][1]])
    # RIGHT_CALF
    parts[RIGHT_CALF].extend([corners[31], corners[29], corners[43], corners[42]])
    # LEFT_THIGH
    parts[LEFT_THIGH].extend([corners[23], corners[24], corners[32], corners[30]])
    # LEFT_CALF
    parts[LEFT_CALF].extend([corners[30], corners[32], corners[40], corners[41]])

    # print result
    cImg = cv2.imread("results/contour.jpg")
    for p in parts:
        for dot in p:
            cv2.circle(cImg, (int(dot[0]), int(dot[1])), 7, (255,255,255), 2)
    cv2.imwrite("results/final.jpg", cImg)

    return parts


def parseImage(img, gender):
    #print(img.shape)

    # 1. Get contour
    contours, hierarchy = utils.getContour(img, visualize=False)
    smoothedImg = cv2.imread("results/contour.jpg")
    validHull, validContours = utils.getConvexHull(smoothedImg, contours, hierarchy)
    corners = utils.getCorners('results/contour.jpg')

    # parseBody(img, corners, validContours, validHull)
    if gender == MAN:
        parts = parseMan(img, corners, validContours, validHull)
    else:
        parts = parseWoman(img, corners, validContours, validHull)

    return parts


# ?????? ?????? ?????? ??????????????? segmentation??? ?????? ??? ????????? ?????? ????????? 9???,
# ?????? ?????? ?????? ??????????????? ?????? 39?????? ???????????? 15?????? ??????, ?????? ??????????????? ?????? 39?????? ???????????? 15?????? ????????? ??????
#
# input:
#   clothesOnHumanImg: numpy.ndarray (cv2.imread??? ??? ??????, RGB ??????, size: M1*N1*3)
#   humanImg: numpy.ndarray (RGB ??????, size: M2*N2*3)
# output: (????????????)
#   segImgs: list[9] (BodyParts(Enum)??? ????????? ????????? ?????? ????????? ????????? ???????????? ??????????????? ??? ???????????? ??????. ??? element??? M1*N1*4 size (alpha?????? ??????))
#   clothesPoints: list[39][2] (????????? ????????? ????????????, (x, y) ??????)
#   clothesJoints: list[15][2] (???(??? ?????? ??????)?????? ????????? ?????????)
#   posePoints: list[39][2] (???????????? ????????? ????????????)
#   poseJoints: list[15][2] (???????????? ????????? ?????????)
def imageToSegAndPoints(clothesImg, cGender, humanImg, hGender):
    clothesPoints = parseImage(clothesImg, cGender)
    utils.segmentation(clothesImg, clothesPoints, type="clothes")

    posePoints = parseImage(humanImg, hGender)
    utils.segmentation(humanImg, posePoints, type="body")

    return clothesPoints, posePoints


def detectCorners(img, name, IDX):
    contourList, _ = utils.getContourDL(img, 50, "results/temp_contour.jpg")
    bBox = utils.rectangle(contourList)
    if IDX == BodyParts.RIGHT_BRANCHIAL.value or IDX == BodyParts.LEFT_BRANCHIAL.value:
        n = 3
    elif IDX == BodyParts.BODY.value:
        # TODO: skip and gather at end
        n = 4
    else:
        n = 4
    pnts = utils.getExtremitiesDL(bBox, n)
    #print(len(pnts))
    if IDX == 0:
        #print(pnts)
        sum1 = tuple(map(sum, zip(pnts[0], pnts[1])))
        sum2 = tuple(map(sum, zip(pnts[1], pnts[2])))
        sum3 = tuple(map(sum, zip(pnts[2], pnts[3])))
        sum4 = tuple(map(sum, zip(pnts[3], pnts[0])))
        pnts.append(tuple([0.5*i for i in sum1]))
        pnts.append(tuple([0.5*i for i in sum2]))
        pnts.append(tuple([0.5*i for i in sum3]))
        pnts.append(tuple([0.5*i for i in sum4]))
        #print(pnts)
    #print(len(pnts))
    cImg = cv2.imread(name)
    for dot in pnts:
        cv2.circle(cImg, (int(dot[0]), int(dot[1])), 7, (255,255,255), 2)
    cv2.imwrite(f"results/final_{IDX}.jpg", cImg)

    return pnts


if __name__ == "__main__":
    #man = "data/man-hands-on-waist-full-body.png"
    #woman = "data/woman-hands-on-waist-full-body.png"
    #wImg = cv2.imread(woman)
    #mImg = cv2.imread(man)

    # imageToSegAndPoints(wImg, WOMAN, mImg, MAN)

    # AFTER SEGMENTATION FROM DL:
    all_cor = []
    corners = []
    for i in range(0, 9):
        #print(i)
        IDX = i
        fName = f"segImage/human_seg_{IDX}.png"
        img = cv2.imread(fName)
        pnts = detectCorners(img, fName)
        for j in pnts:
            all_cor.append(j)
        corners.extend(pnts)
    #print(all_cor)
    #print(len(all_cor))