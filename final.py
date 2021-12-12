import cv2
from matplotlib.pyplot import contour
import utils
import numpy as np

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
    print(img.shape)

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


# 옷을 입은 사람 이미지에서 segmentation을 통해 각 부위만 남긴 이미지 9장,
# 옷을 입은 사람 이미지에서 나온 39개의 꼭지점과 15개의 관절, 사람 이미지에서 나온 39개의 꼭지점과 15개의 관절을 반환
#
# input:
#   clothesOnHumanImg: numpy.ndarray (cv2.imread만 한 상태, RGB 순서, size: M1*N1*3)
#   humanImg: numpy.ndarray (RGB 순서, size: M2*N2*3)
# output: (순서대로)
#   segImgs: list[9] (BodyParts(Enum)에 명시된 순서로 해당 부위만 남기고 나머지는 배경처리한 옷 이미지의 배열. 각 element가 M1*N1*4 size (alpha채널 추가))
#   clothesPoints: list[39][2] (옷에서 추출한 꼭지점들, (x, y) 순서)
#   clothesJoints: list[15][2] (옷(을 입은 사람)에서 추출한 관절들)
#   posePoints: list[39][2] (사람에서 추출한 꼭지점들)
#   poseJoints: list[15][2] (사람에서 추출한 관절들)
def imageToSegAndPoints(clothesImg, cGender, humanImg, hGender):
    clothesPoints = parseImage(clothesImg, cGender)
    utils.segmentation(clothesImg, clothesPoints, type="clothes")

    posePoints = parseImage(humanImg, hGender)
    utils.segmentation(humanImg, posePoints, type="body")

    return clothesPoints, posePoints


def detectCorners(img, name):
    contourList, hList = utils.getContourDL(img, 50, "results/temp_contour.jpg")
    bBox = utils.rectangle(contourList)
    _, _, pnts = utils.getConvexHullDL(img, contourList, hList)
    pnts = utils.getExtremitiesDL(bBox)
    cImg = cv2.imread(name)
    for dot in pnts:
        cv2.circle(cImg, (int(dot[0]), int(dot[1])), 7, (255,255,255), 2)
    cv2.imwrite(f"results/final_{IDX}.jpg", cImg)

    return pnts


if __name__ == "__main__":
    man = "data/man-hands-on-waist-full-body.png"
    woman = "data/woman-hands-on-waist-full-body.png"
    wImg = cv2.imread(woman)
    mImg = cv2.imread(man)

    # imageToSegAndPoints(wImg, WOMAN, mImg, MAN)

    # AFTER SEGMENTATION FROM DL:
    corners = []
    for i in range(0, 9):
        print(i)
        IDX = i
        fName = f"data/seg_{IDX}_man2.png"
        img = cv2.imread(fName)

        pnts = detectCorners(img, fName)
        corners.extend(pnts)