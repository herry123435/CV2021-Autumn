# Note
# 1. 각 함수의 parameter 이름은 이해를 돕기 위한 것입니다. 당연히 다르게 구현하셔도 되구요. 
#   여기 있는 함수들은 나중에 지우고 다른 파일에서 그대로 import하여 적용하거나 하면 되겠습니다.  
# 2. 신체에서 각 파트의 순서는 정확히 정하지 않은 것 같아 제 임의대로 BodyParts(Enum)에 정했습니다. 
#   혹시 다른 순서가 편하시면 합의 하에 나중에 바꿔도 좋습니다. 
# 3. 제가 회의 내용을 정확히 이해하지 못한 부분이 있을 수 있으니 본인 파트에서 이상한 점이 있으면 알려주시면 감사하겠습니다. 

from enum import Enum
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from jointDetection import *



# Body parts order
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

# 선영님/현수님
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
def imageToSegAndPoints(clothesOnHumanImg, humanImg):
    return None, None, None, None, None

# 민재님
# 옷을 입은 사람 이미지에서 나온 꼭지점과 사람 이미지에서 나온 꼭지점 간의 대응 관계를 찾아내어 순서대로 정리하여 반환
# 
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
def findMatchings(clothesPoints, clothesJoints, posePoints, poseJoints):
    return None, None

# 경서
# 9장의 부위 이미지와 사람 이미지를 꼭지점 대응 관계를 기반으로 워핑하여 결과 이미지 반환
# 
# input: 
#   segImgs: list[9] (BodyParts(Enum) 순서)
#   humanImg: numpy.ndarray (RGB 순서, size: M2*N2*3)
#   orderedClothesPoints: list[39][2] (대응 순서에 맞게 정리한 옷에서 추출한 꼭지점들) 
#   clothesJoints: list[15][2] (옷(을 입은 사람)에서 추출한 관절들) 
#   orderedPosePoints: list[39][2] (대응 순서에 맞게 정리한 사람에서 추출한 꼭지점들) 
#   poseJoints: list[15][2] (사람에서 추출한 관절들) 
# output:
#   clothesTriedOn: numpy.ndarray (최종 결과물, RGB 순서, size: M2*M2*3)
def warpClothesSegs(segImgs, humanImg, orderedClothesPoints, clothesJoints, orderedPosePoints, poseJoints):
    return np.zeros((1280, 1280, 3))

if __name__ == '__main__' :
    clothesDir = 'clothesInput'
    humanDir = 'humanInput'
    clothesSegDir = 'clothesOutput'

    # get Input
    clothesOnHumanSrc = 'manWithSweater.jpeg' #'clothesInput.jpeg'
    humanSrc = 'humanPose01.jpeg'#'humanInput.jpeg'
    clothesSegment = 'seg_0_clothes.jpg'

    # read image
    clothesOnHumanImg = cv2.cvtColor(cv2.imread(os.path.join(clothesDir, clothesOnHumanSrc), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    humanImg = cv2.cvtColor(cv2.imread(os.path.join(humanDir, humanSrc), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # Task 1: 선영/현수
    segImgs, clothesPoints, clothesJoints, posePoints, poseJoints = imageToSegAndPoints(clothesOnHumanImg, humanImg)

    # Task 1-1: 선영
    img = cv2.imread(os.path.join(clothesSegDir, clothesSegment), cv2.IMREAD_COLOR)
    segImgs = np.zeros((9, img.shape[0], img.shape[1], 4))
    for i in range(0,9):
        img = cv2.imread(os.path.join(clothesSegDir, 'seg_{}_clothes.jpg'.format(i)), cv2.IMREAD_COLOR)
        segImgs[i] = np.copy(img)
    clothesJoints = detectJoint(clothesOnHumanImg)
    poseJoints = detectJoint(humanImg)

    # Task 2: 민재
    orderedClothesPoints, orderedPostPoints = findMatchings(clothesPoints, clothesJoints, posePoints, poseJoints)

    # Task 3: 경서
    clothesTriedOn = warpClothesSegs(segImgs, humanImg, orderedClothesPoints, clothesJoints, orderedPostPoints, poseJoints)

    # show result
    plt.subplot(1, 3, 1), plt.imshow(clothesOnHumanImg)
    plt.title("clothes"), plt.axis('off')
    plt.subplot(1, 3, 2), plt.imshow(humanImg)
    plt.title("human pose"), plt.axis('off')
    plt.subplot(1, 3, 3), plt.imshow(clothesTriedOn)
    plt.title("tried on"), plt.axis('off')
    plt.tight_layout()
    plt.show()

    # save result
    resultImg = cv2.cvtColor(clothesTriedOn, cv2.COLOR_RGB2BGR)
    cv2.imwrite('virtualTryOnResult.jpeg', resultImg)
