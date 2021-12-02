from typing_extensions import final
import cv2
from matplotlib.pyplot import contour, draw
import numpy as np

def visualizeImage(title, image, write=False):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    if write:
        cv2.imwrite(f'{title}', image)
    cv2.destroyAllWindows()

def isSimilar(a, b, th=10):
    if abs(a-b) < th:
        return True
    return False

def getContour(image, th=50, outputFileName="results/contour.jpg", visualize=False):
    """Get contour image from normal image. Visualize, save and return the image.

    Args:
        imageFileName (string): file name of input image
        outputFileName (string): file name of output image. Default is contour.jpg
        visualize (bool): show result
    Return:
        np.array(shape=(h, w, 3)): resulting image
    """
    # Convert image to a usage matrix
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img_gray = cv2.blur(img_gray, (5,5))
    img_gray = cv2.bitwise_not(img_gray)
    _, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)

    # Find contour
    contourList, hierarchyList = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    # Remove noisy contours
    contours = []
    th = 1000
    for i in range(len(contourList)):
        area = cv2.contourArea(contourList[i])
        if area >= th:
            contours.append(contourList[i])

    contour_img = np.zeros(image.shape)
    cv2.drawContours(image=contour_img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    print(type(contour_img), contour_img.shape)

    # Output (visualize and save)
    if visualize:
        visualizeImage(outputFileName, contour_img, False)
    cv2.imwrite(outputFileName, contour_img)
    return contourList, hierarchyList


def groupSimilarHull(hull):
    Hth = Wth = 20
    arr = []
    for i, h in enumerate(hull):
        h = [h[0][0], h[0][1]]
        # h = h[0]
        if i == 0:
            arr.append([h])
        else:
            broke = False
            for j, a in enumerate(arr):
                for p in a:
                    if abs(p[0] - h[0]) < Hth and abs(p[1] - h[1]) < Wth:
                        broke = True
                        break
                if broke:
                    break
            if broke:
                arr[j].append(h)
            else:
                arr.append([h])

    l = []
    for a in arr:
        # print(a)
        x = y = 0
        for p in a:
            x += p[0]
            y += p[1]
        x /= len(a)
        y /= len(a)
        l.append([int(x), int(y)])

    return l


# def getConvexHull(contours, hierarchy):
#     # create hull array for convex hull points
#     hull = []

#     # calculate points for each contour
#     for i in range(len(contours)):
#         # creating convex hull object for each contour
#         hull.append(cv2.convexHull(contours[i], False))

#     return hull


def getCorners(fileName):
    img = cv2.imread(fileName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,5,5,0.04)
    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    corners = groupSimilarPoints(corners)
    for i in range(1, len(corners)):
        cv2.circle(img, (int(corners[i,0]), int(corners[i,1])), 7, (255,255,255), 2)
        cv2.putText(img, f"{i}", (int(corners[i,0]), int(corners[i,1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_4)

    # visualizeImage("results/corners.jpg", img, True)
    cv2.imwrite("results/corners.jpg", img)

    return corners

def getConvexHull(src, contours, hierarchy):
    # # Convert image to a usage matrix
    img_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    img_gray = cv2.blur(img_gray, (5,5))
    img_gray = cv2.bitwise_not(img_gray)
    _, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
    # # Find outer contour
    # contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # contour_img = np.zeros(src.shape)
    # cv2.drawContours(image=contour_img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    # print(type(contour_img), contour_img.shape)
    # # utils2.visualizeImage("contours", contour_img, False)

    # create hull array for convex hull points
    hull = []
    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))

    # create an empty black image
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

    validHull = []
    validContours = []
    # remove small areas
    for c, h in zip(contours, hull):
        area = cv2.contourArea(c)
        if area > 1000:
            validHull.append(h)
            validContours.append(c)

    color_contours = (0, 255, 0) # green - color for contours
    color = (255, 0, 0) # blue - color for convex hull
    # draw contours and hull points
    for i in range(len(validContours)):
        print(i)
        # draw ith contour
        cv2.drawContours(drawing, validContours, i, color_contours, 2, 8)
        # draw ith convex hull object
        cv2.drawContours(drawing, validHull, i, color, 2, 8)
        tHull = groupSimilarHull(validHull[i])
        for h in tHull:
            cv2.putText(drawing, f"({i}, {h})", h, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_4)

    # visualizeImage("results/convexHull.jpg", drawing, True)
    cv2.imwrite("results/convexHull.jpg", drawing)

    return validHull, validContours


def groupSimilarPoints(hull):
    Hth = Wth = 20
    arr = []
    for i, h in enumerate(hull):
        h = [h[0], h[1]]
        # h = h[0]
        if i == 0:
            arr.append([h])
        else:
            broke = False
            for j, a in enumerate(arr):
                for p in a:
                    if abs(p[0] - h[0]) < Hth and abs(p[1] - h[1]) < Wth:
                        broke = True
                        break
                if broke:
                    break
            if broke:
                arr[j].append(h)
            else:
                arr.append([h])

    for i, a in enumerate(arr):
        if len(a) > 1:
            x = y = 0
            for p in a:
                x += p[0]
                y += p[1]
            arr[i] = [x/len(a), y/len(a)]
        else:
            arr[i] = a[0]

    arr = np.array(arr)

    return arr

def getExtremities(hulls):
    bodyIdx = maxi = 0
    for i, h in enumerate(hulls):
        area = cv2.contourArea(h)
        if area > maxi:
            maxi = area
            bodyIdx = i
    c = hulls[bodyIdx]
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    bottom = []
    print(extBot[1])
    for h in hulls[bodyIdx]:
        h = h[0]
        if isSimilar(h[1],extBot[1]):
            bottom.append(h)

    extBotLeft = extBotRight = extBot
    for b in bottom:
        if b[0] <= extBotLeft[0]:
            extBotLeft = b
        elif b[0] >= extBotRight[0]:
            extBotRight = b

    return extTop, extLeft, extRight, extBotLeft, extBotRight

def nClosestTo(n, p, arr, axis):
    sorted = arr[arr[:, axis].argsort()]
    simIdx = 0
    th0 = th1 = 10
    for i, s in enumerate(sorted):
        if isSimilar(s[0], p[0], th0) and isSimilar(s[1], p[1], th1):
            if abs(s[0] - p[0]) < th0 and abs(s[1] - p[1]) < th1:
                th0 = abs(s[0] - p[0])
                th1 = abs(s[1] - p[1])
                simIdx = i
    ret = []
    i = 0
    while(len(ret) != n and i < len(sorted)):
        i += 1
        if simIdx - i < 0:
            if simIdx + i < len(sorted):
                ret.append(sorted[simIdx + i])
                continue
            else:
                break
        if simIdx + i >= len(sorted):
            if simIdx - i >= 0:
                ret.append(sorted[simIdx - i])
                continue
            else:
                break

        if sorted[simIdx][axis] - sorted[simIdx-i][axis] < sorted[simIdx+i][axis] - sorted[simIdx][axis]:
            ret.append(sorted[simIdx-i])
        else:
            ret.append(sorted[simIdx+i])

    return ret


def segmentation(image, parts, type="clothes"):
    for i, part in enumerate(parts):
        arr = []
        for p in part:
            arr.append([[p[0], p[1]]])
        final = np.array(arr)
        cpy = image.copy()
        mask = np.zeros(cpy.shape)
        # cv2.fillConvexPoly(cpy, np.int32(final), (0,255,255))
        cv2.drawContours(mask, np.int32([final]), -1, (0, 255, 255), thickness=cv2.FILLED, lineType=cv2.LINE_AA)
        cv2.imwrite(f"results/{type}_{i}.png", mask)

        mask = cv2.imread(f"results/{type}_{i}.png")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cpy = cv2.cvtColor(cpy, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        dst = cv2.bitwise_and(cpy, cpy, mask=mask)
        cv2.imwrite(f"results/{type}_{i}.png", dst)

        src = cv2.imread(f"results/{type}_{i}.png")
        tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
        b, g, r = cv2.split(src)
        rgba = [b,g,r, alpha]
        dst = cv2.merge(rgba,4)
        cv2.imwrite(f"results/{type}_{i}.png", dst)