import numpy as np
import cv2

def WarpImage_TPS(source,target,img):
	tps = cv2.createThinPlateSplineShapeTransformer()

	source=source.reshape(-1,len(source),2)
	target=target.reshape(-1,len(target),2)

	matches=list()
	for i in range(0,len(source[0])):

		matches.append(cv2.DMatch(i,i,0))

	tps.estimateTransformation(target, source, matches)  # note it is target --> source

	new_img = tps.warpImage(img)

	# get the warp kps in for source and target
	tps.estimateTransformation(source, target, matches)  # note it is source --> target
	# there is a bug here, applyTransformation must receive np.float32 data type
	f32_pts = np.zeros(source.shape, dtype=np.float32)
	f32_pts[:] = source[:]
	transform_cost, new_pts1 = tps.applyTransformation(f32_pts)  # e.g., 1 x 4 x 2
	f32_pts = np.zeros(target.shape, dtype=np.float32)
	f32_pts[:] = target[:]
	transform_cost, new_pts2 = tps.applyTransformation(f32_pts)  # e.g., 1 x 4 x 2

	return new_img, new_pts1, new_pts2

def thin_plate_transform(x,y,offw,offh,imshape,shift_l=-0.05,shift_r=0.05,num_points=5,offsetMatrix=False):
	rand_p=np.random.choice(x.size,num_points,replace=False)
	movingPoints=np.zeros((1,num_points,2),dtype='float32')
	fixedPoints=np.zeros((1,num_points,2),dtype='float32')

	movingPoints[:,:,0]=x[rand_p]
	movingPoints[:,:,1]=y[rand_p]
	fixedPoints[:,:,0]=movingPoints[:,:,0]+offw*(np.random.rand(num_points)*(shift_r-shift_l)+shift_l)
	fixedPoints[:,:,1]=movingPoints[:,:,1]+offh*(np.random.rand(num_points)*(shift_r-shift_l)+shift_l)

	tps=cv2.createThinPlateSplineShapeTransformer()
	good_matches=[cv2.DMatch(i,i,0) for i in range(num_points)]
	tps.estimateTransformation(movingPoints,fixedPoints,good_matches)

	imh,imw=imshape
	x,y=np.meshgrid(np.arange(imw),np.arange(imh))
	x,y=x.astype('float32'),y.astype('float32')
	# there is a bug here, applyTransformation must receive np.float32 data type
	newxy=tps.applyTransformation(np.dstack((x.ravel(),y.ravel())))[1]
	newxy=newxy.reshape([imh,imw,2])

	if offsetMatrix:
		return newxy,newxy-np.dstack((x,y))
	else:
		return newxy

# the correspondences need at least four points

## for origin -> warp
# Zp = np.array([[217, 39], [204, 95], [174, 223], [648, 402]]) # (x, y) in each row
# Zs = np.array([[283, 54], [166, 101], [198, 250], [666, 372]])
# im = cv2.imread('./data/origin.jpg')

# Experiments:
## EXP 1: TPS with four points
# Zp = np.array([[364, 578], [878, 557], [395, 1421], [873, 1420]]) # ul, ur, ll, lr
# man = np.array([[913, 905], [1570, 862], [914, 1697], [1513, 1778]])
# Zs = np.array([
# 	[Zp[0][0], Zp[0][1]],
# 	[Zp[0][0] + (man[1][0] - man[0][0]), Zp[0][1] + (man[1][1] - man[0][1])],
# 	[Zp[0][0] + (man[2][0] - man[0][0]), Zp[0][1] + (man[2][1] - man[0][1])],
# 	[Zp[0][0] + (man[3][0] - man[0][0]), Zp[0][1] + (man[3][1] - man[0][1])]
# ])
# num = 1

## EXP 2: TPS with 6 points (elbows included)
man = np.array([[913, 905], [1570, 862], [914, 1697], [1513, 1778], [788, 1397], [1665, 1381]]) # ul, ur, ll, lr, lelbow, relbow
Zp = np.array([[364, 578], [878, 557], [395, 1421], [873, 1420], [256, 946], [1009, 970]])
Zs = np.array([
	[Zp[0][0], Zp[0][1]],
	[Zp[0][0] + (man[1][0] - man[0][0]), Zp[0][1] + (man[1][1] - man[0][1])],
	[Zp[0][0] + (man[2][0] - man[0][0]), Zp[0][1] + (man[2][1] - man[0][1])],
	[Zp[0][0] + (man[3][0] - man[0][0]), Zp[0][1] + (man[3][1] - man[0][1])],
	[Zp[0][0] + (man[4][0] - man[0][0]), Zp[0][1] + (man[4][1] - man[0][1])],
	[Zp[0][0] + (man[5][0] - man[0][0]), Zp[0][1] + (man[5][1] - man[0][1])]
])
num = 2


r = 15
im = cv2.imread('./data/shirt.jpg')

# print man's points
# manIm = cv2.imread('./data/man_pose1.jpg')
# for p in man:
# 	cv2.circle(manIm, (p[0], p[1]), r, [0, 0, 0], 3)
# cv2.imwrite('./data/man.jpg', manIm)



# draw parallel grids
for y in range(0, im.shape[0], 10):
		im[y, :, :] = 255
for x in range(0, im.shape[1], 10):
		im[:, x, :] = 255

new_im, new_pts1, new_pts2 = WarpImage_TPS(Zp, Zs, im)
new_pts1, new_pts2 = new_pts1.squeeze(), new_pts2.squeeze()
print(new_pts1, new_pts2)

# new_xy = thin_plate_transform(x=Zp[:, 0], y=Zp[:, 1], offw=3, offh=2, imshape=im.shape[0:2], num_points=4)

for p in Zp: # source in red
	cv2.circle(im, (p[0], p[1]), r, [0, 0, 255], 3)
for p in Zs: # target in blue
	cv2.circle(im, (p[0], p[1]), r, [255, 0, 0], 2)
# cv2.imshow('w', im)
cv2.imwrite(f'./data/source_shirt{num}.jpg', im)
# cv2.waitKey(500)


for p in Zs: # target in blue
	cv2.circle(new_im, (p[0], p[1]), r, [255, 0, 0], 2)
for p in new_pts1:
	cv2.circle(new_im, (int(p[0]), int(p[1])), r, [0, 0, 255], 3)
# cv2.imshow('w2', new_im)
cv2.imwrite(f'./data/warped_shirt{num}.jpg', new_im)
# cv2.waitKey(0)

