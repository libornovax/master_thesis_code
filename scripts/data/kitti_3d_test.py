"""
Script for translating the KITTI 3D bounding box annotation format into the BB3TXT data format.

----------------------------------------------------------------------------------------------------
python kitti2bb3txt.py path_labels path_images outfile.bb3txt
----------------------------------------------------------------------------------------------------
"""

__date__   = '04/12/2017'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'


import argparse
import os
import numpy as np
import cv2

from mappings.utils import LabelMappingManager
from shared.geometry import R3x3_y, t3x1, Rt4x4


####################################################################################################
#                                           DEFINITIONS                                            # 
####################################################################################################

# IMPORTANT !!
# The labels must translate precisely into the numbers in the kitti.yaml mapping file!
LABELS = {
	'Car': 1,
	'Van': 2,
	'Truck': 3,
	'Pedestrian': 4,
	'Person_sitting': 5,
	'Cyclist': 6,
	'Tram': 7,
	# Throw away 'Misc' and 'DontCare'
}

# Initialize the LabelMappingManager
LMM     = LabelMappingManager()
MAPPING = LMM.get_mapping('kitti')


####################################################################################################
#                                            SETTINGS                                              # 
####################################################################################################

# y coordinate of the ground plane
GROUND = 2.1


####################################################################################################
#                                            FUNCTIONS                                             # 
####################################################################################################

def read_camera_matrix(line):
	"""
	Reads a camera matrix P (3x4) stored in the row-major scheme.

	Input:
		line: Row-major stored matrix separated by spaces, first element is the matrix name
	Returns:
		camera matrix P 4x4
	"""
	data = line.split(' ')

	if data[0] != 'P2:':
		print('ERROR: We need left camera matrix (P2)!')
		exit(1)

	P = np.asmatrix([[float(data[1]), float(data[2]),  float(data[3]),  float(data[4])],
		             [float(data[5]), float(data[6]),  float(data[7]),  float(data[8])],
		             [float(data[9]), float(data[10]), float(data[11]), float(data[12])]])
	return P


def extract_3D_bb(data, P, image_ground):
	"""
	Extract 3D bounding box coordinates in the image from the KITTI labels.

	Input:
		data: One split line of the label file (line.split(' '))
		P:    3x4 camera projection matrix
	Returns:
		matrix of corners: fbr, rbr, fbl, rbl, ftr, rtr, ftl, rtl
	"""
	# Object dimensions
	h  = float(data[8])
	w  = float(data[9])
	l  = float(data[10])
	# Position of the center point on the ground plane (xz plane)
	cx = float(data[11])
	cy = float(data[12])
	cz = float(data[13])
	# Rotation of the object around y
	ry = float(data[14])

	# 3D box corners - careful, the coordinate system of the car is that x points
	# forward, not z! (It is rotated by 90deg with respect to the camera one)
	#                 fbr, rbr,   fbl, rbl,  ftr,  rtr,  ftl, rtl
	X = np.asmatrix([[l/2, -l/2,  l/2, -l/2, l/2,  -l/2, l/2, -l/2],
		             [0,    0,    0,   0,    -h,   -h,   -h,  -h],
		             [-w/2, -w/2, w/2, w/2,  -w/2, -w/2, w/2, w/2],
		             [1,    1,    1,   1,    1,    1,    1,   1]])
	# Rotate the 3D box around y axis and translate it to the correct position in the camera frame
	X = Rt4x4(R3x3_y(ry), t3x1(cx, cy, cz)) * X

	print(X[:,0])

	x = P * X
	# x is in homogeneous coordinates -> get u, v
	x = x / x[2,:]
	x = x[0:2,:]

	
	# Plot the boxes to the ground plane
	X = 10*X
	X[0,:] += 500
	X[2,:] = 1000 - X[2,:]
	cv2.line(image_ground, (int(X[0,0]), int(X[2,0])), (int(X[0,1]), int(X[2,1])), (0,180,255))
	cv2.line(image_ground, (int(X[0,1]), int(X[2,1])), (int(X[0,3]), int(X[2,3])), (0,180,255))
	cv2.line(image_ground, (int(X[0,3]), int(X[2,3])), (int(X[0,2]), int(X[2,2])), (0,180,255))
	cv2.line(image_ground, (int(X[0,2]), int(X[2,2])), (int(X[0,0]), int(X[2,0])), (0,180,255))


	return x


def reconstruct_ground_X(u, v, KR_3x3_inv, C_3x1):
	"""
	"""
	x_3x1 = np.asmatrix([[u], [v], [1.0]])
	X_d_3x1 = KR_3x3_inv * x_3x1  # Direction of X from the camera center

	# Intersect the ground plane
	Xy = GROUND
	lm = (Xy - C_3x1[1,0]) / X_d_3x1[1,0]
	Xx = C_3x1[0,0] + lm * X_d_3x1[0,0]
	Xz = C_3x1[2,0] + lm * X_d_3x1[2,0]

	return np.asmatrix([[Xx],[Xy],[Xz]])


def project_X_to_x(X, P_3x4):
	"""
	"""
	X_4x1 = np.asmatrix([[X[0,0]], [X[1,0]], [X[2,0]], [1.0]])
	x = P_3x4 * X_4x1;
	x = x[0:2,0] / x[2,0];

	return x


def reconstruct_3D_bb(fblx, fbly, fbrx, fbry, rblx, rbly, ftly, P_3x4, image, image_ground):
	"""
	Reconstructs the 3D bounding box the same way as we will do it in a detector.

	Input:
		fblx, fbly, fbrx, fbry, rblx, rbly, ftly: coordinates of bb corners in the image
		P:     4x3 camera matrix
		image: image for plotting the bounding box
	"""
	P_4x4 = np.identity(4)
	P_4x4[0:3, 0:4] = P_3x4
	# print(P_4x4)

	KR_3x3_inv = np.linalg.inv(P_3x4[0:3,0:3])
	C_3x1 = - KR_3x3_inv * P_3x4[0:3,3]
	# P_3x4_inv = np.asmatrix(np.zeros((3,4)))
	# P_3x4_inv[0:3,0:3] = KR_3x3_inv
	# P_3x4_inv[0:3,3] = C_3x1

	# print('Center check:')
	# print(P_3x4 * np.asmatrix([[C_3x1[0,0]], [C_3x1[1,0]], [C_3x1[2,0]], [1.0]]))


	FBL = reconstruct_ground_X(fblx, fbly, KR_3x3_inv, C_3x1)
	FBR = reconstruct_ground_X(fbrx, fbry, KR_3x3_inv, C_3x1)
	RBL = reconstruct_ground_X(rblx, rbly, KR_3x3_inv, C_3x1)
	RBR = FBR + (RBL-FBL)
	
	print('Recovered 3D coordinates FBL: ' + str(FBL))
	print('Recovered 3D coordinates FBR: ' + str(FBR))
	print('Recovered 3D coordinates RBL: ' + str(RBL))
	
	fbl_r = project_X_to_x(FBL, P_3x4)
	fbr_r = project_X_to_x(FBR, P_3x4)
	rbl_r = project_X_to_x(RBL, P_3x4)

	print(str(fblx) + ', ' + str(fbly) + ' -> ' + str(fbl_r[0,0]) + ', ' + str(fbl_r[1,0]))
	print(str(fbrx) + ', ' + str(fbry) + ' -> ' + str(fbr_r[0,0]) + ', ' + str(fbr_r[1,0]))
	print(str(rblx) + ', ' + str(rbly) + ' -> ' + str(rbl_r[0,0]) + ', ' + str(rbl_r[1,0]))

	# cv2.circle(image, (int(x[0,0]), int(x[1,0])), 10, (255,80,255), -1)
	# cv2.circle(image, (int(fblx), int(fbly)), 5, (0,177,251), -1)


	# Draw the bounding box to the ground plane
	FBL = 10*FBL
	FBL[0,:] += 500
	FBL[2,:] = 1000 - FBL[2,:]
	FBR = 10*FBR
	FBR[0,:] += 500
	FBR[2,:] = 1000 - FBR[2,:]
	RBL = 10*RBL
	RBL[0,:] += 500
	RBL[2,:] = 1000 - RBL[2,:]
	RBR = 10*RBR
	RBR[0,:] += 500
	RBR[2,:] = 1000 - RBR[2,:]
	cv2.line(image_ground, (int(FBL[0,0]), int(FBL[2,0])), (int(FBR[0,0]), int(FBR[2,0])), (0,255,0), 2)
	cv2.line(image_ground, (int(FBL[0,0]), int(FBL[2,0])), (int(RBL[0,0]), int(RBL[2,0])), (255,0,0), 2)
	cv2.line(image_ground, (int(RBL[0,0]), int(RBL[2,0])), (int(RBR[0,0]), int(RBR[2,0])), (0,0,255))
	cv2.line(image_ground, (int(FBR[0,0]), int(FBR[2,0])), (int(RBR[0,0]), int(RBR[2,0])), (255,0,0))






def process_image(path_image, path_label_file, path_calib_file):
	"""
	Processes one image from the dataset.

	Input:
		path_image:      Path to the image file
		path_label_file: Path to the label file with KITTI labels
		path_calib_file: Path to the calibration file for this image
	"""
	with open(path_label_file, 'r') as infile_label, open(path_calib_file, 'r') as infile_calib:
		# Read camera calibration matrices
		for line in infile_calib:
			if line[:2] == 'P2':
				P = read_camera_matrix(line.rstrip('\n'))


		image = cv2.imread(path_image)
		image_ground = np.zeros((1000, 1000, 3), dtype=np.uint8)
		cv2.line(image_ground, (500,0), (500,1000), (50,50,50), 2)

		# Read the objects
		for line in infile_label:
			line = line.rstrip('\n')
			data = line.split(' ')
			
			# First element of the data is the label. We don't want to process 'Misc' and
			# 'DontCare' labels
			if data[0] == 'Misc' or data[0] == 'DontCare': continue

			# Check label, if required
			if MAPPING[LABELS[data[0]]] != 'car': continue

			# We do not want to include objects, which are occluded or truncated too much
			if filter and (int(data[2]) >= 2 or float(data[1]) > 0.75): continue 

			# Extract image coordinates (positions) of 3D bounding box corners, the corners are
			# in the following order: fbr, rbr, fbl, rbl, ftr, rtr, ftl, rtl
			x = extract_3D_bb(data, P, image_ground)

			min_uv = np.min(x, axis=1)  # xmin, ymin
			max_uv = np.max(x, axis=1)  # xmax, ymax

			# The size of an image in KITTI is 1250x375. If the bounding box is significantly
			# larger, discard it - probably just some large distortion from camera
			if max_uv[1,0]-min_uv[1,0] > 700 or max_uv[0,0]-min_uv[0,0] > 1500:
				continue

			# Front
			cv2.line(image, (int(x[0,0]), int(x[1,0])), (int(x[0,2]), int(x[1,2])), (0,255,0), 3)
			cv2.line(image, (int(x[0,4]), int(x[1,4])), (int(x[0,6]), int(x[1,6])), (0,255,0))
			cv2.line(image, (int(x[0,0]), int(x[1,0])), (int(x[0,4]), int(x[1,4])), (0,255,0))
			cv2.line(image, (int(x[0,2]), int(x[1,2])), (int(x[0,6]), int(x[1,6])), (0,255,0), 3)
			# Rear
			cv2.line(image, (int(x[0,1]), int(x[1,1])), (int(x[0,3]), int(x[1,3])), (0,0,255))
			cv2.line(image, (int(x[0,5]), int(x[1,5])), (int(x[0,7]), int(x[1,7])), (0,0,255))
			cv2.line(image, (int(x[0,1]), int(x[1,1])), (int(x[0,5]), int(x[1,5])), (0,0,255))
			cv2.line(image, (int(x[0,3]), int(x[1,3])), (int(x[0,7]), int(x[1,7])), (0,0,255))
			# Connections
			cv2.line(image, (int(x[0,0]), int(x[1,0])), (int(x[0,1]), int(x[1,1])), (255,0,0))
			cv2.line(image, (int(x[0,2]), int(x[1,2])), (int(x[0,3]), int(x[1,3])), (255,0,0), 3)
			cv2.line(image, (int(x[0,4]), int(x[1,4])), (int(x[0,5]), int(x[1,5])), (255,0,0))
			cv2.line(image, (int(x[0,6]), int(x[1,6])), (int(x[0,7]), int(x[1,7])), (255,0,0))


			# -- RECONSTRUCTION -- #
			# Reconstruct the 3D bounding box only from the data we have from a detector
			reconstruct_3D_bb(x[0,2], x[1,2], x[0,0], x[1,0], x[0,3], x[1,3], x[1,6], P, image, image_ground)


		# Show image
		cv2.imshow('img', image)
		cv2.imshow('Ground plane', image_ground)
		cv2.waitKey()




def translate_file(path_labels, path_images):
	"""
	Runs the translation of the KITTI 3d bounding box label format into the BB3TXT format.

	Input:
		path_labels: Path to the "label_2" folder of the KITTI dataset
		path_images: Path to the "image_2" folder with images from the KITTI dataset
	"""

	# Get the list of all label files in the directory
	filenames = [f for f in os.listdir(path_labels) if os.path.isfile(os.path.join(path_labels, f))]

	if len(filenames) != 7481:
		print('Wrong number (%d) of files in the KITTI dataset! Should be 7481.'%(len(filenames)))
		return

	# Read each file and its calibration matrix
	for f in filenames:
		path_label_file = os.path.join(path_labels, f)
		path_calib_file = os.path.join(path_labels.rstrip('/').rstrip('label_2'), 'calib', f)

		if not os.path.exists(path_calib_file):
			print('ERROR: We need camera calibration matrices "%s"'%(path_calib_file))
			exit(1)

		path_image = os.path.join(path_images, os.path.splitext(f)[0]) + '.png'
		if not os.path.isfile(path_image):
			print('WARNING: Image "%s" does not exist!'%(path_image))

		process_image(path_image, path_label_file, path_calib_file)



####################################################################################################
#                                               MAIN                                               # 
####################################################################################################

def parse_arguments():
	"""
	Parse input options of the script.
	"""
	parser = argparse.ArgumentParser(description='Convert KITTI label files into BBTXT.')
	parser.add_argument('path_labels', metavar='path_labels', type=str,
						help='Path to the "label_2" folder of the KITTI dataset')
	parser.add_argument('path_images', metavar='path_images', type=str,
						help='Path to the "image_2" folder of the KITTI dataset')


	args = parser.parse_args()

	if not os.path.exists(args.path_labels):
		print('Input path "%s" does not exist!'%(args.path_labels))
		parser.print_help()
		exit(1)
	if not os.path.exists(args.path_images):
		print('Input path "%s" does not exist!'%(args.path_images))
		parser.print_help()
		exit(1)

	return args


def main():
	args = parse_arguments()
	translate_file(args.path_labels, args.path_images)

	args.outfile.close()


if __name__ == '__main__':
    main()


