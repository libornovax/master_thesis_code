"""
Script for translating the KITTI 3D bounding box annotation format into the BB3TXT data format.

A BB3TXT file is formatted like this:
filename label confidence xmin ymin xmax ymax fblx fbly fbrx fbry rblx rbly ftly
filename label confidence xmin ymin xmax ymax fblx fbly fbrx fbry rblx rbly ftly
filename label confidence xmin ymin xmax ymax fblx fbly fbrx fbry rblx rbly ftly
...

----------------------------------------------------------------------------------------------------
python kitti2bb3txt.py path_labels path_images outfile.bb3txt
----------------------------------------------------------------------------------------------------
"""

__date__   = '03/17/2017'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'


import argparse
import os
import numpy as np
import cv2

from mappings.utils import LabelMappingManager
from mappings.utils import available_categories
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


def extract_3D_bb(data, P):
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
	x = P * X
	# x is in homogeneous coordinates -> get u, v
	x = x / x[2,:]
	x = x[0:2,:]

	# image = cv2.imread(path_image)
	# # Front
	# cv2.line(image, (int(x[0,0]), int(x[1,0])), (int(x[0,2]), int(x[1,2])), (0,255,0), 3)
	# cv2.line(image, (int(x[0,4]), int(x[1,4])), (int(x[0,6]), int(x[1,6])), (0,255,0))
	# cv2.line(image, (int(x[0,0]), int(x[1,0])), (int(x[0,4]), int(x[1,4])), (0,255,0))
	# cv2.line(image, (int(x[0,2]), int(x[1,2])), (int(x[0,6]), int(x[1,6])), (0,255,0), 3)
	# # Rear
	# cv2.line(image, (int(x[0,1]), int(x[1,1])), (int(x[0,3]), int(x[1,3])), (0,0,255))
	# cv2.line(image, (int(x[0,5]), int(x[1,5])), (int(x[0,7]), int(x[1,7])), (0,0,255))
	# cv2.line(image, (int(x[0,1]), int(x[1,1])), (int(x[0,5]), int(x[1,5])), (0,0,255))
	# cv2.line(image, (int(x[0,3]), int(x[1,3])), (int(x[0,7]), int(x[1,7])), (0,0,255))
	# # Connections
	# cv2.line(image, (int(x[0,0]), int(x[1,0])), (int(x[0,1]), int(x[1,1])), (255,0,0))
	# cv2.line(image, (int(x[0,2]), int(x[1,2])), (int(x[0,3]), int(x[1,3])), (255,0,0), 3)
	# cv2.line(image, (int(x[0,4]), int(x[1,4])), (int(x[0,5]), int(x[1,5])), (255,0,0))
	# cv2.line(image, (int(x[0,6]), int(x[1,6])), (int(x[0,7]), int(x[1,7])), (255,0,0))
	# # Show image
	# cv2.imshow('img', image)
	# cv2.waitKey()

	return x


def flip_3D_bb(x, image_width):
	"""
	Flips the annotation of the image around y axis.

	Input:
		x:           coordinates of points fbr, rbr, fbl, rbl, ftr, rtr, ftl, rtl
		image_width: width of the flipped image
	Return:
		x - flipped coordinates
	"""
	# First flip the x coordinates of the points
	x[0,:] = image_width - x[0,:]

	# Now switch left and right points
	x_out = np.matrix(np.copy(x))
	x_out[:,0] = x[:,2]
	x_out[:,1] = x[:,3]
	x_out[:,2] = x[:,0]
	x_out[:,3] = x[:,1]
	x_out[:,4] = x[:,6]
	x_out[:,5] = x[:,7]
	x_out[:,6] = x[:,4]
	x_out[:,7] = x[:,5]

	return x_out


def process_image(path_image, path_label_file, path_calib_file, label, flip, outfile):
	"""
	Processes one image from the dataset and writes it out to the outfile.

	Input:
		path_image:      Path to the image file
		path_label_file: Path to the label file with KITTI labels
		path_calib_file: Path to the calibration file for this image
		label:           Which class label should be extracted from the dataset (default None)
		flip:            True/False whether the images should also be flipped by this script
		outfile:         File handle of the open output BBTXT file
	"""

	if flip:
		# We have to flip the image and save it
		image = cv2.imread(path_image)
		image = cv2.flip(image, 1)

		image_width = image.shape[1]

		filename   = os.path.basename(path_image)
		directory  = os.path.dirname(path_image).rstrip('/') + '_flip'
		path_image = os.path.join(directory, filename)

		if not os.path.exists(directory): os.makedirs(directory)

		cv2.imwrite(path_image, image)


	with open(path_label_file, 'r') as infile_label, open(path_calib_file, 'r') as infile_calib:
		# Read camera calibration matrices
		for line in infile_calib:
			if line[:2] == 'P2':
				P = read_camera_matrix(line.rstrip('\n'))

		# Read the objects
		for line in infile_label:
			line = line.rstrip('\n')
			data = line.split(' ')
			
			# First element of the data is the label. We don't want to process 'Misc' and
			# 'DontCare' labels
			if data[0] == 'Misc' or data[0] == 'DontCare': continue

			# Check label, if required
			if label is not None and MAPPING[LABELS[data[0]]] != label: continue


			# Extract image coordinates (positions) of 3D bounding box corners, the corners are
			# in the following order: fbr, rbr, fbl, rbl, ftr, rtr, ftl, rtl
			x = extract_3D_bb(data, P)

			if flip:
				x = flip_3D_bb(x, image_width)

			min_uv = np.min(x, axis=1)  # xmin, ymin
			max_uv = np.max(x, axis=1)  # xmax, ymax

			# The size of an image in KITTI is 1250x375. If the bounding box is significantly
			# larger, discard it - probably just some large distortion from camera
			if max_uv[1,0]-min_uv[1,0] > 700 or max_uv[0,0]-min_uv[0,0] > 1500:
				continue


			line_out = path_image + ' '
			line_out += str(LABELS[data[0]]) + ' '
			# For confidence we put one - just to have something
			line_out += '1 '
			# 3D bounding box is specified by the image coordinates of the front bottom left and
			# right corners, rear bottom left corner and y coordinate of the front top left
			# corner
			line_out += str(min_uv[0,0]) + ' ' + str(min_uv[1,0]) + ' ' \
					  + str(max_uv[0,0]) + ' ' + str(max_uv[1,0]) + ' ' \
					  + str(x[0,2]) + ' ' + str(x[1,2]) + ' ' + str(x[0,0]) + ' ' \
			 	      + str(x[1,0]) + ' ' + str(x[0,3]) + ' ' + str(x[1,3]) + ' ' \
			 	      + str(x[1,6]) + '\n'

			outfile.write(line_out)



def translate_file(path_labels, path_images, outfile, label, flip):
	"""
	Runs the translation of the KITTI 3d bounding box label format into the BB3TXT format.

	Input:
		path_labels: Path to the "label_2" folder of the KITTI dataset
		path_images: Path to the "image_2" folder with images from the KITTI dataset
		outfile:     File handle of the open output BBTXT file
		label:       Which class label should be extracted from the dataset (default None)
		flip:        True/False whether the images should also be flipped by this script
	"""
	print('-- TRANSLATING KITTI TO BB3TXT')

	# Get the list of all label files in the directory
	filenames = [f for f in os.listdir(path_labels) if os.path.isfile(os.path.join(path_labels, f))]

	if len(filenames) != 7481:
		print('Wrong number (%d) of files in the KITTI dataset! Should be 7481.'%(len(filenames)))
		return

	# Read each file and write the labels from it
	for f in filenames:
		path_label_file = os.path.join(path_labels, f)
		path_calib_file = os.path.join(path_labels.rstrip('/').rstrip('label_2'), 'calib', f)

		if not os.path.exists(path_calib_file):
			print('ERROR: We need camera calibration matrices "%s"'%(path_calib_file))
			exit(1)

		path_image = os.path.join(path_images, os.path.splitext(f)[0]) + '.png'
		if not os.path.isfile(path_image):
			print('WARNING: Image "%s" does not exist!'%(path_image))

		process_image(path_image, path_label_file, path_calib_file, label, False, outfile)
		if flip:
			# Add also the flipped image
			process_image(path_image, path_label_file, path_calib_file, label, True, outfile)


	print('-- TRANSLATION DONE')


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
	parser.add_argument('outfile', metavar='path_outfile', type=argparse.FileType('w'),
						help='Path to the output BBTXT file (including the extension)')
	parser.add_argument('--label', metavar='label', type=str, default=None,
						help='Single class of objects that should be separated from the dataset. ' \
							 'One from ' + str(available_categories(MAPPING)))
	parser.add_argument('--flip', dest='flip', action='store_true', default=False,
		                help='If provided, the images will also be flipped')

	args = parser.parse_args()

	if not os.path.exists(args.path_labels):
		print('Input path "%s" does not exist!'%(args.path_labels))
		parser.print_help()
		exit(1)
	if not os.path.exists(args.path_images):
		print('Input path "%s" does not exist!'%(args.path_images))
		parser.print_help()
		exit(1)
	if args.label is not None and args.label not in available_categories(MAPPING):
		print('Unknown class label "%s"!'%(args.label))
		exit(1)

	return args


def main():
	args = parse_arguments()
	translate_file(args.path_labels, args.path_images, args.outfile, args.label, args.flip)

	args.outfile.close()


if __name__ == '__main__':
    main()


