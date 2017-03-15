"""
Script for translating the KITTI annotation format into the BBTXT data format. We use
the 3D annotations and create bounding boxes out of them ourselves because the provided ones are
cropped to the image boundaries.

A BBTXT file is formatted like this:
filename label confidence xmin ymin xmax ymax
filename label confidence xmin ymin xmax ymax
filename label confidence xmin ymin xmax ymax
...

The script can generate 4 different output files, which correspond to the difficulty given by
the KITTI difficulty definition (I ignore the truncation value):
	easy:     Min. bounding box height: 40 Px, Max. occlusion level: Fully visible
	moderate: Min. bounding box height: 25 Px, Max. occlusion level: Partly occluded
	hard:     Min. bounding box height: 25 Px, Max. occlusion level: Difficult to see
	all:      Does no filtering, outputs everything

----------------------------------------------------------------------------------------------------
python kitti2bbtxt.py path_labels path_images [easy,moderate,hard,all] outfile.bbtxt
----------------------------------------------------------------------------------------------------
"""

__date__   = '12/01/2016'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'


import argparse
import os
import numpy as np

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

def check_label_difficulty(occlusion, ymin, ymax, difficulty):
	"""
	Check if the given label falls into the difficulty category.

	Input:
		data: One split line of the label file (line.split(' '))
		difficulty: One of [easy, moderate, hard, all]
	Returns:
		True if the label falls within the given difficulty group, False otherwise
	"""
	if difficulty == 'easy'     and occlusion > 0: return False
	if difficulty == 'moderate' and occlusion > 1: return False
	if difficulty == 'hard'     and occlusion > 2: return False

	# Now check the bounding box height
	height = float(ymax) - float(ymin)
	if difficulty == 'easy'     and height < 40: return False
	if difficulty == 'moderate' and height < 25: return False
	if difficulty == 'hard'     and height < 25: return False

	return True


def compute_hw_ratio(xmin, ymin, xmax, ymax):
	"""
	Compute the ratio of the bounding box dimensions h/w.
	"""
	w = float(xmax) - float(xmin)
	h = float(ymax) - float(ymin)

	return h/w


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


def extract_2D_bb(data, P):
	"""
	Extract xmin, ymin, xmax and ymax from the 3D annotation.

	We do this because the provided 2D bounding boxes are cropped (aligned) within the limits of
	the image, but that is not what we want for our purposes! We need the whole bounding box as it
	is in order to compute the centroid of the object

	Input:
		data: One split line of the label file (line.split(' '))
		P:    3x4 camera projection matrix
	Returns:
		xmin, ymin, xmax, ymax coordinates
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
	X = np.asmatrix([[l/2, -l/2,  l/2, -l/2, l/2,  -l/2, l/2, -l/2],
		             [0,    0,    0,   0,    -h,   -h,   -h,  -h],
		             [-w/2, -w/2, w/2, w/2,  -w/2, -w/2, w/2, w/2],
		             [1,    1,    1,   1,    1,    1,    1,   1]])
	# Rotate the 3D box around y axis and translate it to the correct position in the camera frame
	X = Rt4x4(R3x3_y(ry), t3x1(cx, cy, cz)) * X
	x = P * X
	# x is in homogenous coordinates -> get u, v
	x = x / x[2,:]

	# Coordinates of the 2D are the extremes
	min_uv = np.min(x, axis=1)  # xmin, ymin
	max_uv = np.max(x, axis=1)  # xmax, ymax

	# image = cv2.imread(path_image)
	# # Front
	# cv2.line(image, (int(x[0,0]), int(x[1,0])), (int(x[0,2]), int(x[1,2])), (0,255,0))
	# cv2.line(image, (int(x[0,4]), int(x[1,4])), (int(x[0,6]), int(x[1,6])), (0,255,0))
	# cv2.line(image, (int(x[0,0]), int(x[1,0])), (int(x[0,4]), int(x[1,4])), (0,255,0))
	# cv2.line(image, (int(x[0,2]), int(x[1,2])), (int(x[0,6]), int(x[1,6])), (0,255,0))
	# # Rear
	# cv2.line(image, (int(x[0,1]), int(x[1,1])), (int(x[0,3]), int(x[1,3])), (0,0,255))
	# cv2.line(image, (int(x[0,5]), int(x[1,5])), (int(x[0,7]), int(x[1,7])), (0,0,255))
	# cv2.line(image, (int(x[0,1]), int(x[1,1])), (int(x[0,5]), int(x[1,5])), (0,0,255))
	# cv2.line(image, (int(x[0,3]), int(x[1,3])), (int(x[0,7]), int(x[1,7])), (0,0,255))
	# # Connections
	# cv2.line(image, (int(x[0,0]), int(x[1,0])), (int(x[0,1]), int(x[1,1])), (255,0,0))
	# cv2.line(image, (int(x[0,2]), int(x[1,2])), (int(x[0,3]), int(x[1,3])), (255,0,0))
	# cv2.line(image, (int(x[0,4]), int(x[1,4])), (int(x[0,5]), int(x[1,5])), (255,0,0))
	# cv2.line(image, (int(x[0,6]), int(x[1,6])), (int(x[0,7]), int(x[1,7])), (255,0,0))
	# # 2D bounding box
	# cv2.rectangle(image, (int(min_uv[0,0]), int(min_uv[1,0])), (int(max_uv[0,0]), int(max_uv[1,0])), (0,0,255), 3)
	# cv2.circle(image, (int((min_uv[0,0]+max_uv[0,0])/2), int((min_uv[1,0]+max_uv[1,0])/2)), 10, (0,0,255), -1)
	# # Show image
	# cv2.imshow('img', image)
	# cv2.waitKey()

	return min_uv[0,0], min_uv[1,0], max_uv[0,0], max_uv[1,0]


def translate_file(path_labels, path_images, difficulty, outfile, filter, label):
	"""
	Runs the translation of the KITTI label format into the BBTXT format.

	Input:
		path_labels: Path to the "label_2" folder of the KITTI dataset
		path_images: Path to the "image_2" folder with images from the KITTI dataset
		difficulty:  Difficulty of the dataset by KITTI standards (ignores truncation percentage!)
		outfile:     File handle of the open output BBTXT file
		filter:      Ratio of h/w above which will bounding boxes be filtered
		label:       Which class label should be extracted from the dataset (default None)
	"""
	print('-- TRANSLATING KITTI TO BBTXT')

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


				# Extract xmin, ymin, xmax and ymax from the 3D annotation
				xmin, ymin, xmax, ymax = extract_2D_bb(data, P)


				# Check the difficulty of this bounding box
				if not check_label_difficulty(int(data[2]), ymin, ymax, difficulty):
					continue

				# Check the bounding box h/w ratio - this is because there are some incorrect labels
				# in the dataset, which just mark one or two columns or so
				if compute_hw_ratio(xmin, ymin, xmax, ymax) > filter:
					continue

				line_out = path_image + ' '
				line_out += str(LABELS[data[0]]) + ' '
				# For confidence we put one - just to have something
				line_out += '1 '
				# Bounding box
				line_out += str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + '\n'

				outfile.write(line_out)

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
	parser.add_argument('difficulty', metavar='difficulty', type=str,
						help='Difficulty by KITTI standards, one of [easy, moderate, hard, all]')
	parser.add_argument('outfile', metavar='path_outfile', type=argparse.FileType('w'),
						help='Path to the output BBTXT file (including the extension)')
	parser.add_argument('--filter', metavar='filter', type=float, default=99999.9,
						help='Ratio h/w. If provided, the bounding boxes with their h/w above the '\
							 'provided ratio will be filtered out. This is to filter incorrectly ' \
							 'labeled objects')
	parser.add_argument('--label', metavar='label', type=str, default=None,
						help='Single class of objects that should be separated from the dataset. ' \
							 'One from ' + str(available_categories(MAPPING)))

	args = parser.parse_args()

	if not os.path.exists(args.path_labels):
		print('Input path "%s" does not exist!'%(args.path_labels))
		parser.print_help()
		exit(1)
	if not os.path.exists(args.path_images):
		print('Input path "%s" does not exist!'%(args.path_images))
		parser.print_help()
		exit(1)
	if args.difficulty not in ['easy', 'moderate', 'hard', 'all']:
		print('Unknown difficulty "%s"!'%(args.difficulty))
		parser.print_help()
		exit(1)
	if args.label is not None and args.label not in available_categories(MAPPING):
		print('Unknown class label "%s"!'%(args.label))
		exit(1)

	return args


def main():
	args = parse_arguments()
	translate_file(args.path_labels, args.path_images, args.difficulty, args.outfile, args.filter,
				   args.label)
	args.outfile.close()


if __name__ == '__main__':
    main()


