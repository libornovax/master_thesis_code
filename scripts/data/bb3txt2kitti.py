"""
Script for translating the BB3TXT data format to the original KITTI bounding box annotation with 
alpha angle.

A BB3TXT file is formatted like this:
filename label confidence xmin ymin xmax ymax fblx fbly fbrx fbry rblx rbly ftly
filename label confidence xmin ymin xmax ymax fblx fbly fbrx fbry rblx rbly ftly
filename label confidence xmin ymin xmax ymax fblx fbly fbrx fbry rblx rbly ftly
...

The KITTI labels have a single label file for each image and each line contains:
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.


----------------------------------------------------------------------------------------------------
python bb3txt2kitti.py infile.bb3txt infile.pgp path_out
----------------------------------------------------------------------------------------------------
"""

__date__   = '05/18/2017'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'


import argparse
import os
import numpy as np
# from matplotlib import pyplot as plt

from mappings.utils import LabelMappingManager
from shared.bb3txt import load_bb3txt
from shared.pgp import load_pgp
from shared.geometry import R3x3_y, t3x1, Rt4x4


####################################################################################################
#                                           DEFINITIONS                                            # 
####################################################################################################

# IMPORTANT !!
# The labels must translate precisely into the numbers in the kitti.yaml mapping file!
LABELS = {
	1: 'Car',
	2: 'Van',
	3: 'Truck',
	4: 'Pedestrian',
	5: 'Person_sitting',
	6: 'Cyclist',
	7: 'Tram',
	# Throw away 'Misc' and 'DontCare'
}

# Initialize the LabelMappingManager
LMM     = LabelMappingManager()
MAPPING = LMM.get_mapping('kitti')


####################################################################################################
#                                            FUNCTIONS                                             # 
####################################################################################################

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


def viewing_angle(bb3d, pgp):
	"""
	Computes the viewing angle alpha of the object.

	Input:
		bb3d:    A BB3D object
		pgp:     The corresponding PGP object
	Returns:
		alpha
	"""
	# 3D coordinates of FBL FBR RBR RBL FTL FTR RTR RTL
	X_3x8 = pgp.reconstruct_bb3d(bb3d)

	# plt.plot([X_3x8[0,0], X_3x8[0,1]], [X_3x8[2,0], X_3x8[2,1]], color='#00FF00', linewidth=1)
	# plt.plot([X_3x8[0,0], X_3x8[0,3]], [X_3x8[2,0], X_3x8[2,3]], color='#3399FF', linewidth=1)
	# plt.plot([X_3x8[0,1], X_3x8[0,2]], [X_3x8[2,1], X_3x8[2,2]], color='#3399FF', linewidth=1)
	# plt.plot([X_3x8[0,2], X_3x8[0,3]], [X_3x8[2,2], X_3x8[2,3]], color='#3399FF', linewidth=1)

	# Object's center of mass
	CM_3x1 = (X_3x8[:,0] + X_3x8[:,6]) / 2.0
	# Vector from camera center to the object's center of mass
	CMC_3x1 = CM_3x1 - pgp.C_3x1
	# Extract only the top view (x/z plane)
	CMC_2x1 = CMC_3x1[[0,2],0]

	# plt.plot([0,CMC_3x1[0,0]], [0,CMC_3x1[2,0]], color='y', linewidth=2)

	# Direction, which is facing the bounding box
	d_3x1 = X_3x8[:,0] - X_3x8[:,3]
	d_2x1 = d_3x1[[0,2],0]

	# plt.plot([0,d_3x1[0,0]], [0,d_3x1[2,0]], color='b', linewidth=2)

	angle_CMC = np.arctan2(CMC_2x1[1,0], CMC_2x1[0,0])
	angle_d   = np.arctan2(d_2x1[1,0], d_2x1[0,0])

	alpha = angle_CMC - angle_d - np.pi / 2.0

	# Wrap to [0, 2pi]
	alpha = np.mod(alpha, 2*np.pi)
	# Wrap to [-pi, pi]
	if alpha > np.pi: alpha -= 2*np.pi
	if alpha < -np.pi: alpha += 2*np.pi

	# plt.text(CMC_3x1[0,0], CMC_3x1[2,0]-5, str(alpha), fontsize=15, color='#000000')

	return alpha


def write_bb3d(bb3d, pgp, outfile):
	"""
	Writes the given 3D bounding box in the KITTI format to one line of the output file.

	Input:
		bb3d:    A BB3D object
		pgp:     The corresponding PGP object
		outfile: Handle to the open output file
	"""
	# Get the viewing angle
	alpha = viewing_angle(bb3d, pgp)

	outfile.write('%s -1 -1 %.2f %.2f %.2f %.2f %.2f -1 -1 -1 -1000 -1000 -1000 -10 %.2f\n' % 
		(LABELS[bb3d.label], alpha, bb3d.bb2d.xmin, bb3d.bb2d.ymin, bb3d.bb2d.xmax, bb3d.bb2d.ymax, 
		bb3d.confidence))


def translate_file(path_bb3txt, path_pgp, path_out):
	"""
	Runs the translation of the KITTI 3d bounding box label format into the BB3TXT format.

	Input:
		path_bb3txt: Path to the input BB3TXT file to be converted
		path_pgp:    Path to the corresponding PGP file with P matrices and ground planes
		path_out:    Path to the output folder
	"""
	print('-- TRANSLATING BB3TXT TO KITTI')

	# Prepare the output directory
	if not os.path.exists(path_out):
		os.makedirs(path_out)

	# Load the input files
	print('-- Reading BB3TXT file "' + path_bb3txt + '"')
	bb3txt = load_bb3txt(path_bb3txt)
	print('-- Reading PGP file "' + path_pgp + '"')
	pgps   = load_pgp(path_pgp)


	print('-- Translating...')
	i = 0
	for filename in bb3txt:
		image_id = os.path.splitext(os.path.basename(filename))[0]
		# Path to the new KITTI label file
		path_label = os.path.join(path_out, str(image_id) + '.txt')

		if not filename in pgps.keys():
			print('ERROR: Missing PGP entry for file "' + filename + '"!')
			exit(2)

		pgp = pgps[filename]

		# plt.cla()

		# plt.plot([0, pgp.C_3x1[0,0]], [0, pgp.C_3x1[2,0]], color='g', linewidth=2)

		with open(path_label, 'w') as outfile:
			for bb in bb3txt[filename]:
				# Write out the 3D bounding box
				write_bb3d(bb, pgp, outfile)

		# plt.axis('equal')
		# plt.show()

		print(i)
		i += 1


	print('-- TRANSLATION DONE')


####################################################################################################
#                                               MAIN                                               # 
####################################################################################################

def check_path(path, is_folder=False):
	"""
	Checks if the given path exists.

	Input:
		path:      Path to be checked
		is_folder: True if the checked path is a folder
	Returns:
		True if the given path exists
	"""
	if not os.path.exists(path) or (not is_folder and not os.path.isfile(path)):
		print('ERROR: Path "%s" does not exist!'%(path))
		return False

	return True


def parse_arguments():
	"""
	Parse input options of the script.
	"""
	parser = argparse.ArgumentParser(description='Convert KITTI label files into BBTXT.')
	parser.add_argument('path_bb3txt', metavar='path_bb3txt', type=str,
						help='Path to the BB3TXT file to be translated')
	parser.add_argument('path_pgp', metavar='path_pgp', type=str,
						help='Path to the PGP file with P matrices and ground plane equations')
	parser.add_argument('path_out', metavar='path_out', type=str,
						help='Path to the output folder, where the KITTI annotations will be stored')

	args = parser.parse_args()

	if not check_path(args.path_bb3txt) or not check_path(args.path_pgp):
		parser.print_help()
		exit(1)

	if os.path.exists(args.path_out):
		check = raw_input('The output folder "' + args.path_out + '" already exists!\n Do you want to rewrite it? [y/n]\n')
		if check != 'y':
			print('Conversion aborted.')
			exit(1)

	return args


def main():
	args = parse_arguments()

	translate_file(args.path_bb3txt, args.path_pgp, args.path_out)


if __name__ == '__main__':
    main()


