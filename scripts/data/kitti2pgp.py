"""
Script for translating the KITTI camera calibration files into our PGP format.

The PGP file contatins one line for each image with the 3x4 image projection matrix P and the ground
plane equation ax+by+cz+d=0 coefficients:
filename1 p00 p01 p02 p03 p10 p11 p12 p13 p20 p21 p22 p23 a b c d
filename2 p00 p01 p02 p03 p10 p11 p12 p13 p20 p21 p22 p23 a b c d
filename3 p00 p01 p02 p03 p10 p11 p12 p13 p20 p21 p22 p23 a b c d
...

----------------------------------------------------------------------------------------------------
python kitti2pgp.py path_calib outfile.pgp
----------------------------------------------------------------------------------------------------
"""

__date__   = '04/18/2017'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'


import argparse
import os
import numpy as np


####################################################################################################
#                                            SETTINGS                                              # 
####################################################################################################

# Coefficients of the equation of the ground plane ax+by+cz+d=0
# GP_1x4 = np.asmatrix([[-0.00272088,  0.83395045, -0.00562125, -1.49405822]])
# GP_1x4 = np.asmatrix([[ 0.        ,  0.20962029,  0.        , -0.31233423]])
GP_1x4 = np.asmatrix([[0,  1, 0, -2.1]])



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


def process_calib_file(path_calib, filename, outfile):
	"""
	Processes one calib file from the KITTI dataset.

	Input:
		path_calib: Path to the folder with KITTI calibration files
		filename:   Filename of the currently processed calibration file
		outfile:    File handle of the open output PGP file
	"""
	with open(os.path.join(path_calib, filename), 'r') as infile_calib:
		# Read camera calibration matrices
		P = None
		for line in infile_calib:
			if line[:2] == 'P2':
				P = read_camera_matrix(line.rstrip('\n'))

		if P is None:
			print('ERROR: Missing image projection matrix P2 in file "' + filename + '"')
			exit(1)

		# Path to the image for which is this calibration matrix
		path_image = os.path.join(path_calib.rstrip('/').rstrip('calib'), 'image_2', 
								  filename.rstrip('.txt') + '.png')

		if not os.path.exists(path_image):
			print('ERROR: Image for which we are extracting calibration does not exist "' \
				  + path_image + '"')
		
		# OK, we have the camera calibration matrix. Ground plane is the same for all images
		# Write it to the PGP file
		outfile.write(path_image + ' %f %f %f %f %f %f %f %f %f %f %f %f'%(P[0,0], P[0,1], P[0,2], 
			          P[0,3], P[1,0], P[1,1], P[1,2], P[1,3], P[2,0], P[2,1], P[2,2], P[2,3]))
		outfile.write(' %f %f %f %f\n'%(GP_1x4[0,0], GP_1x4[0,1], GP_1x4[0,2], GP_1x4[0,3]))



def translate_file(path_calib, outfile):
	"""
	Runs the translation of the KITTI calibration file format into the PGP format.

	Input:
		path_calib:  Path to the "calib" folder of the KITTI dataset
		outfile:     File handle of the open output PGP file
	"""
	print('-- TRANSLATING KITTI CALIBRATION TO PGP')

	# Get the list of all label files in the directory
	filenames = [f for f in os.listdir(path_calib) if os.path.isfile(os.path.join(path_calib, f))]

	if len(filenames) != 7481:
		print('Wrong number (%d) of files in the KITTI dataset! Should be 7481.'%(len(filenames)))
		return

	# Read each file and its calibration matrix
	for f in filenames:
		process_calib_file(path_calib, f, outfile)

	print('-- TRANSLATION DONE')


####################################################################################################
#                                               MAIN                                               # 
####################################################################################################

def parse_arguments():
	"""
	Parse input options of the script.
	"""
	parser = argparse.ArgumentParser(description='Convert KITTI calibration files into PGP.')
	parser.add_argument('path_calib', metavar='path_calib', type=str,
						help='Path to the "calib" folder of the KITTI dataset')
	parser.add_argument('outfile', metavar='path_outfile', type=argparse.FileType('w'),
						help='Path to the output PGP file (including the extension)')


	args = parser.parse_args()

	if not os.path.exists(args.path_calib):
		print('Input path "%s" does not exist!'%(args.path_calib))
		parser.print_help()
		exit(1)

	return args


def main():
	args = parse_arguments()
	translate_file(args.path_calib, args.outfile)

	args.outfile.close()


if __name__ == '__main__':
    main()


