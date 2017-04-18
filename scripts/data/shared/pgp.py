"""
Functions for loading the PGP files - files containing the P matrix and ground plane equation.

The PGP file contatins one line for each image with the 3x4 image projection matrix P and the ground
plane equation ax+by+cz+d=0 coefficients:
filename1 p00 p01 p02 p03 p10 p11 p12 p13 p20 p21 p22 p23 a b c d
filename2 p00 p01 p02 p03 p10 p11 p12 p13 p20 p21 p22 p23 a b c d
filename3 p00 p01 p02 p03 p10 p11 p12 p13 p20 p21 p22 p23 a b c d
...

"""

__date__   = '18/02/2016'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'

from classes import PGP


####################################################################################################
#                                            FUNCTIONS                                             # 
####################################################################################################

def load_pgp(path_pgp):
	"""
	Loads a PGP file into a dictionary indexed by file names.

	Input:
		path_pgp: Path to a PGP file
	Returns:
		dictionary with PGP objects
	"""
	with open(path_pgp, 'r') as infile:
		# Ok, the file is open so we can start reading
		image_dict = {}

		for line in infile:
			line = line.rstrip('\n')
			data = line.split(' ')

			filename = data[0]
			if filename in image_dict:
				print('ERROR: Duplicate entry in PGP for image "' + filename + '"')
				exit(1)

			image_dict[filename] = PGP(float(data[1]), float(data[2]), float(data[3]), 
				float(data[4]), float(data[5]), float(data[6]), float(data[7]), float(data[8]), 
				float(data[9]), float(data[10]), float(data[11]), float(data[12]), float(data[13]), 
				float(data[14]), float(data[15]), float(data[16]))

		return image_dict

	print('ERROR: File "%s" could not be opened!'%(path_pgp))
	exit(1)
