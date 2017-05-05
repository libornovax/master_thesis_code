"""
Functions for loading the BBTXT files.

A BBTXT file is formatted like this:
filename label confidence xmin ymin xmax ymax
filename label confidence xmin ymin xmax ymax
filename label confidence xmin ymin xmax ymax
...

IMPORTANT! Negative label means occlusion!
"""

__date__   = '12/02/2016'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'

from classes import BB2D


####################################################################################################
#                                            FUNCTIONS                                             # 
####################################################################################################

def load_bbtxt(path_bbtxt):
	"""
	Loads a BBTXT file into a dictionary indexed by file names.

	Input:
		path_bbtxt: Path to a BBTXT file
	Returns:
		dictionary of lists of BB2d objects
	"""
	with open(path_bbtxt, 'r') as infile:
		# Ok, the file is open so we can start reading
		image_dict = {}

		for line in infile:
			line = line.rstrip('\n')
			data = line.split(' ')

			filename = data[0]
			if filename not in image_dict:
				# This image is not in the list yet -> initialize it
				image_dict[filename] = []

			image_dict[filename].append(BB2D(xmin=float(data[3]), ymin=float(data[4]),
											 xmax=float(data[5]), ymax=float(data[6]),
											 label=abs(int(data[1])), confidence=float(data[2]),
											 required=(int(data[1]) >= 0)))

		return image_dict

	print('ERROR: File "%s" could not be opened!'%(path_bbtxt))
	exit(1)


def load_bbtxt_to_list(path_bbtxt):
	"""
	Loads a BBTXT file into a list of BB2D objects. The information about filename will get
	lost - this is a function purely for statistical purposes.

	Input:
		path_bbtxt: Path to a BBTXT file
	Returns:
		list of BB2d objects
	"""
	with open(path_bbtxt, 'r') as infile:
		# Ok, the file is open so we can start reading
		bb2d_list = []

		for line in infile:
			line = line.rstrip('\n')
			data = line.split(' ')

			bb2d_list.append(BB2D(xmin=float(data[3]), ymin=float(data[4]),
								  xmax=float(data[5]), ymax=float(data[6]),
								  label=abs(int(data[1])), confidence=float(data[2]),
								  required=(int(data[1]) >= 0)))

		return bb2d_list

	print('ERROR: File "%s" could not be opened!'%(path_bbtxt))
	exit(1)


def write_bbtxt(bb2d_dict, path_bbtxt):
	"""
	Writes a dictionary of BB2D objects indexed by filenames to a BBTXT file.

	Input:
		bb2d_dict:  Dictionary of lists of BB2d objects indexed by filenames
		path_bbtxt: Path to a BBTXT file
	Returns:
		
	"""
	with open(path_bbtxt, 'w') as outfile:
		# Ok, the file is open so lets start writing

		for filename in bb2d_dict:
			for i in range(len(bb2d_dict[filename])):
				bb2d = bb2d_dict[filename][i]

				# filename label confidence xmin ymin xmax ymax
				outfile.write(str(filename) + ' ')
				outfile.write('%d %f %f %f %f %f\n'%(bb2d.label, bb2d.confidence, bb2d.xmin, 
													 bb2d.ymin, bb2d.xmax, bb2d.ymax))

