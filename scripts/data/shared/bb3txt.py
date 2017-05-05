"""
Functions for loading the BB3TXT files.

A BB3TXT file is formatted like this:
filename label confidence xmin ymin xmax ymax fblx fbly fbrx fbry rblx rbly ftly
filename label confidence xmin ymin xmax ymax fblx fbly fbrx fbry rblx rbly ftly
filename label confidence xmin ymin xmax ymax fblx fbly fbrx fbry rblx rbly ftly
...

All numbers are in image coordintes (u,v - i.e. pixels) and this is what they represent:
	xmin, ymin, xmax, ymax - 2D bounding box
	fblx, fbly - x and y coordinates of the front bottom left corner
	fbrx, fbry - x and y coordinates of the front bottom right corner
	rblx, rbly - x and y coordinates of the rear bottom left corner
	ftly       - y coordinate of the front top left corner

The left and right are taken from the point of view of the object! This means that if we see an
object from the front, the left and right corners will be switched in the image.

"""

__date__   = '03/18/2017'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'

from classes import BB3D


####################################################################################################
#                                            FUNCTIONS                                             # 
####################################################################################################

def load_bb3txt(path_bb3txt):
	"""
	Loads a BB3TXT file into a dictionary indexed by file names.

	Input:
		path_bb3txt: Path to a BB3TXT file
	Returns:
		dictionary of lists of BB3D objects
	"""
	with open(path_bb3txt, 'r') as infile:
		# Ok, the file is open so we can start reading
		image_dict = {}

		for line in infile:
			line = line.rstrip('\n')
			data = line.split(' ')

			filename = data[0]
			if filename not in image_dict:
				# This image is not in the list yet -> initialize it
				image_dict[filename] = []

			image_dict[filename].append(BB3D(xmin=float(data[3]), ymin=float(data[4]),
											 xmax=float(data[5]), ymax=float(data[6]),
											 fblx=float(data[7]), fbly=float(data[8]),
											 fbrx=float(data[9]), fbry=float(data[10]),
											 rblx=float(data[11]), rbly=float(data[12]),
											 ftly=float(data[13]),
											 label=int(data[1]), confidence=float(data[2])))

		return image_dict

	print('ERROR: File "%s" could not be opened!'%(path_bb3txt))
	exit(1)


def load_bb3txt_to_list(path_bb3txt):
	"""
	Loads a BBTXT file into a list of BB2D objects. The information about filename will get
	lost - this is a function purely for statistical purposes.

	Input:
		path_bb3txt: Path to a BBTXT file
	Returns:
		list of BB3D objects
	"""
	with open(path_bb3txt, 'r') as infile:
		# Ok, the file is open so we can start reading
		bb3d_list = []

		for line in infile:
			line = line.rstrip('\n')
			data = line.split(' ')

			bb3d_list.append(BB3D(xmin=float(data[3]), ymin=float(data[4]),
								  xmax=float(data[5]), ymax=float(data[6]),
								  fblx=float(data[7]), fbly=float(data[8]),
								  fbrx=float(data[9]), fbry=float(data[10]),
								  rblx=float(data[11]), rbly=float(data[12]),
								  ftly=float(data[13]),
								  label=int(data[1]), confidence=float(data[2])))

		return bb3d_list

	print('ERROR: File "%s" could not be opened!'%(path_bb3txt))
	exit(1)


def write_bb3txt(bb3d_dict, path_bb3txt):
	"""
	Writes a dictionary of BB3D objects indexed by filenames to a BB3TXT file.

	Input:
		bb3d_dict:  Dictionary of lists of BB3D objects indexed by filenames
		path_bb3txt: Path to a BB3TXT file
	Returns:
		
	"""
	with open(path_bb3txt, 'w') as outfile:
		# Ok, the file is open so lets start writing

		for filename in bb3d_dict:
			for i in range(len(bb3d_dict[filename])):
				bb3d = bb3d_dict[filename][i]

				# filename label confidence xmin ymin xmax ymax fblx fbly fbrx fbry rblx rbly ftly
				outfile.write(str(filename) + ' ')
				outfile.write('%d %f %f %f %f %f %f %f %f %f %f %f %f\n'%(
					bb3d.label, bb3d.confidence, bb3d.bb2d.xmin, bb3d.bb2d.ymin, bb3d.bb2d.xmax, 
					bb3d.bb2d.ymax, bb3d.fblx, bb3d.fbly, bb3d.fbrx, bb3d.fbry, bb3d.rblx, 
					bb3d.rbly, bb3d.ftly))

