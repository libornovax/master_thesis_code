"""
Script for translating the dataset from Jiri Trefny (jura) to BBTXT format.

Since this dataset contains only cars all objects will be assigned label 1.

A BBTXT file is formatted like this:
filename label confidence xmin ymin xmax ymax
filename label confidence xmin ymin xmax ymax
filename label confidence xmin ymin xmax ymax
...

The label txt files in the jura dataset have the following form:
filename:car_ID left top right bottom bumper_center;car_ID left top right bottom bumper_center....
filename:car_ID left top right bottom bumper_center;car_ID left top right bottom bumper_center....
...

----------------------------------------------------------------------------------------------------
python jura2bbtxt.py path_labels path_images outfile.bbtxt
----------------------------------------------------------------------------------------------------
"""

__date__   = '12/06/2016'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'


import argparse
import os


####################################################################################################
#                                           DEFINITIONS                                            # 
####################################################################################################



####################################################################################################
#                                            FUNCTIONS                                             # 
####################################################################################################

def translate_file(path_file, path_images, outfile):
	"""
	Translates a single TXT file with jura labels and appends its output to the given BBTXT file.

	Input:
		path_file:   Path to the TXT file to be translated
		path_images: Path to the folder, which contains "Toyota" and "LiborNovak" folders
		outfile:     File handle of the open output BBTXT file
	"""
	with open(path_file, 'r') as infile:
		for line in infile:
			line = line.rstrip('\n')

			# Get the image file path
			data = line.split(':')
			path_image = os.path.join(path_images, data[0])

			if not os.path.isfile(path_image):
				print('WARNING: Image "%s" does not exist!'%(path_image))

			# Get the annotations
			annotations = data[1].split(';')
			for annotation in annotations:
				if annotation != '':
					# Get the numbers
					coords = annotation.split(' ')

					# All annotations are cars -> put 1 for class. For confidence we put 1 - just
					# to have something
					line_out = path_image + ' 1 1 '
					# Bounding box
					line_out += coords[1] + ' ' + coords[2] + ' ' + coords[3] + ' ' + coords[4] + '\n'

					outfile.write(line_out)


def translate_files(path_labels, path_images, outfile):
	"""
	Runs the translation of Jiri Trefny's label format into the BBTXT format. Translates all TXT
	files in the path_labels folder into a single BBTXT file.

	Input:
		path_labels: Path to the folder with label files to be translated
		path_images: Path to the folder, which contains "Toyota" and "LiborNovak" folders
		outfile:     File handle of the open output BBTXT file
	"""
	print('-- TRANSLATING JIRI TREFNY\'S ANNOTATION TO BBTXT')

	# Get the content of the path_labels directory
	txt_names = [f for f in os.listdir(path_labels) if os.path.splitext(f)[1] == '.txt']

	for filename in txt_names:
		print('-- Processing: ' + filename)
		translate_file(os.path.join(path_labels, filename), path_images, outfile)

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
	parser.add_argument('path_labels', metavar='path_labels', type=str,
						help='Path to the folder with label files to be translated (all .txt files'\
						' from this folder will be loaded)')
	parser.add_argument('path_images', metavar='path_images', type=str,
						help='Path to the folder, which contains "Toyota" and "LiborNovak" folders')
	parser.add_argument('outfile', metavar='path_outfile', type=argparse.FileType('w'),
						help='Path to the output BBTXT file (including the extension)')

	args = parser.parse_args()

	if not check_path(args.path_labels, True) or not check_path(args.path_images, True):
		parser.print_help()
		exit(1)

	return args


def main():
	args = parse_arguments()
	translate_files(args.path_labels, args.path_images, args.outfile)
	args.outfile.close()


if __name__ == '__main__':
    main()


