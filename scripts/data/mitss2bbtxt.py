"""
Script for translating the MIT Street Scenes Dataset annotation to BBTXT format.

A BBTXT file is formatted like this:
filename label confidence xmin ymin xmax ymax
filename label confidence xmin ymin xmax ymax
filename label confidence xmin ymin xmax ymax
...
----------------------------------------------------------------------------------------------------
python mitss2bbtxt.py path_labels path_images outfile.bbtxt
----------------------------------------------------------------------------------------------------
"""

__date__   = '03/12/2017'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'


import argparse
import os
import xml.etree.ElementTree

from mappings.utils import LabelMappingManager
from mappings.utils import available_categories


####################################################################################################
#                                           DEFINITIONS                                            # 
####################################################################################################

# IMPORTANT !!
# The labels must translate precisely into the numbers in the mitss.yaml mapping file!
LABELS = {
    'building': 1,
    'bicycle': 2,
    'car': 3,
    'sky': 4,
    'tree': 5,
    'pedestrian': 6,
    'store': 7,
    'sidewalk': 8,
    'road': 9,
}

# Initialize the LabelMappingManager
LMM     = LabelMappingManager()
MAPPING = LMM.get_mapping('mitss')


####################################################################################################
#                                            FUNCTIONS                                             # 
####################################################################################################

def translate_file(path_file, path_images, outfile, label):
	"""
	Translates a single XML file with MIT Street Scenes labels and appends its output to the given
	BBTXT file.

	Input:
		path_file:   Path to the XML file to be translated
		path_images: Path to the "Original" folder, which contains the images
		outfile:     File handle of the open output BBTXT file
		label:       Which class label should be extracted from the dataset (default None = all)
	"""
	e = xml.etree.ElementTree.parse(path_file).getroot()
	path_image = os.path.join(path_images, e.find('filename').text.rstrip('\n').strip('\n'))

	if not os.path.isfile(path_image):
		print('WARNING: Image "%s" does not exist!'%(path_image))

	# Go through all objects and extract their bounding boxes
	for obj in e.findall('object'):
		# There are some empty objects
		if obj.find('name') is not None:

			original_label = obj.find('name').text.rstrip('\n').strip('\n')

			# Check label if required
			if label is not None and MAPPING[LABELS[original_label]] != label: continue

			# This dataset is annotated by polygon segmentations
			xmin = 999999
			ymin = 999999
			xmax = 0
			ymax = 0
			for pt in obj.find('polygon').findall('pt'):
				x = float(pt.find('x').text.rstrip('\n').strip('\n'))
				y = float(pt.find('y').text.rstrip('\n').strip('\n'))

				if x > xmax: xmax = x
				if x < xmin: xmin = x
				if y > ymax: ymax = y
				if y < ymin: ymin = y

			line_out  = path_image + ' ' + str(LABELS[original_label]) + ' 1 '
			line_out += str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + '\n'

			outfile.write(line_out)


def translate_files(path_labels, path_images, outfile, label):
	"""
	Loads all XML annotation files and translates them to BBTXT.

	Input:
		path_labels: Path to the folder with XML label files to be translated
		path_images: Path to the "Original" folder, which contains the images
		outfile:     File handle of the open output BBTXT file
		label:       Which class label should be extracted from the dataset (default None = all)
	"""
	print('-- TRANSLATING MIT STREET SCENES ANNOTATION TO BBTXT')

	# Get the content of the path_labels directory
	xml_names = [f for f in os.listdir(path_labels) if os.path.splitext(f)[1] == '.xml']

	for filename in xml_names:
		# print('-- Processing: ' + filename)
		translate_file(os.path.join(path_labels, filename), path_images, outfile, label)

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
						help='Path to the folder with XML label files (Anno_XML) to be translated '\
						'(all .xml files from this folder will be loaded)')
	parser.add_argument('path_images', metavar='path_images', type=str,
						help='Path to the folder, which contains the images (Original)')
	parser.add_argument('outfile', metavar='path_outfile', type=argparse.FileType('w'),
						help='Path to the output BBTXT file (including the extension)')
	parser.add_argument('--label', metavar='label', type=str, default=None,
						help='Single class of objects that should be separated from the dataset. ' \
							 'One from ' + str(available_categories(MAPPING)))

	args = parser.parse_args()

	if not check_path(args.path_labels, True) or not check_path(args.path_images, True):
		parser.print_help()
		exit(1)
	if args.label is not None and args.label not in available_categories(MAPPING):
		print('Unknown class label "%s"!'%(args.label))
		exit(1)

	return args


def main():
	args = parse_arguments()
	translate_files(args.path_labels, args.path_images, args.outfile, args.label)
	args.outfile.close()


if __name__ == '__main__':
    main()


