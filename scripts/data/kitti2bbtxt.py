"""
Script for translating the KITTI annotation format into the BBTXT data format.

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


####################################################################################################
#                                            FUNCTIONS                                             # 
####################################################################################################

def check_label_difficulty(data, difficulty):
	"""
	Check if the given label falls into the difficulty category.

	Input:
		data: One split line of the label file (line.split(' '))
		difficulty: One of [easy, moderate, hard, all]
	Returns:
		True if the label falls within the given difficulty group, False otherwise
	"""
	occlusion = int(data[2])
	if difficulty == 'easy'     and occlusion > 0: return False
	if difficulty == 'moderate' and occlusion > 1: return False
	if difficulty == 'hard'     and occlusion > 2: return False

	# Now check the bounding box height
	height = float(data[7]) - float(data[5])
	if difficulty == 'easy'     and height < 40: return False
	if difficulty == 'moderate' and height < 25: return False
	if difficulty == 'hard'     and height < 25: return False

	return True


def translate_file(path_labels, path_images, difficulty, outfile):
	"""
	Runs the translation of the KITTI label format into the BBTXT format.

	Input:
		path_labels: Path to the "label_2" folder of the KITTI dataset
		path_images: Path to the "image_2" folder with images from the KITTI dataset
		difficulty:  Difficulty of the dataset by KITTI standards (ignores truncation percentage!)
		outfile:     File handle of the open output BBTXT file
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
		with open(path_label_file, 'r') as infile:
			for line in infile:
				line = line.rstrip('\n')
				data = line.split(' ')
				
				# First element of the data is the label. We don't want to process 'Misc' and
				# 'DontCare' labels
				if data[0] == 'Misc' or data[0] == 'DontCare': continue

				# Check the difficulty of this bounding box
				if not check_label_difficulty(data, difficulty): continue

				path_image = os.path.join(path_images, os.path.splitext(f)[0]) + '.png'
				if not os.path.isfile(path_image):
					print('WARNING: Image "%s" does not exist!'%(path_image))

				line_out = path_image + ' '
				line_out += str(LABELS[data[0]]) + ' '
				# For confidence we put one - just to have something
				line_out += '1 '
				# Bounding box
				line_out += data[4] + ' ' + data[5] + ' ' + data[6] + ' ' + data[7] + '\n'

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

	return args


def main():
	args = parse_arguments()
	translate_file(args.path_labels, args.path_images, args.difficulty, args.outfile)
	args.outfile.close()


if __name__ == '__main__':
    main()


