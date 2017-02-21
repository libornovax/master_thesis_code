"""
Generate training images from the UIUC training dataset - we need to place the training samples on
some background because the crops themselves are not enough for training DNNs.

The annotations are generated in the BBTXT format:
filename label confidence xmin ymin xmax ymax
filename label confidence xmin ymin xmax ymax
filename label confidence xmin ymin xmax ymax
...

----------------------------------------------------------------------------------------------------
python uiuc_generator.py path_background path_uiuc path_out dataset_size width height
----------------------------------------------------------------------------------------------------
"""

__date__   = '02/21/2016'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'

import argparse
import os
import cv2
import numpy as np
import random


class UIUCGenerator(object):
	def __init__(self, width, height, path_background, path_uiuc):
		"""
		Input:
			width: width of each generated image
			height: height of each generated image
			path_background: path to the folder with background images
			path_uiuc: path to the folder with positive UIUC images
		"""
		self.width = int(width)
		self.height = int(height)
		self.path_background = path_background
		self.path_uiuc = path_uiuc

		self.image_counter = 0


	def create_dataset(self, path, size):
		"""
		Generates a dataset of images and saves them to the given folder.

		Input:
			path: path to the output folder
			size: number of images in the dataset
		"""
		if not os.path.exists(path):
			os.makedirs(path)

		# Load the contents of the folders
		images_background = [f for f in os.listdir(self.path_background) if f != '.DS_Store']
		images_uiuc = [f for f in os.listdir(self.path_uiuc) if f != '.DS_Store']

		path_bbtxt = os.path.join(path, 'annotations.bbtxt')
		with open(path_bbtxt, 'w') as outfile:
			# Generate the images
			for i in range(size):
				# Pick a random background image and a random training sample
				name_image_bg = images_background[int(round(random.uniform(0, len(images_background)-1)))]
				name_image_fg = images_uiuc[int(round(random.uniform(0, len(images_uiuc)-1)))]

				image, labels = self._generate_image(path, name_image_bg, name_image_fg)

				# Write image
				cv2.imwrite(labels['path'], image)

				# Write out labels
				for bb in labels['bbs']:
					outfile.write(labels['path'] + ' 1 1 ' + str(bb['x_min']) + ' ' + str(bb['y_min']) 
						+ ' ' + str(bb['x_max']) + ' ' + str(bb['y_max']) + '\n')




	################################################################################################
	#                                          PRIVATE                                             #
	################################################################################################

	def _generate_image(self, path_out, name_image_bg, name_image_fg):
		"""
		Generates a random image with UIUC cars.

		Input:
			path_out: Path to the output folder
			name_image_bg: Image to be used as background
			name_image_fg: UIUC car image name
		Output:
			3 channel BGR image,
			dict with labels {bbs, path}
		"""
		path = os.path.join(path_out, str(self.image_counter) + '.png')

		# Load background and foreground image
		image_bg = cv2.imread(os.path.join(self.path_background, name_image_bg), cv2.IMREAD_GRAYSCALE)
		image_fg = cv2.imread(os.path.join(self.path_uiuc, name_image_fg), cv2.IMREAD_GRAYSCALE)


		# Crop the background image to the requested size
		if image_bg.shape[0] >= self.height and image_bg.shape[1] >= self.width:
			# We can just crop it
			image_bg = image_bg[-self.height:, 0:self.width]
		else:
			# Rescale the background image
			image_bg = cv2.resize(image_bg, (self.width, self.height))
		

		# Create blending mask - we want to smooth the edges
		mask = np.zeros(image_fg.shape, np.float32)
		cv2.rectangle(mask, (5, 5), (mask.shape[1]-5, mask.shape[0]-5), (1), -1)
		mask = cv2.GaussianBlur(mask, (15,15), 3)

		# Place the car on a random position in the image
		x = int(round(random.uniform(0, self.width-image_fg.shape[1])))
		y = int(round(random.uniform(0, self.height-image_fg.shape[0])))

		image_bg[y:y+image_fg.shape[0], x:x+image_fg.shape[1]] = (1-mask)*image_bg[y:y+image_fg.shape[0], x:x+image_fg.shape[1]] + mask*image_fg

		# Add some noise
		sigma = random.uniform(0, 30)
		image_bg = np.clip(image_bg + np.random.normal(0, sigma, (self.height, self.width)), 0, 255)


		labels = {
			'bbs': [],
			'path' : path,
		}
		labels['bbs'].append({
			'x_min': x,
			'y_min': y,
			'x_max': x + image_fg.shape[1],
			'y_max': y + image_fg.shape[0],
		})

		# Raise the image counter
		self.image_counter += 1

		image_rgb = cv2.cvtColor(image_bg.astype(np.uint8), cv2.COLOR_GRAY2BGR)

		return image_rgb, labels



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
	Parse input options of the script
	"""
	parser = argparse.ArgumentParser(description='Generate artificial images with UIUC cars.')

	parser.add_argument('path_background', metavar='path_background', type=str,
	                    help='Path to the folder with background images')
	parser.add_argument('path_uiuc', metavar='path_uiuc', type=str,
	                    help='Path to the folder with positive UIUC samples')
	parser.add_argument('path_out', metavar='path_out', type=str,
	                    help='Path to the output folder')
	parser.add_argument('dataset_size', metavar='dataset_size', type=int,
						help='Size of the dataset (number of images)')
	parser.add_argument('width', metavar='width', type=int,
	                    help='Width of the generated images')
	parser.add_argument('height', metavar='height', type=int,
	                    help='Height of the generated images')

	args = parser.parse_args()

	if not check_path(args.path_background, True) or not check_path(args.path_uiuc, True):
		parser.print_help()
		exit(1)

	return args


def main():
	args = parse_arguments()
	
	cg = UIUCGenerator(args.width, args.height, args.path_background, args.path_uiuc)
	cg.create_dataset(args.path_out, int(args.dataset_size))


if __name__ == '__main__':
    main()


