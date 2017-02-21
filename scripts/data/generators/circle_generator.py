"""
Generates artificial images with black circles of a fixed radius and noisy background. To be used
for training and testing implementations of object detectors and learning algorithms.

The annotations are generated in the BBTXT format:
filename label confidence xmin ymin xmax ymax
filename label confidence xmin ymin xmax ymax
filename label confidence xmin ymin xmax ymax
...

----------------------------------------------------------------------------------------------------
python circle_generator.py 
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


class CircleGenerator(object):
	def __init__(self, width, height, circle_radius=15):
		"""
		Input:
			width:         width of each generated image
			height:        height of each generated image
			circle_radius: radius of the circle to be placed on the images
		"""
		self.width = int(width)
		self.height = int(height)
		self.circle_radius = int(circle_radius)

		self.image_counter = 0
		self.dataset_labels = []


	def create_dataset(self, path, size):
		"""
		Generates a dataset of images and saves them to the given folder.

		Input:
			path: path to the output folder
			size: number of images in the dataset
		"""
		if not os.path.exists(path):
			os.makedirs(path)

		path_bbtxt = os.path.join(path, 'annotations.bbtxt')

		with open(path_bbtxt, 'w') as outfile:
			# Generate the images
			for i in range(size):
				image, labels = self._generate_image(path)

				self.dataset_labels.append(labels)
				cv2.imwrite(labels['path'], image)

				# Write out labels
				for bb in labels['bbs']:
					outfile.write(labels['path'] + ' 1 1 ' + str(bb['x_min']) + ' ' + str(bb['y_min']) 
						+ ' ' + str(bb['x_max']) + ' ' + str(bb['y_max']) + '\n')



	################################################################################################
	#                                          PRIVATE                                             #
	################################################################################################

	def _generate_image(self, path_out=''):
		"""
		Generates a random image with circles on it.

		Input:
			path_out: Path to the output folder (if not set the generator can only generate 
				      batches)
		Output:
			3 channel BGR image,
			dict with labels {bbs, path}
		"""
		path = os.path.join(path_out, str(self.image_counter) + '.png')

		# Random parameters of the image
		bg_intensity = int(random.uniform(50, 200))
		sigma = random.uniform(0, 60)
		num_circles = int(random.uniform(3, 7))


		# This is our random background
		image = np.random.normal(0, sigma, (self.height, self.width)) + bg_intensity
		# Crop the values to be within [0,255]
		image[image > 255] = 255
		image[image < 0] = 0

		# Place the circles to the image
		labels = {
			'bbs': [],
			'path' : path,
		}
		for i in range(num_circles):
			x = int(random.uniform(0, self.width))
			y = int(random.uniform(0, self.height))

			cv2.circle(image, (x, y), self.circle_radius, (0,), -1)

			labels['bbs'].append({
				'x_min': x - self.circle_radius,
				'y_min': y - self.circle_radius,
				'x_max': x + self.circle_radius,
				'y_max': y + self.circle_radius,
			})

		# Raise the image counter
		self.image_counter += 1

		image_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

		return image_rgb, labels



####################################################################################################
#                                               MAIN                                               # 
####################################################################################################

def parse_arguments():
	"""
	Parse input options of the script
	"""
	parser = argparse.ArgumentParser(description='Generate artificial images with circles.')

	parser.add_argument('path_out', metavar='path_out', type=str,
						help='Path to the output folder')
	parser.add_argument('dataset_size', metavar='dataset_size', type=int,
						help='Size of the dataset (number of images)')
	parser.add_argument('width', metavar='width', type=int,
	                    help='Width of the generated images')
	parser.add_argument('height', metavar='height', type=int,
	                    help='Height of the generated images')
	parser.add_argument('radius', metavar='radius', type=int,
	                    help='Radius of the circle in the image')

	args = parser.parse_args()

	return args


def main():
	args = parse_arguments()
	
	cg = CircleGenerator(args.width, args.height, args.radius)
	cg.create_dataset(args.path_out, int(args.dataset_size))


if __name__ == '__main__':
    main()


