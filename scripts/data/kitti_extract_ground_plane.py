"""
Script for extracting the ground plane from the KITTI dataset.

We need to determine the ground plane position and orientation in order to be able to reconstruct
points on it, which we are trying to detect.

We will collect all the points on the ground plane from the dataset and then fit a plane to them
with RANSAC.

----------------------------------------------------------------------------------------------------
python kitti_extract_ground_plane.py path_labels
----------------------------------------------------------------------------------------------------
"""

__date__   = '04/13/2017'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'


import argparse
import os
import numpy as np
from matplotlib import pyplot as plt

from shared.geometry import R3x3_y, t3x1, Rt4x4


####################################################################################################
#                                            FUNCTIONS                                             # 
####################################################################################################





####################################################################################################
#                                             CLASSES                                              # 
####################################################################################################

class GroundPlaneEstimator(object):
	"""
	Takes care of the estimation of the ground plane position in the KITTI dataset.
	"""
	def __init__(self, path_labels):
		"""
		Input:
			path_labels: Path to the "label_2" folder of the KITTI dataset
		"""
		super(GroundPlaneEstimator, self).__init__()

		self.path_labels = path_labels
		self.gp_points   = []


	def run_estimation(self):
		"""
		Runs the whole process of estimating the ground plane.
		"""
		print('-- ESTIMATING GROUND PLANE POSITION')

		# Read label files and get all ground plane points
		print('-- Reading label files')
		self._read_label_files()
		print('-- Label files contain ' + str(len(self.gp_points)) + ' points')

		self.gp_X = np.asmatrix(np.zeros((3, len(self.gp_points))))
		for i in xrange(len(self.gp_points)):
			self.gp_X[:,i] = self.gp_points[i]

		plt.scatter(self.gp_X[2,:], self.gp_X[1,:])
		plt.show()

		# Run RANSAC on those points
		print('-- Running RANSAC plane estimation')


	def _read_label_files(self):
		"""
		Reads all label files and extract the points on the ground plane.
		"""
		filenames = [f for f in os.listdir(self.path_labels) 
					 if os.path.isfile(os.path.join(self.path_labels, f))]

		if len(filenames) != 7481:
			print('Wrong number (%d) of files in the KITTI dataset! Should be 7481.'%(len(filenames)))
			exit(1)

		# Read each label file
		for f in filenames:
			path_label_file = os.path.join(self.path_labels, f)

			self._process_label_file(path_label_file)


	def _process_label_file(self, path_label_file):
		"""
		Processes one label file.

		Input:
			path_label_file: Path to the TXT label file in KITTI format to be processed.
		"""
		with open(path_label_file, 'r') as infile_label:
			# Read the objects
			for line in infile_label:
				line = line.rstrip('\n')
				data = line.split(' ')
				
				# First element of the data is the label. We don't want to process 'Misc' and
				# 'DontCare' labels
				if data[0] == 'Misc' or data[0] == 'DontCare': continue

				# Extract the points of this object on the ground plane
				self._extract_ground_plane_pts(data)


	def _extract_ground_plane_pts(self, data):
		"""
		Extract 3D points from the object bounding box, which lie on the ground plane.

		Input:
			data: One split line of the label file (line.split(' '))
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

		# 3D box corners on the ground plane. Careful, the coordinate system of the car is that
		# x points forward, not z! (It is rotated by 90deg with respect to the camera one)
		#                 fbr, rbr,   fbl, rbl
		X = np.asmatrix([[l/2, -l/2,  l/2, -l/2],
			             [0,    0,    0,   0   ],
			             [-w/2, -w/2, w/2, w/2 ],
			             [1,    1,    1,   1   ]])
		# Rotate the 3D box around y axis and translate it to the correct position in the cam. frame
		X = Rt4x4(R3x3_y(ry), t3x1(cx, cy, cz)) * X

		self.gp_points.append(X[0:3,0])
		self.gp_points.append(X[0:3,1])
		self.gp_points.append(X[0:3,2])
		self.gp_points.append(X[0:3,3])


	def _ransac_plane(self):
		"""
		Finds "optimal" ground plane position given the points.
		"""
		pass



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


	args = parser.parse_args()

	if not os.path.exists(args.path_labels):
		print('Input path "%s" does not exist!'%(args.path_labels))
		parser.print_help()
		exit(1)

	return args


def main():
	args = parse_arguments()
	
	gpe = GroundPlaneEstimator(args.path_labels)

	gpe.run_estimation()



if __name__ == '__main__':
    main()


