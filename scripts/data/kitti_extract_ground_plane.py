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
import random

# import matplotlib
# matplotlib.use('Agg')  # Prevents from using X interface for plotting
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from shared.geometry import R3x3_y, t3x1, Rt4x4


####################################################################################################
#                                           DEFINITIONS                                            # 
####################################################################################################

# Parameter for RANSAC
# Distance from the plane (in meters), which is considered as an inlier region
INLIER_TRHESHOLD = 1.0

# Number of estimation iterations carried out by RANSAC
RANSAC_ITERS = 10000



####################################################################################################
#                                            FUNCTIONS                                             # 
####################################################################################################

def plane_3p(p1, p2, p3):
	"""
	Computes the equation of a plane passing through the 3 given points.

	Input:
		p1, p2, p3: 3x1 np.matrix coordinates of points in the plane
	Returns:
		[a, b, c, d] coefficients as a 1x4 np.matrix
	"""
	l1 = p2 - p1
	l2 = p3 - p1

	normal = np.cross(l1, l2, axis=0)
	d = - (normal[0,0]*p1[0,0] + normal[1,0]*p1[1,0] + normal[2,0]*p1[2,0])

	return np.asmatrix([normal[0,0], normal[1,0], normal[2,0], d])


def show_X_and_gp(gp_X_4xn, inliers_mask, gp_1x4):
	"""
	Show a 3D plot of the estimated ground plane.
	"""
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_aspect('equal')

	outliers_mask = inliers_mask == False
	outliers_4xn = np.array(np.compress(np.array(outliers_mask)[0], gp_X_4xn, axis=1))
	ax.scatter(outliers_4xn[2,:], outliers_4xn[0,:], -outliers_4xn[1,:], color='red')

	inliers_4xn = np.array(np.compress(np.array(inliers_mask)[0], gp_X_4xn, axis=1))
	ax.scatter(inliers_4xn[2,:], inliers_4xn[0,:], -inliers_4xn[1,:], color='green')
	

	X = np.arange(-20, 20, 1)
	Y = np.arange(-1, 10, 1)
	X, Y = np.meshgrid(X, Y)
	Z = - (gp_1x4[0,0]*X + gp_1x4[0,1]*Y + gp_1x4[0,3]) / gp_1x4[0,2]
	ax.plot_surface(Z, X, -Y, linewidth=0, alpha=0.5, antialiased=True)


	# Bounding box of the car

	ax.plot([3,3,3,3,3], [1.5, 1.5, -1.5, -1.5, 1.5], [0,-1.9,-1.9,0,0], color='green')
	ax.plot([-3,-3,-3,-3,-3], [1.5, 1.5, -1.5, -1.5, 1.5], [0,-1.9,-1.9,0,0], color='red')
	ax.plot([3, -3], [1.5, 1.5], [0,0], color='blue')
	ax.plot([3, -3], [1.5, 1.5], [-1.9,-1.9], color='blue')
	ax.plot([3, -3], [-1.5, -1.5], [0,0], color='blue')
	ax.plot([3, -3], [-1.5, -1.5], [-1.9,-1.9], color='blue')

	ax.set_xlim(-10, 100)
	ax.set_ylim(-10, 100)
	ax.set_zlim(-10, 100)

	ax.set_xlabel('Z')
	ax.set_ylabel('X')
	ax.set_zlabel('Y')
	plt.show()



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


		# Create a matrix from all the points for easier computation
		self.gp_X_4xn = np.asmatrix(np.ones((4, len(self.gp_points))))
		for i in xrange(len(self.gp_points)):
			self.gp_X_4xn[0:3,i] = self.gp_points[i]

		# plt.scatter(self.gp_X_4xn[2,:], self.gp_X_4xn[1,:])
		# plt.show()


		# Run RANSAC on those points
		print('-- Running RANSAC plane estimation')
		self._ransac_plane()


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
		i = 0
		for f in filenames:
			path_label_file = os.path.join(self.path_labels, f)

			self._process_label_file(path_label_file)

			i += 1
			if i == 1000: break


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

		Returns:
			[a, b, c, d] plane equation ax+by+cz+d=0 coefficients as a 1x4 np.matrix
		"""
		num_points = len(self.gp_points)

		# Variables for storing max number of inliers and the corresponding ground plane
		dist2_sum_min = 99999999999999999
		gp_1x4_max      = np.asmatrix(np.zeros((1,4)))
		inliers_mask_max     = np.asmatrix(np.zeros((4,0)))

		for i in range(RANSAC_ITERS):
			rp = random.sample(range(0, num_points), 3)

			# Compute the equation of the ground plane
			gp_1x4 = plane_3p(self.gp_points[rp[0]], self.gp_points[rp[1]], self.gp_points[rp[2]])

			# Check that the plane gives small errors on the original points - when we have some
			# close to singular situation we have to be careful
			if gp_1x4 * self.gp_X_4xn[:,rp[0]] > 0.000000001 or \
					gp_1x4 * self.gp_X_4xn[:,rp[1]] > 0.000000001 or \
					gp_1x4 * self.gp_X_4xn[:,rp[2]] > 0.000000001:
				print('WARNING: Solution not precise, skipping...')
				continue


			# Find and compute the number of inliers
			distances2 = np.power(gp_1x4 * self.gp_X_4xn, 2)
			dist2_sum = np.sum(distances2, axis=1)
			print(dist2_sum)
			inliers_mask = distances2 < INLIER_TRHESHOLD*INLIER_TRHESHOLD
			num_inliers = np.sum(inliers_mask, axis=1)[0,0]
			
			if num_inliers > num_inliers_max:
				print('New max inliers found: ' + str(num_inliers))
				num_inliers_max = num_inliers
				gp_1x4_max = gp_1x4
				inliers_mask_max = inliers_mask


		print('-- RANSAC FINISHED')
		print('Estimated ground plane: ' + str(gp_1x4_max))
		print('Number of inliers: ' + str(num_inliers_max) + ' out of ' + str(num_points))

		# Show a plot of the plane and inliers vs outliers
		show_X_and_gp(self.gp_X_4xn, inliers_mask_max, gp_1x4_max)

		return gp_1x4_max



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


