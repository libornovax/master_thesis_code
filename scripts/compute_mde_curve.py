"""
Generates mean distance error (MDE) curve for the given detections and ground truth BB3TXT files.
The script generates several files - a PDF plot, PNG plot and CSV files with the values specifying
the curve.

This measure was introduced in [Mousavian, Arsalan, et al. "3D Bounding Box Estimation Using Deep
Learning and Geometry." arXiv preprint arXiv:1612.00496 (2016)]

----------------------------------------------------------------------------------------------------
python compute_mde_curve.py path_gt gt_mapping path_detections detections_mapping path_pgp path_out
----------------------------------------------------------------------------------------------------
"""

__date__   = '16/05/2017'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'


import argparse
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Prevents from using X interface for plotting
from matplotlib import pyplot as plt

from data.shared.bb3txt import load_bb3txt
from data.shared.pgp import load_pgp
from data.mappings.utils import LabelMappingManager


####################################################################################################
#                                           DEFINITIONS                                            # 
####################################################################################################

# Number of steps (points) on the precision/recall curve
NUM_STEPS = 20

# Categories (labels) for which we plot the PR curves
CATEGORIES = [
	'car',
	# 'person'
]

# Colors of the categories
COLORS = {
	'car': '#3399FF', 
	'person': '#FF33CC',
	# '#FF3300', '#40BF0D', '#FFE300', '#000000'
}

# Initialize the LabelMappingManager
LMM = LabelMappingManager()


####################################################################################################
#                                            FUNCTIONS                                             # 
####################################################################################################


def center_from_X_3x8(X_3x8):
	"""
	Computes the center of the 3D bounding box with corners (FBL FBR RBR RBL FTL FTR RTR RTL) in
	the X_3x8 matrix.

	Input:
		X_3x8: matrix with 3D coordinates of a 3D bounding box corners ordered (FBL FBR RBR RBL FTL
		       FTR RTR RTL)
	Returns:
		BBC_3x1 matrix with coordinates of the center
	"""
	BBC_3x1 = (X_3x8[:,0] + X_3x8[:,6]) / 2.0

	return BBC_3x1


def compute_distance_and_error(bb3d_gt, bb3d_det, pgp):
	"""
	Computes the distance to the ground truth object from the camera and the error in the bounding
	box center position.

	Input:
		bb3d_gt:  BB3D object with ground truth
		bb3d_det: BB3D object with the detection
		pgp:      PGP for the image from which the bounding boxes are
	Returns:
		distance, error
	"""
	X_gt_3x8  = pgp.reconstruct_bb3d(bb3d_gt)
	X_det_3x8 = pgp.reconstruct_bb3d(bb3d_det)

	# Coordinates of the bounding box centers
	X_gt_C_3x1  = center_from_X_3x8(X_gt_3x8)
	X_det_C_3x1 = center_from_X_3x8(X_det_3x8)

	distance = np.linalg.norm(X_gt_C_3x1 - pgp.C_3x1)
	error    = np.linalg.norm(X_gt_C_3x1 - X_det_C_3x1)

	return distance, error



def distances_and_errors(gt, detections, min_iou, pgp):
	"""
	Computes distances and distance errors for the given ground truth and detections.

	Input:
		gt:             List of BB3D ground truth bounding boxes
		detections:     List of BB3D detections' bounding boxes
		min_iou:        Minimum intersection over union threshold for a true positive
		pgp:            The PGP for the image
	Returns:
		distances, errors
	"""
	# First compute a matrix of intersection over unions - we will be searching best matches in it
	ious = np.zeros([len(gt), len(detections)], dtype=np.float)
	for i in range(len(gt)):
		for j in range(len(detections)):
			ious[i,j] = gt[i].bb2d.iou(detections[j].bb2d)

	# Indices of ground truth bbs and detections - will be used to track which ones we removed
	gtis = range(len(gt))
	dtis = range(len(detections))

	distances = []
	errors    = []

	# Find maxima - best matches -> true positives
	while ious.shape[0] > 0 and ious.shape[1] > 0 and ious.max() > min_iou:
		i, j = np.unravel_index(ious.argmax(), ious.shape)

		# Compute the distance and distance error
		distance, error = compute_distance_and_error(gt[gtis[i]], detections[dtis[j]], pgp)
		distances.append(distance)
		errors.append(error)

		# Remove this ground truth and detection from the matrix
		ious = np.delete(ious, i, axis=0)
		ious = np.delete(ious, j, axis=1)
		# Remove this ground truth and detection from index lists
		del gtis[i]
		del dtis[j]

	return distances, errors


def mde_curve_points(distances, errors, thresholds):
	"""
	Compute points of a MDE curve from the given error measures.

	Input:
		distances:  List of distances to the ground truth
		errors:     List of errors to the corresponding ground truths
		thresholds: List of distance thresholds
	Output:
		means, stds
	"""
	errors_split = []
	for t in range(len(thresholds)+1):
		errors_split.append([])

	# Split the errors into bins by distance 
	for i in range(len(distances)):
		assigned = False
		for t in range(len(thresholds)):
			if distances[i] < thresholds[t]:
				assigned = True
				errors_split[t].append(errors[i])

		if not assigned:
			errors_split[-1].append(errors[i])


	means = []
	stds  = []
	for t in range(len(thresholds)+1):
		errs = np.array(errors_split[t])
		mn  = np.mean(errs)
		dev = np.std(errs)

		means.append(mn)
		stds.append(dev)

		print('%d: %f +- %f (%d detections)'%(t, mn, dev, errs.shape[0]))

	return means, stds


####################################################################################################
#                                             CLASSES                                              # 
####################################################################################################

class MDEPlotter(object):
	"""
	Takes care of plotting mean distance error (MDE) curves for the given combination of ground
	truth and detection BB3TXT files.
	"""
	def __init__(self, path_gt, gt_mapping, path_detections, detections_mapping, path_pgp, iou, 
			     title):
		"""
		Input:
			path_gt:            Path to the BB3TXT file with ground truth
			gt_mapping:         Name of the mapping of the path_gt BB3TXT file
			path_detections:    Path to the BB3TXT file with detections
			detections_mapping: Name of the mapping of the path_detections BB3TXT file
			path_pgp:           Path to the PGP file with ground planes and P matrices
			iou:                Minimum intersection over union of a true positive detection
			title:              Title of the plot
		"""
		super(MDEPlotter, self).__init__()
		
		self.iou      = iou
		self.title    = title

		print('-- Loading ground truth: ' + path_gt)
		self.iml_gt         = load_bb3txt(path_gt)
		print('-- Loading detections: ' + path_detections)
		self.iml_detections = load_bb3txt(path_detections)
		print('-- Loading PGP: ' + path_pgp)
		self.pgps           = load_pgp(path_pgp)

		# We only need the list of images with detections
		self.file_list = list(self.iml_detections.keys())
		self._check_file_list()

		# Get both label mappings
		self.gt_mapping 		= LMM.get_mapping(gt_mapping)
		self.detections_mapping = LMM.get_mapping(detections_mapping)

		self._initialize_plot()

		self.categories   = []
		self.thresholds   = [10, 20, 30, 40]
		self.thresholds_x = [10, 20, 30, 40, 50]
		self.thresholds_t = ['[0, 10)', '[10, 20)', '[20, 30)', '[30, 40)', '> 40']
		self.errors       = [[], [], [], [], []]

		
	def _check_file_list(self):
		"""
		Checks if the loaded ground truth and detections have some common files.
		"""
		if len(self.file_list) == len(self.iml_gt)+len(self.iml_detections):
			# There is no intersection of files in the data -> probably wrong ground truth
			print('ERROR: The detections and ground truth files do not have common paths! ' \
				  'Probably a wrong ground truth file was loaded...')
			exit(1)
	

	def _initialize_plot(self):
		"""
		Initializes the plotting canvas for plotting the MDE curves
		"""
		plt.xlabel('distance (m)')
		plt.ylabel('mean distance error (m)')
		plt.title(self.title)


	def _add_curve(self, distances, errors, category):
		"""
		Puts a new MDE curve into the plot.

		Input:
			distances: List of distances to the ground truth
			errors:    List of errors to the corresponding ground truths
			category: Object category (label), which the curve corresponds to
		"""
		means, stds = mde_curve_points(distances, errors, self.thresholds)

		plt.plot(self.thresholds_x, means, label='Ours', color=COLORS[category], linewidth=2)
		# plt.plot(self.thresholds_x, means, label=category, color=COLORS[category], linewidth=2)

		plt.errorbar(self.thresholds_x, means, yerr=stds, color=COLORS[category])

		# self.categories.append(category)
		# self.precisions.append(precisions)
		# self.recalls.append(recalls)
		# self.precisionsr.append(precisionsr)
		# self.recallsr.append(recallsr)
		# self.precisionsd.append(precisionsd)
		# self.recallsd.append(recallsd)
		# self.precisionsrd.append(precisionsrd)
		# self.recallsrd.append(recallsrd)
		# self.tps.append(tps)
		# self.fps.append(fps)
		# self.fns.append(fns)
		# self.fnsr.append(fnsr)
		# self.fpsd.append(fpsd)


	def plot(self, category):
		"""
		Input:
			category: Object category (label), which the curve corresponds to
		"""
		print('-- Plotting category: ' + category)

		distances = []
		errors    = []

		# Process each image from the file list
		for filename in self.file_list:
			if filename in self.iml_gt:
				# Filter the bounding boxes - we only want the current category
				gt_category = [bb for bb in self.iml_gt[filename]
							   if self.gt_mapping[bb.label] == category]
			else:
				gt_category = []

			if filename in self.iml_detections:
				# Filter the bounding boxes - we only want the current category
				detections_category = [bb for bb in self.iml_detections[filename]
									   if self.detections_mapping[bb.label] == category]
			else:
				detections_category = []

			pgp = self.pgps[filename]


			dists, errs = distances_and_errors(gt_category, detections_category, self.iou, pgp)

			distances = distances + dists
			errors    = errors + errs


		# Compute mean distance error curve and plot it
		self._add_curve(distances, errors, category)


	def save_plot(self, path_out):
		"""
		Saves the current plot to PDF, PNG and CSV.

		Input:
			path_out: Path to the output file(s) (without extension)
		"""
		# Reference methods
		plt.plot(self.thresholds_x, [1.55, 1.7, 2.6, 4.25, 6.5], label='SubCNN', color='#FF3300', linewidth=2)
		plt.errorbar(self.thresholds_x, [1.55, 1.7, 2.6, 4.25, 6.5], yerr=[0.1, 0.07, 0.15, 0.3, 0.55], color='#FF3300')
		plt.plot(self.thresholds_x, [1.5, 1.05, 1.8, 2.4, 2.95], label='Deep3DBox', color='#40BF0D', linewidth=2)
		plt.errorbar(self.thresholds_x, [1.5, 1.05, 1.8, 2.4, 2.95], yerr=[0.1, 0.05, 0.1, 0.13, 0.3], color='#40BF0D')


		plt.axis((8, 52, 0, 9))

		plt.xticks(self.thresholds_x, self.thresholds_t)
		plt.legend(loc='upper left', prop={'size':13})

		plt.savefig(path_out + '.pdf')
		plt.savefig(path_out + '.png')

		# Save each category to a different CSV file
		# for c in range(len(self.categories)):
		# 	with open(path_out + '_' + self.categories[c] + '.csv', 'w') as outfile:
		# 		outfile.write('tp fp fn fnr fpd precision recall precisionr recallr precisiond ' \
		# 					  'recalld precisionrd recallrd\n')
		# 		for i in range(len(self.recalls[c])):
		# 			outfile.write('%d %d %d %d %d %f %f %f %f %f %f %f %f\n'%(self.tps[c][i],
		# 						  self.fps[c][i], self.fns[c][i], self.fnsr[c][i], self.fpsd[c][i], 
		# 						  self.precisions[c][i], self.recalls[c][i], self.precisionsr[c][i], 
		# 						  self.recallsr[c][i], self.precisionsd[c][i], self.recallsd[c][i], 
		# 						  self.precisionsrd[c][i], self.recallsrd[c][i]))

		print('-- Plots saved to: ' + path_out)



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
	parser = argparse.ArgumentParser(description='Plot the mean distance error (MDE) curve.')
	parser.add_argument('path_gt', metavar='path_gt', type=str,
						help='Path to the BB3TXT ground truth file')
	parser.add_argument('gt_mapping', metavar='gt_mapping', type=str,
						help='Label mapping of the ground truth BB3TXT file. One of ' \
						+ str(LMM.available_mappings()))
	parser.add_argument('path_detections', metavar='path_detections', type=str,
						help='Path to the BB3TXT file with detections that is to be evaluated')
	parser.add_argument('detections_mapping', metavar='detections_mapping', type=str,
						help='Label mapping of the detections BB3TXT file. One of ' \
						+ str(LMM.available_mappings()))
	parser.add_argument('path_pgp', metavar='path_pgp', type=str,
						help='Path to the PGP file with information about P matrices and ground ' \
						'plane')
	parser.add_argument('path_out', metavar='path_out', type=str,
						help='Path to the output file (without extension) - extensions will be ' \
						'added automatically because more files will be generated')
	parser.add_argument('--iou', type=float, default=0.7,
						help='Minimum intersection over union (IOU) for a detection to be counted' \
						' as a true positive')
	parser.add_argument('--title', type=str, default='',
						help='Title of the plot')

	args = parser.parse_args()

	if not check_path(args.path_detections) or not check_path(args.path_gt) \
			or not check_path(args.path_pgp):
		parser.print_help()
		exit(1)

	if args.iou <= 0.0 or args.iou > 1.0:
		print('ERROR: Invalid number for IOU "%f"! Must be in (0,1].'%(args.iou))
		exit(1)

	return args


def main():
	args = parse_arguments()

	plotter = MDEPlotter(args.path_gt, args.gt_mapping, args.path_detections,
						args.detections_mapping, args.path_pgp, args.iou, args.title)

	# Plot all categories
	for category in CATEGORIES:
		plotter.plot(category)

	plotter.save_plot(args.path_out)


if __name__ == '__main__':
    main()


