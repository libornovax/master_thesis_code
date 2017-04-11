"""
Generates precision/recall curves for the given detections and ground truth BBTXT files.
The script generates several files - a PDF plot, PNG plot and CSV files with the precision and
recall values, which were used to generate the curves.

----------------------------------------------------------------------------------------------------
python compute_pr_curve.py path_gt gt_mapping path_detections detections_mapping path_out
----------------------------------------------------------------------------------------------------
"""

__date__   = '12/02/2016'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'


import argparse
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Prevents from using X interface for plotting
from matplotlib import pyplot as plt

from data.shared.bbtxt import load_bbtxt
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

def tp_fp_fn(gt, detections, min_iou, dont_care):
	"""
	Computes true positives (tp), false positives (fp) and false negatives (fn) from the given two
	sets of bounding boxes - ground truth and detections.

	Input:
		gt:             List of BB2D ground truth bounding boxes
		detections:     List of BB2D detections' bounding boxes
		min_iou:        Minimum intersection over union threshold for a true positive
		dont_care:      List of BB2D don't care regions
	Output:
		tp, fp, fn, fnr, fpd (all ints)
		fnr: False negatives on only required ground truth
		fpd: False positives only from the outside of the specified don't care regions
	"""
	# First compute a matrix of intersection over unions - we will be searching best matches in it
	ious = np.zeros([len(gt), len(detections)], dtype=np.float)
	for i in range(len(gt)):
		for j in range(len(detections)):
			ious[i,j] = gt[i].iou(detections[j])

	# Indices of ground truth bbs and detections - will be used to track which ones we removed
	gtis = range(len(gt))
	dtis = range(len(detections))

	# Find maxima - best matches -> true positives
	tp = 0
	while ious.shape[0] > 0 and ious.shape[1] > 0 and ious.max() > min_iou:
		i, j = np.unravel_index(ious.argmax(), ious.shape)
		# Remove this ground truth and detection from the matrix
		ious = np.delete(ious, i, axis=0)
		ious = np.delete(ious, j, axis=1)
		# Remove this ground truth and detection from index lists
		del gtis[i]
		del dtis[j]
		tp = tp + 1

	# If we have some rows and columns left they are falses
	fp = ious.shape[1]  # Unmatched detections
	fn = ious.shape[0]  # Unmatched ground truth

	# Remove the remaining ground truth, which is not required
	fnr = fn
	for i in gtis:
		if not gt[i].required: fnr -= 1

	# Remove the remaining detections, which lie inside of don't care regions
	fpd = fp
	for i in dtis:
		for d in range(len(dont_care)):
			# If more than 75% of the detection lies inside of a don't care region we discard it
			if detections[i].intersection_area(dont_care[d]) / detections[i].area() > 0.75:
				fpd -= 1;

	return int(tp), int(fp), int(fn), int(fnr), int(fpd)


def pr_curve_points(tps, fps, fns):
	"""
	Compute points of precision/recall curve from the given error measures.

	Input:
		tps: np.array of true positives' counts (length N)
		fps: np.array of false positives' counts (length N)
		fns: np.array of false negatives' counts (length N)
	Output:
		precisions, recalls (lists of floats of length N)
	"""
	precisions = list(np.true_divide(tps, tps+fps))
	recalls    = list(np.true_divide(tps, tps+fns))

	# Points 0,0 are obviously wrong so this will prevent them from being plotted
	for i in range(len(precisions)):
		if precisions[i] == 0 and recalls[i] == 0:
			precisions[i] = np.nan
			recalls[i]    = np.nan

	return precisions, recalls


####################################################################################################
#                                             CLASSES                                              # 
####################################################################################################

class PRPlotter(object):
	"""
	Takes care of plotting precision/recall (PR) curves for the given combination of ground truth
	and detection BBTXT files.
	"""
	def __init__(self, path_gt, gt_mapping, path_detections, detections_mapping, iou, title):
		"""
		Input:
			path_gt:            Path to the BBTXT file with ground truth
			gt_mapping:         Name of the mapping of the path_gt BBTXT file
			path_detections:    Path to the BBTXT file with detections
			detections_mapping: Name of the mapping of the path_detections BBTXT file
			iou:                Minimum instersection over union of a true positive detection
			title:              Title of the plot
		"""
		super(PRPlotter, self).__init__()
		
		self.iou      = iou
		self.title    = title

		print('-- Loading ground truth: ' + path_gt)
		self.iml_gt         = load_bbtxt(path_gt)
		print('-- Loading detections: ' + path_detections)
		self.iml_detections = load_bbtxt(path_detections)

		# IMPORTANT! Get complete list of images! We cannot just take the ground truth files because
		# the BBTXT does not contain empty images, neither we can take just the detections because
		# that BBTXT does not contain files without detections (which could possibly contain ground
		# truth objects). Therefore we have to take a union of the two. The processed images, which
		# have no ground truth and no detections are not a problem because they do not contribute
		# to the statistics
		self.file_list = list(set(self.iml_gt.keys()).union(self.iml_detections.keys()))
		self._check_file_list()

		# Get both label mappings
		self.gt_mapping 		= LMM.get_mapping(gt_mapping)
		self.detections_mapping = LMM.get_mapping(detections_mapping)

		self._initialize_plot()

		self.categories = []
		self.tps        = []
		self.fps        = []
		self.fns        = []
		self.precisions = []
		self.recalls    = []


		
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
		Initializes the plotting canvas for plotting the PR curves
		"""
		# Equal error rate line
		plt.plot((0, 1), (0, 1), c='#DDDDDD')

		plt.grid()
		plt.xlabel('precision')
		plt.ylabel('recall')
		plt.title(self.title + ' (iou=%.2f)'%(self.iou))


	def _add_curve(self, tps, fps, fns, fnsr, fpsd, category):
		"""
		Puts a new PR curve into the plot.

		Input:
			tps:      np.array of true positives' counts (length N)
			fps:      np.array of false positives' counts (length N)
			fns:      np.array of false negatives' counts (length N)
			fnsr:     np.array of false negatives' counts on required gt (length N)
			fpsd:     np.array of false negatives' counts outside of don't care regions (length N)
			category: Object category (label), which the curve corresponds to
		"""
		# Compute the precision and recall for the PR curve
		precisions, recalls     = pr_curve_points(tps, fps, fns)
		precisionsr, recallsr   = pr_curve_points(tps, fps, fnsr)
		precisionsd, recallsd   = pr_curve_points(tps, fpsd, fns)
		precisionsrd, recallsrd = pr_curve_points(tps, fpsd, fnsr)

		plt.plot(precisions, recalls, label=category, color=COLORS[category], linewidth=2)
		plt.plot(precisionsd, recallsd, label=category+' - don\'t care', color=COLORS[category])
		plt.plot(precisionsr, recallsr, label=category+' - required', color=COLORS[category], 
				 linestyle='--')
		plt.plot(precisionsrd, recallsrd, label=category+' - required, don\'t care', 
				 color=COLORS[category], linestyle=':')

		self.categories.append(category)
		self.precisions.append(precisions)
		self.recalls.append(recalls)
		self.tps.append(tps)
		self.fps.append(fps)
		self.fns.append(fns)


	def plot(self, category):
		"""
		Input:
			category: Object category (label), which the curve corresponds to
		"""
		print('-- Plotting category: ' + category)

		tps  = np.zeros(NUM_STEPS, dtype=np.int)
		fps  = np.zeros(NUM_STEPS, dtype=np.int)
		fns  = np.zeros(NUM_STEPS, dtype=np.int)
		fnsr = np.zeros(NUM_STEPS, dtype=np.int)  # Only required bounding boxes
		fpsd = np.zeros(NUM_STEPS, dtype=np.int)  # False positives without those in 'dontcare' regions

		for s, conf_thr in enumerate(np.linspace(0, 1, NUM_STEPS)):
			# Process each image from the file list
			for filename in self.file_list:
				if filename in self.iml_gt:
					# Filter the bounding boxes - we only want the current category
					gt_category = [bb for bb in self.iml_gt[filename]
								   if self.gt_mapping[bb.label] == category]
					dont_care   = [bb for bb in self.iml_gt[filename]
								 if self.gt_mapping[bb.label] == 'dontcare']
				else:
					gt_category = []
					dont_care   = []

				if filename in self.iml_detections:
					# Filter the bounding boxes - we only want the current category
					detections_category = [bb for bb in self.iml_detections[filename]
										   if self.detections_mapping[bb.label] == category and
										   bb.confidence >= conf_thr]
				else:
					detections_category = []

				tp, fp, fn, fnr, fpd = tp_fp_fn(gt_category, detections_category, self.iou, dont_care)

				tps[s]  = tps[s] + tp
				fps[s]  = fps[s] + fp
				fns[s]  = fns[s] + fn
				fnsr[s] = fnsr[s] + fnr
				fpsd[s] = fpsd[s] + fpd

		# Compute precision and recall and add the curve to the plot
		self._add_curve(tps, fps, fns, fnsr, fpsd, category)


	def save_plot(self, path_out):
		"""
		Saves the current plot to PDF, PNG and CSV.

		Input:
			path_out: Path to the output file(s) (without extension)
		"""
		plt.legend(loc='lower left', prop={'size':13})

		plt.savefig(path_out + '.pdf')
		plt.savefig(path_out + '.png')

		# Save each category to a different CSV file
		for c in range(len(self.categories)):
			with open(path_out + '_' + self.categories[c] + '.csv', 'w') as outfile:
				outfile.write('tp fp fn precision recall\n')
				for i in range(len(self.recalls[c])):
					outfile.write('%d %d %d %f %f\n'%(self.tps[c][i], self.fps[c][i],
								  self.fns[c][i], self.precisions[c][i], self.recalls[c][i]))

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
	parser = argparse.ArgumentParser(description='Plot the precision/recall curve.')
	parser.add_argument('path_gt', metavar='path_gt', type=str,
						help='Path to the BBTXT ground truth file')
	parser.add_argument('gt_mapping', metavar='gt_mapping', type=str,
						help='Label mapping of the ground truth BBTXT file. One of ' \
						+ str(LMM.available_mappings()))
	parser.add_argument('path_detections', metavar='path_detections', type=str,
						help='Path to the BBTXT file with detections that is to be evaluated')
	parser.add_argument('detections_mapping', metavar='detections_mapping', type=str,
						help='Label mapping of the detections BBTXT file. One of ' \
						+ str(LMM.available_mappings()))
	parser.add_argument('path_out', metavar='path_out', type=str,
						help='Path to the output file (without extension) - extensions will be ' \
						'added automatically because more files will be generated')
	parser.add_argument('--iou', type=float, default=0.5,
						help='Minimum intersection over union (IOU) for a detection to be counted' \
						' as a true positive')
	parser.add_argument('--title', type=str, default='',
						help='Title of the plot')

	args = parser.parse_args()

	if not check_path(args.path_detections) or not check_path(args.path_gt):
		parser.print_help()
		exit(1)

	if args.iou <= 0.0 or args.iou > 1.0:
		print('ERROR: Invalid number for IOU "%f"! Must be in (0,1].'%(args.iou))
		exit(1)

	return args


def main():
	args = parse_arguments()

	plotter = PRPlotter(args.path_gt, args.gt_mapping, args.path_detections,
						args.detections_mapping, args.iou, args.title)

	# Plot all categories
	for category in CATEGORIES:
		plotter.plot(category)

	plotter.save_plot(args.path_out)


if __name__ == '__main__':
    main()


