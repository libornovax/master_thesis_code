"""
Computes several statistics about the provided dataset. Among others, number of bounding boxes for
each category, minimum and maximum dimensions.

----------------------------------------------------------------------------------------------------
python dataset_statistics.py path/to/labels.bbtxt 'mapping'
----------------------------------------------------------------------------------------------------
"""

__date__   = '03/01/2017'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'

import argparse
import os
import numpy as np

from mappings.utils import LabelMappingManager
from mappings.utils import available_categories
from shared.bbtxt import load_bbtxt_to_list


####################################################################################################
#                                           DEFINITIONS                                            # 
####################################################################################################

# Initialize the LabelMappingManager
LMM = LabelMappingManager()



####################################################################################################
#                                            FUNCTIONS                                             # 
####################################################################################################


####################################################################################################
#                                             CLASSES                                              # 
####################################################################################################

class DatasetStats(object):
	def __init__(self, path_bbtxt, mapping):
		"""
		Input:
			path_bbtxt: Path to the BBTXT file with ground truth
			mapping:    Label mapping of the given BBTXT file
		"""
		self.path_bbtxt = path_bbtxt
		self.mapping    = LMM.get_mapping(mapping)
		self.categories = available_categories(self.mapping)

	
	def compute_statistics(self):
		"""
		"""
		bb2d_list = load_bbtxt_to_list(self.path_bbtxt)

		# Compute statistics for each category
		for category in self.categories:
			self._compute_statistics_category(bb2d_list, category)


	################################################################################################
	#                                          PRIVATE                                             #
	################################################################################################

	def _compute_statistics_category(self, bb2d_list, category):
		"""
		"""
		widths  = np.empty(len(bb2d_list))
		heights = np.empty(len(bb2d_list))

		# Extract all bounding boxes with the requested category
		i = 0
		for bb in bb2d_list:
			if self.mapping[bb.label] == category:
				widths[i]  = bb.width()
				heights[i] = bb.height()
				i += 1

		# Clip the widths and heights to the actual number of bbs for this category
		# Compute statistics on the bounding box sizes
		widths  = np.sort(widths[:i])
		heights = np.sort(heights[:i])

		
		print('-- Category: ' + category)
		print('Number of bboxes: %d'%(len(widths)))
		print('Width:  from %.1f to %.1f (median: %1.f)'%(widths[0], widths[-1], np.median(widths)))
		# print(str(widths[:5]) + ' ... ' + str(widths[-5:]))
		print('Height: from %.1f to %.1f (median: %1.f)'%(heights[0], heights[-1], np.median(heights)))
		# print(str(heights[:5]) + ' ... ' + str(heights[-5:]))
		print('--')



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
	parser = argparse.ArgumentParser(description='Compute statistics of the dataset given by a ' \
												 'BBTXT file.')

	parser.add_argument('path_bbtxt', metavar='path_bbtxt', type=str,
	                    help='A BBTXT file with ground truth')
	parser.add_argument('mapping', metavar='mapping', type=str,
						help='Label mapping of the BBTXT file. One of ' \
						+ str(LMM.available_mappings()))

	args = parser.parse_args()

	if not check_path(args.path_bbtxt):
		parser.print_help()
		exit(1)

	if args.mapping not in LMM.available_mappings():
		print('ERROR: Label mapping "' + args.mapping + '" not found!')
		parser.print_help()
		exit(1)

	return args


def main():
	args = parse_arguments()
	
	ds = DatasetStats(args.path_bbtxt, args.mapping)
	ds.compute_statistics()


if __name__ == '__main__':
    main()


