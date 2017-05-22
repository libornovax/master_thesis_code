"""
Composes a plot out of the given validation loss learning curves.

The script generates several files - a PDF and PNG plot.

----------------------------------------------------------------------------------------------------
python plot_multiple_learning_curves.py --paths_csv f1.csv f2.csv --labels "F1" "F2" --path_out learning_curves
----------------------------------------------------------------------------------------------------
"""

__date__   = '05/22/2017'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'


import argparse
import os
import csv

import matplotlib
matplotlib.use('Agg')  # Prevents from using X interface for plotting
from matplotlib import pyplot as plt



####################################################################################################
#                                           DEFINITIONS                                            # 
####################################################################################################

# Colors of the plots
COLORS = [
	'#3399FF', '#FF3300', '#FF33CC', '#40BF0D', '#FFE300', '#000000'
]


####################################################################################################
#                                            FUNCTIONS                                             # 
####################################################################################################


def load_csv(path_csv):
	"""
	Loads a CSV file with points on a validation loss curve. The CSV file must have the "iter 
	loss_train loss_valid" columns separated by a single space to be a valid file.

	Input:
		path_csv: Path to a CSV file
	Output:
		iterations, losses (lists of points on the learning curve)
	"""
	iterations   = []
	losses       = []

	with open(path_csv, 'r') as infile:
		csv_reader = csv.DictReader(infile, delimiter=' ')

		# Check field names
		if not set(csv_reader.fieldnames).issuperset(set(['iter', 'loss_train', 'loss_valid'])):
			print('ERROR: File "%s" does not contain required fields!'%(path_csv))
			exit(1)

		for row in csv_reader:
			if float(row['iter']) != 0 and float(row['loss_valid']) != 0:
				iterations.append(float(row['iter']))
				losses.append(float(row['loss_valid']))

	return iterations, losses


def initialize_plot(title):
	"""
	Initializes the plotting canvas for plotting the learning curves.

	Input:
		title: Title of the plot
	"""
	# Equal error rate line

	plt.grid()
	plt.xlabel('iteration')
	plt.ylabel('loss')
	plt.title(title)


def save_plot(path_out):
	"""
	Saves the current plot to PDF and PNG.

	Input:
		path_out: Path to the output file(s) (without extension)
	"""
	plt.legend(prop={'size':13})

	plt.savefig(path_out + '.pdf', bbox_inches='tight')
	plt.savefig(path_out + '.png', bbox_inches='tight')

	print('-- Plot saved to: ' + path_out)


def plot_learning_curves(paths_csv, labels, path_out, title, ylimit):
	"""
	Creates the plot with the given learning curves and saves it in path_out.

	Input:
		paths_csv: Paths to CSV files with learning curves' data (list)
		labels:    Labels of the learning curves (list)
		path_out:  Path to the output file without extension
		title:     Title of the plot
		ylimit:    Limit of the y axis (or None)
	"""
	initialize_plot(title)

	xmin = 9999999999
	xmax = 0

	for i, path in enumerate(paths_csv):
		iterations, losses = load_csv(path)

		plt.plot(iterations, losses, label=labels[i]+' - valid', color=COLORS[i], linewidth=2, 
				 linestyle='--')

		if iterations[0] < xmin: xmin = iterations[0]
		if iterations[-1] > xmax: xmax = iterations[-1]


	if ylimit is not None:
		plt.axis((xmin, xmax, 0, ylimit))

	save_plot(path_out)



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
	parser = argparse.ArgumentParser(description='Combine several learning curves into one plot.')
	parser.add_argument('--paths_csv', nargs='+', type=str, required=True,
						help='Paths to the CSV files with learning curve points. They must have ' \
						'"iter loss_train loss_valid" columns')
	parser.add_argument('--labels', nargs='+', type=str, required=True,
						help='Labels of the learning curves. In the order of the CSV files')
	parser.add_argument('--path_out', type=str, required=True,
						help='Path to the output file (without extension) - extensions will be ' \
						'added automatically because more files will be generated')
	parser.add_argument('--title', type=str, default='',
						help='Title of the plot')
	parser.add_argument('--ylimit', type=float, default=None,
						help='Clip the y axis on this value')

	args = parser.parse_args()

	for path in args.paths_csv:
		if not check_path(path):
			parser.print_help()
			exit(1)

	if len(args.paths_csv) != len(args.labels):
		print('ERROR: Number of CSV files and labels must be the same! (%d != %d)'%(len(args.paths_csv), len(args.labels)))
		exit(1)

	return args


def main():
	args = parse_arguments()

	plot_learning_curves(args.paths_csv, args.labels, args.path_out, args.title, args.ylimit)


if __name__ == '__main__':
    main()


