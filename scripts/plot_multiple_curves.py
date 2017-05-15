"""
Composes a plot out of the given precision/recall (PR) curves. The PR curves are given as CSV files
with the points on the precision/recall curve. Each CSV file is loaded and plotted onto the same
canvas and is assigned the name in the corresponding given label field.

The script generates several files - a PDF and PNG plot.

----------------------------------------------------------------------------------------------------
python plot_multiple_curves.py --paths_csv f1.csv f2.csv --labels "F1" "F2" --path_out pr_curves
----------------------------------------------------------------------------------------------------
"""

__date__   = '02/17/2017'
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
	Loads a CSV file with points on a PR curve. The CSV file must have the "tp fp fn precision
	recall" columns separated by a single space to be a valid file.

	Input:
		path_csv: Path to a CSV file
	Output:
		precisions, recalls (lists of points on the PR curve)
	"""
	precisions   = []
	recalls      = []
	precisionsr  = []
	recallsr     = []
	precisionsd  = []
	recallsd     = []
	precisionsrd = []
	recallsrd    = []

	with open(path_csv, 'r') as infile:
		csv_reader = csv.DictReader(infile, delimiter=' ')

		# Check field names
		if not set(csv_reader.fieldnames).issuperset(set(['tp', 'fp', 'fn', 'precision', 'recall'])):
			print('ERROR: File "%s" does not contain required fields!'%(path_csv))
			exit(1)

		for row in csv_reader:
			if float(row['precision']) != 0 and float(row['recall']) != 0:
				precisions.append(float(row['precision']))
				recalls.append(float(row['recall']))
				precisionsr.append(float(row['precisionr']))
				recallsr.append(float(row['recallr']))
				precisionsd.append(float(row['precisiond']))
				recallsd.append(float(row['recalld']))
				precisionsrd.append(float(row['precisionrd']))
				recallsrd.append(float(row['recallrd']))

	return precisions, recalls, precisionsr, recallsr, precisionsd, recallsd, precisionsrd, recallsrd


def initialize_plot(title):
	"""
	Initializes the plotting canvas for plotting the PR curves.PR

	Input:
		title: Title of the plot
	"""
	# Equal error rate line
	plt.plot((0, 1), (0, 1), c='#DDDDDD')

	plt.grid()
	plt.xlabel('precision')
	plt.ylabel('recall')
	plt.title(title)


def save_plot(path_out):
	"""
	Saves the current plot to PDF and PNG.

	Input:
		path_out: Path to the output file(s) (without extension)
	"""
	plt.legend(loc='lower left', prop={'size':13})

	plt.savefig(path_out + '.pdf')
	plt.savefig(path_out + '.png')

	print('-- Plot saved to: ' + path_out)


def plot_pr_curves(paths_csv, labels, path_out, title, all):
	"""
	Creates the plot with the given PR curves and saves it in path_out.

	Input:
		paths_csv: Paths to CSV files with PR curves' data (list)
		labels:    Labels of the PR curves (list)
		path_out:  Path to the output file without extension
		title:     Title of the plot
		all:       True if all PR curves should be plotted
	"""
	initialize_plot(title)

	for i, path in enumerate(paths_csv):
		precisions, recalls, precisionsr, recallsr, precisionsd, recallsd, precisionsrd, \
			recallsrd = load_csv(path)

		if not all:
			plt.plot(precisions, recalls, label=labels[i], color=COLORS[i], linewidth=2)
		else:
			plt.plot(precisions, recalls, label=labels[i]+'', color=COLORS[i], linewidth=2)
			plt.plot(precisionsd, recallsd, label=labels[i]+' - don\'t care', color=COLORS[i])
			plt.plot(precisionsr, recallsr, label=labels[i]+' - required', color=COLORS[i], 
					 linestyle='--')
			plt.plot(precisionsrd, recallsrd, label=labels[i]+' - required, don\'t care', 
					 color=COLORS[i], linestyle=':')

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
	parser = argparse.ArgumentParser(description='Combine several PR curves into one plot.')
	parser.add_argument('--paths_csv', nargs='+', type=str, required=True,
						help='Paths to the CSV files with PR curve points. They must have "tp fp ' \
						'fn precision recall" columns')
	parser.add_argument('--labels', nargs='+', type=str, required=True,
						help='Labels of the PR curves. In the order of the CSV files')
	parser.add_argument('--path_out', type=str, required=True,
						help='Path to the output file (without extension) - extensions will be ' \
						'added automatically because more files will be generated')
	parser.add_argument('--title', type=str, default='',
						help='Title of the plot')
	parser.add_argument('--all', action='store_true',
						help='Plot all PR curves (not just the main one)')

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

	plot_pr_curves(args.paths_csv, args.labels, args.path_out, args.title, args.all)


if __name__ == '__main__':
    main()


