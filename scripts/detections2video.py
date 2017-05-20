"""
Generates a video file from the given detection file. Either 2D bounding boxes are plotted into
the image if BBTXT or BB3TXT file is supplied or 3D bounding boxes if BB3TXT and the corresponding
PGP file is supplied.

----------------------------------------------------------------------------------------------------
python detections2video.py path/detections.bbtxt detection_mapping path/out.mp4
----------------------------------------------------------------------------------------------------
"""

__date__   = '05/19/2017'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'


import argparse
import os
import cv2

from data.shared.bb3txt import load_bb3txt
from data.shared.bbtxt import load_bbtxt
from data.shared.pgp import load_pgp
from data.mappings.utils import LabelMappingManager


####################################################################################################
#                                           DEFINITIONS                                            # 
####################################################################################################

# Colors of the bounding boxes of the categories
COLORS = {
	'car': '#3399FF',
	'person': '#FF33CC',
}

# Initialize the LabelMappingManager
LMM = LabelMappingManager()


####################################################################################################
#                                            FUNCTIONS                                             # 
####################################################################################################

def ri(x):
	return int(round(x))


def hex2bgr(hex):
	hex = hex.strip('#')
	return (int(hex[4:6], 16), int(hex[2:4], 16), int(hex[0:2], 16))


def get_path_to_image(path_image, path_datasets=None):
	"""
	If path_datasets is not none it replaces the path in the image path with this one.
	"""
	if path_datasets is not None:
		# Find the position of the "datasets" folder in the path
		pos = path_image.find('/datasets/') + 1
		if pos >= 0:
			path_image = os.path.join(path_datasets, path_image[pos:])
		return path_image
	else:
		return path_image


####################################################################################################
#                                             CLASSES                                              # 
####################################################################################################

class VideoGenerator(object):
	"""
	"""
	def __init__(self, path_detections, detections_mapping, confidence, fps, offset=0, 
				 length=99999999, path_datasets=None, path_pgp=None):
		"""
		Input:
			path_detections:    Path to the BBTXT or BB3TXT file with detections
			detections_mapping: Name of the mapping of the path_detections file
			confidence:         Minimum confidence of a detection to be displayed
			fps:                Frame rate of the generated video
			offset:             Offset of the video from the start (in frames)
			length:             Length of the video in frames
			path_datasets:      Path to the "datasets" folder on this machine, replaces the path
								that is in the BBTXT (BB3TXT) files if provided
			path_pgp:           Path to the PGP file with image projection matrices and ground plane
			                    equations
		"""
		super(VideoGenerator, self).__init__()
		
		self.confidence    = confidence
		self.fps           = fps
		self.offset        = offset
		self.max_length    = length
		self.path_datasets = path_datasets
		self.path_pgp      = path_pgp

		self.detections_mapping = LMM.get_mapping(detections_mapping)


		print('-- Loading detections: ' + path_detections)
		if path_pgp is not None:
			self.iml_detections = load_bb3txt(path_detections)
			print('-- Loading PGP: ' + path_detections)
			self.pgps = load_pgp(path_pgp)
		else:
			self.iml_detections = load_bbtxt(path_detections)
			self.pgps = None

		self._create_sorted_sequence()


	def _create_sorted_sequence(self):
		"""
		Creates a sorted list of files, which we will be cycling through.
		"""
		self.file_sequence = self.iml_detections.keys()
		self.file_sequence.sort()


	def _plot_bboxes(self, image, filename):
		"""
		Plots bounding boxes into the image.

		Input:
			image: np.array (cv2.imread read) image
			filename: Path to the image
		"""
		bbs = self.iml_detections[filename]

		if self.pgps is not None:
			# Plot 3D bounding boxes
			if filename not in self.pgps.keys():
				print('ERROR: Missing PGP for file "' + filename + '"!')
				exit(2)

			pgp = self.pgps[filename]

			for bb in bbs:
				if bb.confidence >= self.confidence:
					# Get all 4 bounding box corners in 3D (FBL FBR RBR RBL FTL FTR RTR RTL)
					X_3x8 = pgp.reconstruct_bb3d(bb)
					# Project them back to image
					x_2x8 = pgp.project_X_to_x(X_3x8)

					color = hex2bgr(COLORS[self.detections_mapping[bb.label]])

					# Plot front side
					cv2.line(image, (ri(x_2x8[0,4]), ri(x_2x8[1,4])), (ri(x_2x8[0,5]), ri(x_2x8[1,5])), (0,255,0), 2)
					cv2.line(image, (ri(x_2x8[0,5]), ri(x_2x8[1,5])), (ri(x_2x8[0,1]), ri(x_2x8[1,1])), (0,255,0), 2)
					cv2.line(image, (ri(x_2x8[0,0]), ri(x_2x8[1,0])), (ri(x_2x8[0,1]), ri(x_2x8[1,1])), (0,255,0), 2)
					cv2.line(image, (ri(x_2x8[0,0]), ri(x_2x8[1,0])), (ri(x_2x8[0,4]), ri(x_2x8[1,4])), (0,255,0), 2)
					# Plot rear side
					cv2.line(image, (ri(x_2x8[0,2]), ri(x_2x8[1,2])), (ri(x_2x8[0,3]), ri(x_2x8[1,3])), (0,0,255), 2)
					cv2.line(image, (ri(x_2x8[0,7]), ri(x_2x8[1,7])), (ri(x_2x8[0,3]), ri(x_2x8[1,3])), (0,0,255), 2)
					cv2.line(image, (ri(x_2x8[0,7]), ri(x_2x8[1,7])), (ri(x_2x8[0,6]), ri(x_2x8[1,6])), (0,0,255), 2)
					cv2.line(image, (ri(x_2x8[0,6]), ri(x_2x8[1,6])), (ri(x_2x8[0,2]), ri(x_2x8[1,2])), (0,0,255), 2)
					# Plot connections
					cv2.line(image, (ri(x_2x8[0,4]), ri(x_2x8[1,4])), (ri(x_2x8[0,7]), ri(x_2x8[1,7])), color, 2)
					cv2.line(image, (ri(x_2x8[0,5]), ri(x_2x8[1,5])), (ri(x_2x8[0,6]), ri(x_2x8[1,6])), color, 2)
					cv2.line(image, (ri(x_2x8[0,1]), ri(x_2x8[1,1])), (ri(x_2x8[0,2]), ri(x_2x8[1,2])), color, 2)
					cv2.line(image, (ri(x_2x8[0,0]), ri(x_2x8[1,0])), (ri(x_2x8[0,3]), ri(x_2x8[1,3])), color, 2)

					# txt = self.detections_mapping[bb.label] + ' %.3f'%(bb.confidence)
					# cv2.putText(image, txt, (ri(bb.fblx), ri(bb.ftly-5)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color)

		else:
			# Plot 2D bounding boxes
			for bb in bbs:
				if bb.confidence >= self.confidence:
					color = hex2bgr(COLORS[self.detections_mapping[bb.label]])
					cv2.rectangle(image, (ri(bb.xmin), ri(bb.ymin)), (ri(bb.xmax), ri(bb.ymax)), color, 2)

					# txt = self.detections_mapping[bb.label] + ' %.3f'%(bb.confidence)
					# cv2.putText(image, txt, (ri(bb.xmin), ri(bb.ymin-5)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color)


	def generate_video(self, path_out):
		"""
		Generates the video from the currently opened BBTXT or BB3TXT file.

		Input:
			path_out: Path to the output MP4 file
		"""
		img = cv2.imread(get_path_to_image(self.file_sequence[0], self.path_datasets))
		height, width, layers =  img.shape

		fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
		video = cv2.VideoWriter(path_out, fourcc, self.fps, (width, height))

		length = 0
		for i in range(len(self.file_sequence)):
			if i < self.offset: continue
			if length >= self.max_length: break
			print('Processing frame ' + str(i))

			filename = self.file_sequence[i]
			image = cv2.imread(get_path_to_image(filename, self.path_datasets))

			# Plot the bounding boxes into the image
			self._plot_bboxes(image, filename)

			# Add the image to the video sequence
			if image.shape != img.shape:
				image = cv2.resize(image, (height, width))

			video.write(image)

		video.release()
		cv2.destroyAllWindows()



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
	parser = argparse.ArgumentParser(description='Generate a video from detections.')
	parser.add_argument('path_detections', metavar='path_detections', type=str,
						help='Path to the BBTXT or BB3TXT file with detections to be shown')
	parser.add_argument('detections_mapping', metavar='detections_mapping', type=str,
						help='Label mapping of the detections file. One of ' \
						+ str(LMM.available_mappings()))
	parser.add_argument('path_out', metavar='path_out', type=str,
						help='Path to the output MP4 file')
	parser.add_argument('--confidence', type=float, default=0.5,
						help='Minimum confidence of shown bounding boxes')
	parser.add_argument('--fps', type=int, default=24,
						help='Frame rate of the generated video')
	parser.add_argument('--offset', type=int, default=0,
						help='Offsets the start of the video by the given number of frames')
	parser.add_argument('--length', type=int, default=999999999,
						help='Length of the video (in frames)')
	parser.add_argument('--path_datasets', type=str, default=None,
						help='Path to the "datasets" folder on this machine - will be used to ' \
						'replace the path from the BBTXT and BB3TXT files so we could show the ' \
						'images even if the test was carried out on a different PC')
	parser.add_argument('--path_pgp', type=str, default=None,
						help='Path to the PGP file with image projection matrices and ground ' \
						'plane equations. This allows showing the whole 3D bounding box')

	args = parser.parse_args()

	if not check_path(args.path_detections) or \
			(args.path_datasets is not None and not check_path(args.path_datasets, True)) or \
			(args.path_pgp is not None and not check_path(args.path_pgp)):
		parser.print_help()
		exit(1)

	if os.path.exists(args.path_out):
		check = raw_input('The output file "' + args.path_out + '" already exists!\nDo you want to rewrite it? [y/n]\n')
		if check != 'y':
			print('Video generation aborted.')
			exit(1)
		else:
			os.remove(args.path_out)

	return args


def main():
	args = parse_arguments()

	print('-- DETECTIONS TO VIDEO CONVERTER')

	vg = VideoGenerator(args.path_detections, args.detections_mapping, args.confidence, args.fps, 
						args.offset, args.length, args.path_datasets, args.path_pgp)
	
	vg.generate_video(args.path_out)


if __name__ == '__main__':
    main()


