"""
Creates PROTOTXT files for a multiscale accumulator network in Caffe. The input is a configuration
file with the network description. This is a sample:
my_beautiful_network
r1
conv k3      o64
conv k3  d2  o64
pool
conv k3      o128
conv k3  d2  o128
pool
conv k3      o256
conv k3  d2  o256
macc x2
macc x4

----------------------------------------------------------------------------------------------------
python macc_net_generator.py path/to/config.txt path/to/output/folder
----------------------------------------------------------------------------------------------------
"""

__date__   = '02/26/2017'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'

import argparse
import os
from math import ceil


# The size of the circle in the accumulator with respect to max(w,h) of a bounding box
# This is only used to compute the size of detected objects by each accumulator
CIRCLE_SIZE = 0.3


####################################################################################################
#                                            FUNCTIONS                                             # 
####################################################################################################

def get_value(data, id):
	"""
	Finds a value with the given id among data and returns its value. The data are strings with one
	letters (id), followed by numbers. There are no spaces.

	Input:
		data: List of strings (something like ['o12', 'r45', 't9'])
		id:   Id of the item to find - its string id
	"""
	for val in data:
		if id in val:
			return int(val[len(id):])

	return None


####################################################################################################
#                                             CLASSES                                              # 
####################################################################################################

class MACCNetGenerator(object):
	def __init__(self, path_config):
		"""
		Input:
			path_config: Path to a configuration file with net structure
		"""
		self.path_config = path_config

		self.reset()


	def reset(self):
		self.previous_layer = 'data'
		self.downsampling = 1
		self.new_conv_id = 1
		self.last_in_scale = {}
		self.last_in_scale_fov = {}
		self.fov_base = 1
		self.fov_previous = 1
		self.fov_prev_downsampling = 1
		self.accs = []
		self.min_acc_downsampling = 1


	def generate_prototxt_files(self, path_out):
		"""
		"""
		if not os.path.exists(path_out):
			os.makedirs(path_out)

		lines = []

		# Parse the configuration file and create the train_val.prototxt and deploy.protoxt files
		with open(self.path_config, 'r') as infile:
			# First line contains the name of the network
			self.name = infile.readline().rstrip('\n')
			# Second line contains the radius of the circle in the accumulator
			self.radius = get_value([infile.readline().rstrip('\n')], 'r')
			
			for line in infile:
				lines.append(line.rstrip('\n'))
				print(line)


		# Create the train_val.prototxt file
		with open(os.path.join(path_out, self.name + '_train_val.prototxt'), 'w') as outfile:
			self.reset()

			outfile.write('name: "' + self.name + '"\n\n')
			outfile.write(self._layer_data('TRAIN'))
			outfile.write(self._layer_data('TEST'))
			
			outfile.write('\n# ' + '-'*38 + ' NETWORK STRUCTURE ' + '-'*39 + ' #\n')
			outfile.write(self._downsampling())
			
			for line in lines:
				self._add_layer(line, outfile, False)
			
			outfile.write(self._layer_loss())


		# Create the deploy.prototxt file
		with open(os.path.join(path_out, self.name + '_deploy.prototxt'), 'w') as outfile:
			self.reset()

			outfile.write('name: "' + self.name + '"\n\n')
			outfile.write(self._layer_input())
			
			outfile.write('\n# ' + '-'*38 + ' NETWORK STRUCTURE ' + '-'*39 + ' #\n')
			outfile.write(self._downsampling())
			
			for line in lines:
				self._add_layer(line, outfile, True)


	################################################################################################
	#                                          PRIVATE                                             #
	################################################################################################

	def _add_layer(self, line, outfile, deploy):
		"""
		Adds one layer to the PROTOTXT file specified by the line.

		Input:
			line: string with layer description (one line from the config file)
			outfile: File handle into which we will write the layer
			deploy: True/False
		"""
		layer_type = line[:4]

		if layer_type == 'conv':
			# Convolutional layer
			outfile.write(self._layer_conv(line, deploy))
			outfile.write(self._layer_relu())
		elif layer_type == 'pool':
			# Pooling layer
			outfile.write(self._layer_pool())
		elif layer_type == 'macc':
			# Multiscale accumulator - this is also a convolutional layer, but with
			# 1 output channel
			outfile.write(self._layer_macc(line, deploy))


	def _layer_relu(self):
		"""
		Creates description of a ReLU layer.
		"""
		return ('layer {\n' \
				'  name: "relu_' + self.previous_layer + '"\n' \
				'  type: "ReLU"\n' \
				'  bottom: "' + self.previous_layer + '"\n' \
				'  top: "' + self.previous_layer + '"\n' \
				'}\n')


	def _layer_conv(self, specs, deploy=False):
		"""
		Creates a description of a convolutional layer.

		Input:
			specs: string (one line from the config file) with the layer description
			deploy: True/False - includes or does not include weight filling
		"""
		name = 'conv_x%d_%d'%(self.downsampling, self.new_conv_id)
		self.new_conv_id += 1

		# Parse specs
		data = specs[5:].split()
		num_output  = get_value(data, 'o')
		kernel_size = get_value(data, 'k')
		dilation    = get_value(data, 'd')

		if num_output is None:
			print('ERROR: Number of outputs is required in "' + specs + '"!')
			exit()
		if kernel_size is None:
			print('ERROR: Kernel size is required in "' + specs + '"!')
			exit()

		# Compute the padding
		pad = (kernel_size-1) / 2
		if dilation is not None:
			pad = ((kernel_size-1) / 2) * (dilation+1)

		# Field of view computation
		if dilation is not None:
			self.fov_base = self.fov_base-1 + ((dilation+1)*(kernel_size-1) + 1)
		else:
			self.fov_base = self.fov_base-1 + kernel_size

		fov = self.fov_base * self.downsampling + self.fov_prev_downsampling-ceil(self.downsampling/2.0)
		self.fov_previous = fov


		out  = ('layer {\n' \
				'  # ' + '-'*23 + '  FOV %d x %d  (%d+%d=%d)\n'%(fov, fov, self.fov_base * self.downsampling, self.fov_prev_downsampling-ceil(self.downsampling/2.0), fov) + \
				'  name: "' + name + '"\n' \
				'  type: "Convolution"\n' \
				'  bottom: "' + self.previous_layer + '"\n' \
				'  top: "' + name + '"\n')

		if not deploy:
			out += ('  param {\n' \
					'    lr_mult: 1\n' \
					'    decay_mult: 1\n' \
					'  }\n' \
					'  param {\n' \
					'    lr_mult: 2\n' \
					'    decay_mult: 0\n' \
					'  }\n')

		out += ('  convolution_param {\n' \
				'    num_output: %d\n'%(num_output) + \
				'    kernel_size: %d\n'%(kernel_size))
		if pad is not None:
			out +=	'    pad: %d\n'%(pad)
		if dilation is not None:
			out += '    dilation: %d\n'%(dilation+1)

		if not deploy:
			out += ('    weight_filler {\n' \
					'      type: "xavier"\n' \
					'    }\n' \
					'    bias_filler {\n' \
					'      type: "constant"\n' \
					'      value: 0\n' \
					'    }\n')

		out += ('  }\n' \
				'}\n')

		self.previous_layer = name
		self.last_in_scale[self.downsampling] = name
		self.last_in_scale_fov[self.downsampling] = fov

		return out


	def _layer_pool(self):
		"""
		Create a description of a pooling layer. When a pooling layer is created the downsampling
		automatically increases.
		"""
		# Pooling layer downsamples 2x the image
		self.downsampling *= 2
		# Restart the ids of the convolution layers
		self.new_conv_id = 1
		self.fov_base = 1
		self.fov_prev_downsampling = self.fov_previous

		name = 'pool_x%d'%(self.downsampling)

		out  = self._downsampling()
		out += ('layer {\n' \
				'  name: "' + name + '"\n' \
				'  type: "Pooling"\n' \
				'  bottom: "' + self.previous_layer + '"\n' \
				'  top: "' + name + '"\n' \
				'  pooling_param {\n' \
				'    pool: MAX\n' \
				'    kernel_size: 2\n' \
				'    stride: 2\n' \
				'  }\n' \
				'}\n')

		self.previous_layer = name

		return out


	def _layer_macc(self, specs, deploy=False):
		"""
		Creates a description of an accumulator layer from the specs.

		Input:
			specs: string (line from the config file) with the layer description
		"""
		data = specs[5:].split()
		scale  = get_value(data, 'x')

		if scale is None:
			print('ERROR: Scale is required in "' + specs + '"!')
			exit()
		if scale not in self.last_in_scale:
			print('ERROR: Accumulator of this scale cannot be created "' + specs + '"!')
			exit()

		name = 'acc_x%d'%(scale)
		bb_max = (2*self.radius+1) * scale * 1/CIRCLE_SIZE

		out  = ('layer {\n' \
				'  # -----------------------  ACCUMULATOR\n' \
				'  # -----------------------  SCALE 1/%d  (FOV %d x %d)\n'%(scale, self.last_in_scale_fov[scale], self.last_in_scale_fov[scale]) + \
				'  # -----------------------  Train to detect bounding boxes up to %dx%d px\n'%(bb_max, bb_max) + \
				'  name: "' + name + '"\n' \
				'  type: "Convolution"\n' \
				'  bottom: "' + self.last_in_scale[scale] + '"\n' \
				'  top: "' + name + '"\n')

		if not deploy:
			out += ('  param {\n' \
					'    lr_mult: 1\n' \
					'    decay_mult: 1\n' \
					'  }\n' \
					'  param {\n' \
					'    lr_mult: 2\n' \
					'    decay_mult: 0\n' \
					'  }\n')

		out += ('  convolution_param {\n' \
				'    num_output: 1\n' \
				'    kernel_size: 1\n')

		if not deploy:
			out += ('    weight_filler {\n' \
					'      type: "xavier"\n' \
					'    }\n' \
					'    bias_filler {\n' \
					'      type: "constant"\n' \
					'      value: 0\n' \
					'    }\n')

		out += ('  }\n' \
				'}\n')

		# List of accumulators - for the loss layer
		self.accs.append(name)

		# This has to be stored for the loss layer
		if self.min_acc_downsampling > scale:
			self.min_acc_downsampling = scale

		return out


	def _downsampling(self):
		"""
		Prints the current downsampling factor.
		"""
		return ('# ' + '-'*45 + ' x%3d '%(self.downsampling) + '-'*45 + ' #\n')


	def _layer_loss(self):
		"""
		Description of the MultiscaleAccumulatorLoss layer.
		"""
		out  = ('\n# ' + '-'*45 + ' LOSS ' + '-'*45 + ' #\n'
				'layer {\n' \
				'  name: "loss"\n' \
				'  type: "MultiscaleAccumulatorLoss"\n' \
				'  bottom: "label"\n')

		for acc in self.accs:
			out += '  bottom: "' + acc + '"\n'

		out += ('  top: "loss"\n' \
				'  accumulator_loss_param {\n' \
				'    radius: %d\n'%(self.radius) + \
				'    downsampling: %d\n'%(self.min_acc_downsampling) + \
				'    negative_ratio: 4\n' \
				'  }\n' \
				'}\n')

		return out


	def _layer_input(self):
		"""
		Description of the input layer - for deployment.
		"""
		return ('layer {\n' \
				'  name: "data"\n' \
				'  type: "Input"\n' \
				'  top: "data"\n' \
				'  input_param { shape: { dim: 1 dim: 3 dim: 128 dim: 256 } }\n' \
				'}\n')


	def _layer_data(self, phase):
		"""
		Description of the data layer for train_val.

		Input:
			phase: string 'TRAIN' or 'TEST'
		"""
		out  = ('layer {\n' \
				'  name: "data"\n' \
				'  type: "BBTXTData"\n' \
				'  top: "data"\n' \
				'  top: "label"\n' \
				'  include {\n' \
				'    phase: ' + phase + '\n' \
				'  }\n' \
				'  image_data_param {\n' \
				'    source: ""\n' \
				'    batch_size: 32\n' \
				'    new_height: 128\n' \
				'    new_width: 256\n' \
				'  }\n' \
				'  transform_param {\n' \
				'    scale: 0.0078125\n' \
				'    mean_value: 128\n' \
				'  }\n' \
				'}\n')

		return out



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
	parser = argparse.ArgumentParser(description='Generate train_val and deploy PROTOTXT files ' \
												 'of Caffe networks with multiscale accumulators.')

	parser.add_argument('path_config', metavar='path_config', type=str,
	                    help='A configuration TXT file with network structure')
	parser.add_argument('path_out', metavar='path_out', type=str,
	                    help='Path to the output folder')

	args = parser.parse_args()

	if not check_path(args.path_config):
		parser.print_help()
		exit(1)

	return args


def main():
	args = parse_arguments()
	
	ng = MACCNetGenerator(args.path_config)
	ng.generate_prototxt_files(args.path_out)


if __name__ == '__main__':
    main()


