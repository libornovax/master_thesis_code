"""
Functions for loading the data mappings from numbers labels to categories.

INFORMATION ABOUT THE MAPPINGS
------------------------------
The shared values in the mappings are the text categories. In each dataset we have a different set
of numbers (labels of the classes in the dataset), but we need to map them to a common
representation - in this case the text categories.
"""

__date__   = '12/01/2016'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'

import os
import yaml


####################################################################################################
#                                            FUNCTIONS                                             # 
####################################################################################################

def load_mapping(path_mapping):
	"""
	Loads a mapping from a YAML file.

	Input:
		path_mapping: Path to a YAML file with label:category mapping
	Returns:
		dictionary with 'name' and 'mappings'
	"""
	with open(path_mapping, 'r') as infile:
		try:
			mapping = yaml.load(infile)
		except yaml.YAMLError as exc:
			print(exc)
			return None

		# Check if the mapping was loaded correctly
		if 'name' not in mapping or 'mappings' not in mapping:
			print('ERROR: Missing keys in the given mapping file!')
			exit(1)
		for k in mapping['mappings'].keys():
			try:
				# Keys must be integers
				int(k)
			except ValueError as exc:
				print(exc)
				exit(1)
		return mapping

	return None


def available_categories(mapping):
	"""
	Returns a list of available categories in the given mapping.

	Input:
		mapping: Dictionary of mappings label:category
	Returns:
		list of string categories, which are present in this mapping
	"""
	return list(set(mapping.values()))



####################################################################################################
#                                             CLASSES                                              # 
####################################################################################################

class LabelMappingManager(object):
	"""
	Provides interface to all mappings in this folder. Automatically loads all mappings in the YAML
	files in this folder and then it can be queried for different mappings.
	"""
	def __init__(self):
		super(LabelMappingManager, self).__init__()

		# Get all YAML files from this directory - those are current active mappings
		mappings_folder = os.path.dirname(os.path.realpath(__file__))
		mapping_files = [f for f in os.listdir(mappings_folder) if os.path.splitext(f)[1] == '.yaml']
		
		self.load_mappings(mappings_folder, mapping_files)


	def load_mappings(self, mappings_folder, mapping_files):
		"""
		Loads all the mapping files into a dictionary of mappings.

		Input:
			mappings_folder: Path to folder with the mapping_files
			mapping_files:   List of mapping YAML filenames to be loaded
		"""
		self.mappings = {}

		for mf in mapping_files:
			mapping = load_mapping(os.path.join(mappings_folder, mf))
			self.mappings[mapping['name']] = mapping['mappings']


	def get_mapping(self, name):
		"""
		Returns the requested mapping - dictionary of translations from numbers to text labels.

		Input:
			name: Name of the mapping (string)
		Returns:
			dictionary
		"""
		if name not in self.mappings:
			print('ERROR: Requested mapping "%s" does not exist!'%(name))
			print('Available mappings are: ' + str(self.mappings.keys()))
			exit(1)

		return self.mappings[name]


	def available_mappings(self):
		"""
		Returns a list of available mappings.

		Returns:
			list of strings
		"""
		return self.mappings.keys()



