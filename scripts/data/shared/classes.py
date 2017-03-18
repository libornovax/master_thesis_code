"""
Useful classes and functions that can be shared throughout the scripts.
"""

__date__   = '12/02/2016'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'


####################################################################################################
#                                             CLASSES                                              # 
####################################################################################################

class BB2D(object):
	"""
	A 2D bounding box with a label and a confidence. Such a bounding box is meant to be read
	from BBTXT files.
	"""
	def __init__(self, xmin, ymin, xmax, ymax, label=None, confidence=None):
		super(BB2D, self).__init__()
		
		self.xmin = float(xmin)
		self.ymin = float(ymin)
		self.xmax = float(xmax)
		self.ymax = float(ymax)

		self.label 		= int(label) if (label is not None) else label
		self.confidence = float(confidence) if (confidence is not None) else confidence


	def area(self):
		"""
		Computes the area of the bounding box.

		Returns:
			float
		"""
		return float((self.xmax-self.xmin) * (self.ymax-self.ymin))


	def iou(self, other):
		"""
		Computes intersection over union with the other BB2D bounding box.

		Input:
			other: Instance of BB2D
		Output:
			float
		"""
		intersection_width  = max(0.0, min(self.xmax, other.xmax) - max(self.xmin, other.xmin))
		intersection_height = max(0.0, min(self.ymax, other.ymax) - max(self.ymin, other.ymin))
		intersection_area = float(intersection_width * intersection_height)

		return intersection_area / float(self.area()+other.area()-intersection_area)

	def width(self):
		"""
		Width of the bounding box.
		"""
		return float(self.xmax - self.xmin)


	def height(self):
		"""
		Height of the bounding box.
		"""
		return float(self.ymax - self.ymin)


	def __repr__(self):
		"""
		Text representation of this class.
		"""
		return 'BB2D: {[' + str(self.xmin) + ', ' + str(self.ymin) + ', ' + str(self.xmax) + ', ' \
			+ str(self.ymax) + '] label: ' + str(self.label) + ', confidence: ' \
			+ str(self.confidence) + '}'



class BB3D(object):
	"""
	A 3D bounding box with a label and a confidence. Such a bounding box is meant to be read
	from BB3TXT files.
	"""
	def __init__(self, fblx, fbly, fbrx, fbry, rblx, rbly, ftly, label=None, confidence=None):
		super(BB3D, self).__init__()
		
		self.fblx = float(fblx)
		self.fbly = float(fbly)
		self.fbrx = float(fbrx)
		self.fbry = float(fbry)
		self.rblx = float(rblx)
		self.rbly = float(rbly)
		self.ftly = float(ftly)

		self.label 		= int(label) if (label is not None) else label
		self.confidence = float(confidence) if (confidence is not None) else confidence


	def __repr__(self):
		"""
		Text representation of this class.
		"""
		return 'BB3D: {fbl: [' + str(self.fblx) + ',' + str(self.fbly) + '], fbr: [' \
			    + str(self.fbrx) + ',' + str(self.fbry) + '], rbl: [' + str(self.rblx) + ',' \
			    + str(self.rbly) + '], ftly: ' + str(self.ftly) + ', label: ' + str(self.label) \
			    + ', confidence: ' + str(self.confidence) + '}'


