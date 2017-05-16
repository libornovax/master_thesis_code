"""
Useful classes and functions that can be shared throughout the scripts.
"""

__date__   = '12/02/2016'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'

import numpy as np
import geometry


####################################################################################################
#                                             CLASSES                                              # 
####################################################################################################

class BB2D(object):
	"""
	A 2D bounding box with a label and a confidence. Such a bounding box is meant to be read
	from BBTXT files.
	"""
	def __init__(self, xmin, ymin, xmax, ymax, label=None, confidence=None, required=True):
		super(BB2D, self).__init__()
		
		self.xmin = float(xmin)
		self.ymin = float(ymin)
		self.xmax = float(xmax)
		self.ymax = float(ymax)

		self.label 		= abs(int(label)) if (label is not None) else label
		self.confidence = float(confidence) if (confidence is not None) else confidence
		self.required   = required


	def area(self):
		"""
		Computes the area of the bounding box.

		Returns:
			float
		"""
		return float((self.xmax-self.xmin) * (self.ymax-self.ymin))


	def intersection_area(self, other):
		"""
		Computes the area of intersection of itself with another BB2D bounding box.

		Input:
			other: Instance of BB2D
		Output:
			float
		"""
		intersection_width  = max(0.0, min(self.xmax, other.xmax) - max(self.xmin, other.xmin))
		intersection_height = max(0.0, min(self.ymax, other.ymax) - max(self.ymin, other.ymin))

		return float(intersection_width * intersection_height)


	def iou(self, other):
		"""
		Computes intersection over union with the other BB2D bounding box.

		Input:
			other: Instance of BB2D
		Output:
			float
		"""
		intersection_area = self.intersection_area(other)

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
			+ str(self.confidence) + (', required' if self.required else '') + '}'



class BB3D(object):
	"""
	A 3D bounding box with a label and a confidence. Such a bounding box is meant to be read
	from BB3TXT files.
	"""
	def __init__(self, xmin, ymin, xmax, ymax, fblx, fbly, fbrx, fbry, rblx, rbly, ftly, 
				 label=None, confidence=None):
		super(BB3D, self).__init__()
		
		# This is to store the bounding box in 2D
		self.bb2d = BB2D(xmin, ymin, xmax, ymax)
		# 3D bounding box
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



class PGP(object):
	"""
	A 3x4 image projection matrix P and coefficients of the ground plane equation ax+by+cz+d=0. It
	is to be read as one line from a PGP file.
	"""
	def __init__(self, p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, a, b, c, d):
		super(PGP, self).__init__()
		
		self.P_3x4  = np.asmatrix([[p00, p01, p02, p03], [p10, p11, p12, p13], [p20, p21, p22, p23]])
		self.gp_1x4 = np.asmatrix([[a, b, c, d]])

		# Compute the inverse matrix KR and the camera pose in the 3D world
		self.KR_3x3_inv = np.linalg.inv(self.P_3x4[0:3,0:3])
		self.C_3x1      = - self.KR_3x3_inv * self.P_3x4[0:3,3]


	def reconstruct_X_ground(self, u, v):
		"""
		Reconstructs a point given by image coordinates (u,v) on the ground plane in 3D.

		Input:
			u, v: Point coordinates in the image
		Returns:
			X_3x1 point coordinates in the 3D world
		"""
		return geometry.reconstruct_X_in_plane(u, v, self.KR_3x3_inv, self.C_3x1, self.gp_1x4)


	def project_X_to_x(self, X_3xn):
		"""
		Projects the point(s) in the 3D world to the image coordinates.

		Input:
			X_3xn: np.matrix of point coordinates in 3D, each column is one point
		Returns:
			x_2xn: Coordinates of the points' projections in the image
		"""
		return geometry.project_X_to_x(X_3xn, self.P_3x4)


	def reconstruct_bb3d(self, bb3d):
		"""
		Reconstructs the 3D world coordinates of all 3D bounding box corners.

		Input:
			bb3d: A BB3D class instance
		Returns:
			X_3x8 np.matrix of 3D world coordinates of the 8 bounding box corners in the following
			      order: FBL FBR RBR RBL FTL FTR RTR RTL
		"""
		# Reconstruct the corners, which lie in the ground plane
		FBL_3x1 = self.reconstruct_X_ground(bb3d.fblx, bb3d.fbly)
		FBR_3x1 = self.reconstruct_X_ground(bb3d.fbrx, bb3d.fbry)
		RBL_3x1 = self.reconstruct_X_ground(bb3d.rblx, bb3d.rbly)
		RBR_3x1 = FBR_3x1 + (RBL_3x1-FBL_3x1)

		# Top of the 3D bounding box - reconstruct FTL and then just move all the other points
		# We do this by intersecting the ray to FTL with the front side plane of the bounding
		# box - i.e. extract the front side plane equation and then use it to reconstruct the FTL
		n_F_3x1 = FBL_3x1 - RBL_3x1  # Normal vector of the front side
		d_F = - (n_F_3x1[0,0]*FBL_3x1[0,0] + n_F_3x1[1,0]*FBL_3x1[1,0] + n_F_3x1[2,0]*FBL_3x1[2,0])
		# Front plane
		fp_1x4 = np.asmatrix([n_F_3x1[0,0], n_F_3x1[1,0], n_F_3x1[2,0], d_F])

		FTL_3x1 = geometry.reconstruct_X_in_plane(bb3d.fblx, bb3d.ftly, self.KR_3x3_inv, 
												  self.C_3x1, fp_1x4)
		# Bottom to top side vector
		BT_3x1 = FTL_3x1 - FBL_3x1


		# # Fix the coordinates to a rectangular cuboid
		# CM_3x1 = (FBL_3x1 + RBR_3x1) / 2.0

		# # Diagonals
		# d1_3x1 = FBL_3x1 - CM_3x1
		# d2_3x1 = FBR_3x1 - CM_3x1

		# d1_l = np.linalg.norm(d1_3x1)
		# d2_l = np.linalg.norm(d2_3x1)

		# delta = abs(d1_l - d2_l) / 2.0

		# if d1_l > d2_l:
		# 	d1_new_3x1 = d1_3x1 * (1 - delta / d1_l)
		# 	d2_new_3x1 = d2_3x1 * (1 + delta / d2_l)
		# else:
		# 	d1_new_3x1 = d1_3x1 * (1 + delta / d1_l)
		# 	d2_new_3x1 = d2_3x1 * (1 - delta / d2_l)

		# FBL_3x1 = CM_3x1 + d1_new_3x1
		# FBR_3x1 = CM_3x1 + d2_new_3x1
		# RBL_3x1 = CM_3x1 - d2_new_3x1
		# RBR_3x1 = CM_3x1 - d1_new_3x1


		FTL_3x1 = FBL_3x1 + BT_3x1
		FTR_3x1 = FBR_3x1 + BT_3x1
		RTL_3x1 = RBL_3x1 + BT_3x1
		RTR_3x1 = RBR_3x1 + BT_3x1

		# Combine everything to the output matrix
		X_3x8 = np.asmatrix(np.zeros((3, 8)))
		X_3x8[:,0] = FBL_3x1
		X_3x8[:,1] = FBR_3x1
		X_3x8[:,2] = RBR_3x1
		X_3x8[:,3] = RBL_3x1
		X_3x8[:,4] = FTL_3x1
		X_3x8[:,5] = FTR_3x1
		X_3x8[:,6] = RTR_3x1
		X_3x8[:,7] = RTL_3x1

		return X_3x8

