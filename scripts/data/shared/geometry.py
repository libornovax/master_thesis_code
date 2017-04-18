"""
Geometric transformation functions and matrices.

Contains functions for creating translation and rotation matrices and others.
"""

__date__   = '03/15/2016'
__author__ = 'Libor Novak'
__email__  = 'novakli2@fel.cvut.cz'

import numpy as np


####################################################################################################
#                                            ROTATIONS                                             # 
####################################################################################################

def R3x3_x(alpha):
	"""
	Rotation matrix around x axis in 3D.
	"""
	R = np.asmatrix([[1.0,  0.0,             0.0], 
		             [0.0,  np.cos(alpha),   -np.sin(alpha)], 
		             [0.0,  np.sin(alpha),   np.cos(alpha)]])
	return R


def R3x3_y(beta):
	"""
	Rotation matrix around y axis in 3D.
	"""
	R = np.asmatrix([[np.cos(beta),   0.0,   np.sin(beta)], 
		             [0.0,            1.0,   0.0], 
		             [-np.sin(beta),  0.0,   np.cos(beta)]])
	return R


def R3x3_z(gamma):
	"""
	Rotation matrix around z axis in 3D.
	"""
	R = np.asmatrix([[np.cos(gamma),   -np.sin(gamma),   0.0], 
		             [np.sin(gamma),   np.cos(gamma),    0.0], 
		             [0.0,             0.0,              1.0]])
	return R



####################################################################################################
#                                          TRANSLATIONS                                            # 
####################################################################################################

def t3x1(x, y, z):
	"""
	Translation vector.
	"""
	t = np.asmatrix([[x], 
		             [y], 
		             [z]])
	return t


def t3x1_x(x):
	"""
	Translation in x.
	"""
	return t3x1(x, 0.0, 0.0)


def t3x1_y(y):
	"""
	Translation in y.
	"""
	return t3x1(0.0, y, 0.0)


def t3x1_z(z):
	"""
	Translation in z.
	"""
	return t3x1(0.0, 0.0, z)


####################################################################################################
#                                          COMBINATIONS                                            # 
####################################################################################################

def Rt4x4(R, t):
	"""
	Combines rotation and translation to a single 4x4 matrix.
	"""
	Rt = np.asmatrix(np.eye(4))
	Rt[:3,:3] = R
	Rt[:3,3]  = t

	return Rt


####################################################################################################
#                                        IMAGE PROJECTIONS                                         # 
####################################################################################################

def reconstruct_X_in_plane(u, v, KR_3x3_inv, C_3x1, p_1x4):
	"""
	Reconstructs a point in 3D, which lies on the plane p_1x4 and projects to (u,v) in the image.

	Input:
		u, v:       Point coordinates in the image
		KR_3x3_inv: Inverse of the KR matrix from the P matrix (P = KR[I|-C])
		C_3x1:      Position of the camera in the 3D world (from P = KR[I|-C])
		p_1x4:      np.matrix coefficients of ax+by+cz+d=0 plane equation
	Returns:
		X_3x1 point coordinates in the 3D world
	"""
	x_3x1 = np.asmatrix([[u], [v], [1.0]])
	X_d_3x1 = KR_3x3_inv * x_3x1  # Direction of X from the camera center

	# Intersect the plane p_1x4 - find lm from the X = C + lm*X_d equation
	lm = - (p_1x4[0,0:3]*C_3x1 + p_1x4[0,3]) / (p_1x4[0,0:3]*X_d_3x1)
	X_3x1 = C_3x1 + lm[0,0] * X_d_3x1

	return X_3x1


def project_X_to_x(X_3xn, P_3x4):
	"""
	Projects the point(s) in the 3D world to image coordinates.

	Input:
		X_3xn: np.matrix of point coordinates in 3D, each column is one point
		P_3x4: Image projection matrix
	Returns:
		x_2xn: Coordinates of the points' projections in the image
	"""
	X_4xn = np.asmatrix(np.ones((4, X_3xn.shape[1])))
	X_4xn[0:3,:] = X_3xn
	x_3xn = P_3x4 * X_4xn;
	x_2xn = x_3xn[0:2,:] / x_3xn[2,:];

	return x_2xn

