import numpy as np
import os
import scipy.io
import copy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from open3d import linux as open3d  ## pip install open3d-python==0.3.0
import random
import colorsys

class Plot:

	@staticmethod
	def random_colors(N, bright=True, seed=0):
		brightness = 1.0 if bright else 0.7
		hsv = [( 0.15+ i/float(N), 1, brightness) for i in range(N)]
		colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
		random.seed(seed)
		random.shuffle(colors)
		return colors

	@staticmethod
	def draw_pc(pc_xyzrgb):
		pc = open3d.PointCloud()
		pc.points = open3d.Vector3dVector(pc_xyzrgb[:, 0:3])
		if pc_xyzrgb.shape[1]==3:
			open3d.draw_geometries([pc])
			return 0
		if np.max(pc_xyzrgb[:, 3:6])>20: ## 0-255
			pc.colors = open3d.Vector3dVector(pc_xyzrgb[:, 3:6]/255.)
		else:
			pc.colors = open3d.Vector3dVector(pc_xyzrgb[:, 3:6])
		open3d.draw_geometries([pc])
		return 0

	@staticmethod
	def draw_pc_semins(pc_xyz, pc_semins, fix_color_num=None):
		if fix_color_num is not None:
			ins_colors = Plot.random_colors(fix_color_num+1, seed=2)
		else:
			ins_colors = Plot.random_colors(len(np.unique(pc_semins))+1, seed=2)  # cls 14

		##############################
		semins_labels = np.unique(pc_semins)
		semins_bbox = []
		Y_colors = np.zeros((pc_semins.shape[0], 3))
		for id, semins in enumerate(semins_labels):
			valid_ind = np.argwhere(pc_semins == semins)[:, 0]
			if semins<=-1:
				tp=[0,0,0]
			else:
				if fix_color_num is not None:
					tp = ins_colors[semins]
				else:
					tp = ins_colors[id]

			Y_colors[valid_ind] = tp

			### bbox
			valid_xyz = pc_xyz[valid_ind]

			xmin = np.min(valid_xyz[:, 0]); xmax = np.max(valid_xyz[:, 0])
			ymin = np.min(valid_xyz[:, 1]); ymax = np.max(valid_xyz[:, 1])
			zmin = np.min(valid_xyz[:, 2]); zmax = np.max(valid_xyz[:, 2])
			semins_bbox.append([[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

		Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
		Plot.draw_pc(Y_semins)
		return Y_semins