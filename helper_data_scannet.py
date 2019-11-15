from helper_data_plot import Plot as Plot
import os
import numpy as np


### nyu40 class
CLASS_LABELS = ['wall','floor','cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds',
                'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refrigerator',
                'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'nightstand', 'toilet', 'sink', 'lamp',
                'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']
### nyu40 id
CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,18,19,20,21,22,23, 24, 25,26,27, 28,
                      29, 30,31,32, 33, 34, 35, 36, 37,38, 39, 40])


results_scannet_validation = 'results_scannet_validation/'
scene_names = sorted(os.listdir(results_scannet_validation))
if len(scene_names)<=0:
	print('please download sample results first.')
	# https://drive.google.com/file/d/1cV07rP02Yi3Eu6GQxMR2buigNPJEvCq0/view?usp=sharing
	exit()

for scene in scene_names:
	print('scene:', scene)
	pc = np.loadtxt(results_scannet_validation+scene+'/'+scene+'_pc_xyzrgb.txt', np.float32)
	sem_pred = np.loadtxt(results_scannet_validation+scene+'/'+scene+'_sem_pred.txt', np.int16)
	sem_gt = np.loadtxt(results_scannet_validation+scene+'/'+scene +'_sem_gt.txt', np.int16)
	ins_pred = np.loadtxt(results_scannet_validation + scene +'/'+scene +'_ins_pred.txt', np.int16)
	ins_gt = np.loadtxt(results_scannet_validation + scene +'/'+scene +'_ins_gt.txt', np.int16)

	## plot
	Plot.draw_pc(pc_xyzrgb=pc[:, 0:6])
	Plot.draw_pc_semins(pc_xyz=pc[:, 0:3], pc_semins=sem_pred)
	Plot.draw_pc_semins(pc_xyz=pc[:, 0:3], pc_semins=sem_gt)
	Plot.draw_pc_semins(pc_xyz=pc[:, 0:3], pc_semins=ins_pred)
	Plot.draw_pc_semins(pc_xyz=pc[:, 0:3], pc_semins=ins_gt)
