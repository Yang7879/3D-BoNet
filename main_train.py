import os

def train(net, data):
	for ep in range(0, 51,1):
		l_rate = max(0.0005/(2**(ep//20)), 0.00001)

		data.shuffle_train_files(ep)
		total_train_batch_num = data.total_train_batch_num
		print('total train batch num:', total_train_batch_num)
		for i in range(total_train_batch_num):
			###### training
			bat_pc, _, _, bat_psem_onehot, bat_bbvert, bat_pmask = data.load_train_next_batch()
			_, ls_psemce, ls_bbvert_all, ls_bbvert_l2, ls_bbvert_ce, ls_bbvert_iou, ls_bbscore, ls_pmask = net.sess.run([
			net.optim, net.psemce_loss, net.bbvert_loss, net.bbvert_loss_l2, net.bbvert_loss_ce, net.bbvert_loss_iou,net.bbscore_loss, net.pmask_loss],
			feed_dict={net.X_pc:bat_pc[:, :, 0:9], net.Y_bbvert:bat_bbvert, net.Y_pmask:bat_pmask, net.Y_psem:bat_psem_onehot, net.lr:l_rate, net.is_train:True})

			if i%200==0:
				sum_train = net.sess.run(net.sum_merged,
				feed_dict={net.X_pc: bat_pc[:, :, 0:9], net.Y_bbvert: bat_bbvert, net.Y_pmask: bat_pmask, net.Y_psem: bat_psem_onehot, net.lr: l_rate, net.is_train: False})
				net.sum_writer_train.add_summary(sum_train, ep*total_train_batch_num + i)
			print ('ep', ep, 'i', i, 'psemce', ls_psemce, 'bbvert', ls_bbvert_all, 'l2', ls_bbvert_l2, 'ce', ls_bbvert_ce, 'siou', ls_bbvert_iou, 'bbscore', ls_bbscore, 'pmask', ls_pmask)

			###### random testing
			if i%200==0:
				bat_pc, _, _, bat_psem_onehot, bat_bbvert, bat_pmask = data.load_test_next_batch_random()
				ls_psemce, ls_bbvert_all, ls_bbvert_l2, ls_bbvert_ce, ls_bbvert_iou, ls_bbscore, ls_pmask, sum_test, pred_bborder = net.sess.run([
				net.psemce_loss, net.bbvert_loss, net.bbvert_loss_l2, net.bbvert_loss_ce, net.bbvert_loss_iou, net.bbscore_loss, net.pmask_loss, net.sum_merged, net.pred_bborder],
				feed_dict={net.X_pc:bat_pc[:, :, 0:9], net.Y_bbvert:bat_bbvert, net.Y_pmask:bat_pmask, net.Y_psem:bat_psem_onehot, net.is_train:False})
				net.sum_write_test.add_summary(sum_test, ep*total_train_batch_num+i)
				print('ep',ep,'i',i,'test psem', ls_psemce, 'bbvert', ls_bbvert_all, 'l2', ls_bbvert_l2, 'ce', ls_bbvert_ce, 'siou', ls_bbvert_iou, 'bbscore', ls_bbscore, 'pmask', ls_pmask)
				print('test pred bborder', pred_bborder)

			###### saving model
			if i==total_train_batch_num-1 or i==0:
				net.saver.save(net.sess, save_path=net.train_mod_dir + 'model.cptk')
				print ("ep", ep, " i", i, " model saved!")
			if ep % 5 == 0 and i == total_train_batch_num - 1:
				net.saver.save(net.sess, save_path=net.train_mod_dir + 'model' + str(ep).zfill(3) + '.cptk')

			###### full eval, if needed
			if ep%5==0 and i==total_train_batch_num-1:
				from main_eval import Evaluation
				result_path = './log/test_res/' + str(ep).zfill(3)+'_'+test_areas[0] + '/'
				Evaluation.ttest(net, data, result_path, test_batch_size=20)
				Evaluation.evaluation(dataset_path, train_areas, result_path)
				print('full eval finished!')


############
if __name__=='__main__':
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = '0'  ## specify the GPU to use

	from main_3D_BoNet import BoNet
	from helper_data_s3dis import Data_Configs as Data_Configs

	configs = Data_Configs()
	net = BoNet(configs = configs)
	net.creat_folders(name='log', re_train=False)
	net.build_graph()

	####
	from helper_data_s3dis import Data_S3DIS as Data
	train_areas =['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
	test_areas =['Area_5']

	dataset_path = './data_s3dis/'
	data = Data(dataset_path, train_areas, test_areas, train_batch_size=4)
	train(net, data)