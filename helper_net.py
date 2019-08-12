import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment

class Ops:

    @staticmethod
    def lrelu(x, leak=0.2):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

    @staticmethod
    def relu(x):
        return tf.nn.relu(x)

    @staticmethod
    def xxlu(x,label,name=None):
        if label =='relu':
            return  Ops.relu(x)
        if label =='lrelu':
            return  Ops.lrelu(x,leak=0.2)

    @staticmethod
    def variable_sum(var, name):
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    @staticmethod
    def variable_count():
        total_para = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para += variable_para
        return total_para

    @staticmethod
    def fc(x, out_d, name):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        in_d = x.get_shape()[1]
        with tf.device('/cpu:0'):  # to create Variables stored on CPU memory
            w = tf.get_variable(name + '_w', [in_d, out_d], initializer=xavier_init)
            b = tf.get_variable(name + '_b', [out_d], initializer=zero_init)
        y = tf.nn.bias_add(tf.matmul(x, w), b)
        Ops.variable_sum(w, name)
        return y
	
    @staticmethod
    def conv2d(x, k=(1,1), out_c=1, str=1, name='',pad='SAME'):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        in_c = x.get_shape()[3]
        with tf.device('/cpu:0'):  # to create Variables stored on CPU memory
            w = tf.get_variable(name + '_w', [k[0], k[1], in_c, out_c], initializer=xavier_init)
            b = tf.get_variable(name + '_b', [out_c], initializer=zero_init)

        stride = [1, str, str, 1]
        y = tf.nn.bias_add(tf.nn.conv2d(x, w, stride, pad), b)
        Ops.variable_sum(w, name)
        return y

    @staticmethod
    def dropout(x, is_train, keep_prob, name):
        y = tf.cond(is_train, lambda: tf.nn.dropout(x, keep_prob=keep_prob, name=name), lambda: x)
        return y

    ####################################
    @staticmethod
    def gather_tensor_along_2nd_axis(bat_bb_pred, bat_bb_indices):
        bat_size = tf.shape(bat_bb_pred)[0]
        [_, ins_max_num, d1, d2] = bat_bb_pred.get_shape()
        bat_size_range = tf.range(bat_size)
        bat_size_range_flat = tf.reshape(bat_size_range, [-1,1])
        bat_size_range_flat_repeat = tf.tile(bat_size_range_flat, [1, int(ins_max_num)])
        bat_size_range_flat_repeat = tf.reshape(bat_size_range_flat_repeat, [-1])
        
        indices_2d_flat = tf.reshape(bat_bb_indices, [-1])
        indices_2d_flat_repeat = bat_size_range_flat_repeat*int(ins_max_num) + indices_2d_flat

        bat_bb_pred = tf.reshape(bat_bb_pred, [-1, int(d1), int(d2)])
        bat_bb_pred_new = tf.gather(bat_bb_pred, indices_2d_flat_repeat)
        bat_bb_pred_new = tf.reshape(bat_bb_pred_new, [bat_size, int(ins_max_num), int(d1), int(d2)])
   
        return bat_bb_pred_new

    @staticmethod
    def hungarian(loss_matrix, bb_gt):
        box_mask = np.array([[0, 0, 0], [0, 0, 0]])

        def assign_mappings_valid_only(cost, gt_boxes):
            # return ordering : batch_size x num_instances
            loss_total = 0.
            batch_size, num_instances = cost.shape[:2]
            ordering = np.zeros(shape=[batch_size, num_instances]).astype(np.int32)
            for idx in range(batch_size):
                ins_gt_boxes = gt_boxes[idx]
                ins_count = 0
                for box in ins_gt_boxes:
                    if np.array_equal(box, box_mask):
                        break
                    else:
                        ins_count += 1
                valid_cost = cost[idx][:ins_count]
                row_ind, col_ind = linear_sum_assignment(valid_cost)
                unmapped = num_instances - ins_count
                if unmapped > 0:
                    rest = np.array(range(ins_count, num_instances))
                    row_ind = np.concatenate([row_ind, rest])
                    unmapped_ind = np.array(list(set(range(num_instances)) - set(col_ind)))
                    col_ind = np.concatenate([col_ind, unmapped_ind])

                loss_total += cost[idx][row_ind, col_ind].sum()
                ordering[idx] = np.reshape(col_ind, [1, -1])
            return ordering, (loss_total / float(batch_size * num_instances)).astype(np.float32)
        ######
        ordering, loss_total = tf.py_func(assign_mappings_valid_only, [loss_matrix, bb_gt], [tf.int32, tf.float32])

        return ordering, loss_total

    @staticmethod
    def bbvert_association(X_pc, y_bbvert_pred, Y_bbvert, label=''):
        points_num = tf.shape(X_pc)[1]
        bbnum = int(y_bbvert_pred.shape[1])
        points_xyz = X_pc[:, :, 0:3]
        points_xyz = tf.tile(points_xyz[:, None, :, :], [1, bbnum, 1, 1])

        ##### get points hard mask in each gt bbox
        gt_bbox_min_xyz = Y_bbvert[:, :, 0, :]
        gt_bbox_max_xyz = Y_bbvert[:, :, 1, :]
        gt_bbox_min_xyz = tf.tile(gt_bbox_min_xyz[:, :, None, :], [1, 1, points_num, 1])
        gt_bbox_max_xyz = tf.tile(gt_bbox_max_xyz[:, :, None, :], [1, 1, points_num, 1])
        tp1_gt = gt_bbox_min_xyz - points_xyz
        tp2_gt = points_xyz - gt_bbox_max_xyz
        tp_gt = tp1_gt * tp2_gt
        points_in_gt_bbox_prob = tf.cast(tf.equal(tf.reduce_mean(tf.cast(tf.greater_equal(tp_gt, 0.), tf.float32), axis=-1), 1.0), tf.float32)

        ##### get points soft mask in each pred bbox ---> Algorithm 1
        pred_bbox_min_xyz = y_bbvert_pred[:, :, 0, :]
        pred_bbox_max_xyz = y_bbvert_pred[:, :, 1, :]
        pred_bbox_min_xyz = tf.tile(pred_bbox_min_xyz[:, :, None, :], [1, 1, points_num, 1])
        pred_bbox_max_xyz = tf.tile(pred_bbox_max_xyz[:, :, None, :], [1, 1, points_num, 1])
        tp1_pred = pred_bbox_min_xyz - points_xyz
        tp2_pred = points_xyz - pred_bbox_max_xyz
        tp_pred = 100 * tp1_pred * tp2_pred
        tp_pred = tf.maximum(tf.minimum(tp_pred, 20.0), -20.0)
        points_in_pred_bbox_prob = 1.0/(1.0 + tf.exp(-1.0 * tp_pred))
        points_in_pred_bbox_prob = tf.reduce_min(points_in_pred_bbox_prob, axis=-1)

        ##### get bbox cross entropy scores
        prob_gt = tf.tile(points_in_gt_bbox_prob[:, :, None, :], [1, 1, bbnum, 1])
        prob_pred = tf.tile(points_in_pred_bbox_prob[:, None, :, :], [1, bbnum, 1, 1])
        ce_scores_matrix = - prob_gt * tf.log(prob_pred + 1e-8) - (1 - prob_gt) * tf.log(1 - prob_pred + 1e-8)
        ce_scores_matrix = tf.reduce_mean(ce_scores_matrix, axis=-1)

        ##### get bbox soft IOU
        TP = tf.reduce_sum(prob_gt * prob_pred, axis=-1)
        FP = tf.reduce_sum(prob_pred, axis=-1) - TP
        FN = tf.reduce_sum(prob_gt, axis=-1) - TP
        iou_scores_matrix = TP/ (TP + FP + FN + 1e-6)
        # iou_scores_matrix = 1.0/iou_scores_matrix  # bad, don't use
        iou_scores_matrix = -1.0 * iou_scores_matrix  # to minimize

        ##### get bbox l2 scores
        l2_gt = tf.tile(Y_bbvert[:, :, None, :, :], [1, 1, bbnum, 1, 1])
        l2_pred = tf.tile(y_bbvert_pred[:, None, :, :, :], [1, bbnum, 1, 1, 1])
        l2_gt = tf.reshape(l2_gt, [-1, bbnum, bbnum, 2 * 3])
        l2_pred = tf.reshape(l2_pred, [-1, bbnum, bbnum, 2 * 3])
        l2_scores_matrix = tf.reduce_mean((l2_gt - l2_pred) ** 2, reduction_indices=[-1])

        ##### bbox association
        if label == 'use_all_ce_l2_iou':
            associate_maxtrix = ce_scores_matrix + l2_scores_matrix + iou_scores_matrix
        elif label == 'use_both_ce_l2':
            associate_maxtrix = ce_scores_matrix + l2_scores_matrix
        elif label == 'use_both_ce_iou':
            associate_maxtrix = ce_scores_matrix + iou_scores_matrix
        elif label == 'use_both_l2_iou':
            associate_maxtrix = l2_scores_matrix + iou_scores_matrix
        elif label == 'use_only_ce':
            associate_maxtrix = ce_scores_matrix
        elif label == 'use_only_l2':
            associate_maxtrix = l2_scores_matrix
        elif label == 'use_only_iou':
            associate_maxtrix = iou_scores_matrix
        else:
            associate_maxtrix=None
            print('association label error!'); exit()

        ######
        pred_bborder, association_score_min = Ops.hungarian(associate_maxtrix, bb_gt=Y_bbvert)
        pred_bborder = tf.cast(pred_bborder, dtype=tf.int32)
        y_bbvert_pred_new = Ops.gather_tensor_along_2nd_axis(y_bbvert_pred, pred_bborder)

        return y_bbvert_pred_new, pred_bborder

    @staticmethod
    def bbscore_association(y_bbscore_pred_raw, pred_bborder):
        y_bbscore_pred_raw = y_bbscore_pred_raw[:,:,None,None]
        y_bbscore_pred_new = Ops.gather_tensor_along_2nd_axis(y_bbscore_pred_raw, pred_bborder)

        y_bbscore_pred_new = tf.reshape(y_bbscore_pred_new, [-1, int(y_bbscore_pred_new.shape[1])])
        return y_bbscore_pred_new

    ####################################  sem loss
    @staticmethod
    def get_loss_psem_ce(y_psem_logits, Y_psem):
        psemce_loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_psem_logits, labels=Y_psem)
        psemce_loss = tf.reduce_mean(psemce_loss)
        return psemce_loss

    ####################################  bbox loss
    @staticmethod
    def get_loss_bbvert(X_pc, y_bbvert_pred, Y_bbvert, label=''):
        points_num = tf.shape(X_pc)[1]
        bb_num = int(Y_bbvert.shape[1])
        points_xyz = X_pc[:, :, 0:3]
        points_xyz = tf.tile(points_xyz[:, None, :, :], [1, bb_num, 1, 1])

        ##### get points hard mask in each gt bbox
        gt_bbox_min_xyz = Y_bbvert[:, :, 0, :]
        gt_bbox_max_xyz = Y_bbvert[:, :, 1, :]
        gt_bbox_min_xyz = tf.tile(gt_bbox_min_xyz[:, :, None, :], [1, 1, points_num, 1])
        gt_bbox_max_xyz = tf.tile(gt_bbox_max_xyz[:, :, None, :], [1, 1, points_num, 1])
        tp1_gt = gt_bbox_min_xyz - points_xyz
        tp2_gt = points_xyz - gt_bbox_max_xyz
        tp_gt = tp1_gt * tp2_gt
        points_in_gt_bbox_prob = tf.cast(tf.equal(tf.reduce_mean(tf.cast(tf.greater_equal(tp_gt, 0.), tf.float32), axis=-1), 1.0), tf.float32)

        ##### get points soft mask in each pred bbox
        pred_bbox_min_xyz = y_bbvert_pred[:, :, 0, :]
        pred_bbox_max_xyz = y_bbvert_pred[:, :, 1, :]
        pred_bbox_min_xyz = tf.tile(pred_bbox_min_xyz[:, :, None, :], [1, 1, points_num, 1])
        pred_bbox_max_xyz = tf.tile(pred_bbox_max_xyz[:, :, None, :], [1, 1, points_num, 1])
        tp1_pred = pred_bbox_min_xyz - points_xyz
        tp2_pred = points_xyz - pred_bbox_max_xyz
        tp_pred = 100*tp1_pred*tp2_pred
        tp_pred = tf.maximum(tf.minimum(tp_pred, 20.0), -20.0)
        points_in_pred_bbox_prob = 1.0/(1.0 + tf.exp(-1.0 * tp_pred))
        points_in_pred_bbox_prob = tf.reduce_min(points_in_pred_bbox_prob, axis=-1)

        ##### helper -> the valid bbox (the gt boxes are zero-padded during data processing, pickup valid ones here)
        Y_bbox_helper = tf.reduce_sum(tf.reshape(Y_bbvert, [-1, bb_num, 6]), axis=-1)
        Y_bbox_helper = tf.cast(tf.greater(Y_bbox_helper, 0.), tf.float32)

        ##### 1. get ce loss of valid/positive bboxes, don't count the ce_loss of invalid/negative bboxes
        Y_bbox_helper_tp1 = tf.tile(Y_bbox_helper[:, :, None], [1, 1, points_num])
        bbox_loss_ce_all = -points_in_gt_bbox_prob * tf.log(points_in_pred_bbox_prob + 1e-8) \
                       -(1.-points_in_gt_bbox_prob)*tf.log(1.-points_in_pred_bbox_prob + 1e-8)
        bbox_loss_ce_pos = tf.reduce_sum(bbox_loss_ce_all*Y_bbox_helper_tp1)/tf.reduce_sum(Y_bbox_helper_tp1)
        bbox_loss_ce = bbox_loss_ce_pos

        ##### 2. get iou loss of valid/positive bboxes
        TP = tf.reduce_sum(points_in_pred_bbox_prob * points_in_gt_bbox_prob, axis=-1)
        FP = tf.reduce_sum(points_in_pred_bbox_prob, axis=-1) - TP
        FN = tf.reduce_sum(points_in_gt_bbox_prob, axis=-1) - TP
        bbox_loss_iou_all = TP/(TP + FP + FN + 1e-6)
        bbox_loss_iou_all = -1.0*bbox_loss_iou_all
        bbox_loss_iou_pos = tf.reduce_sum(bbox_loss_iou_all*Y_bbox_helper)/tf.reduce_sum(Y_bbox_helper)
        bbox_loss_iou = bbox_loss_iou_pos

        ##### 3. get l2 loss of both valid/positive bboxes
        bbox_loss_l2_all = (Y_bbvert - y_bbvert_pred)**2
        bbox_loss_l2_all = tf.reduce_mean(tf.reshape(bbox_loss_l2_all, [-1, bb_num, 6]), axis=-1)
        bbox_loss_l2_pos = tf.reduce_sum(bbox_loss_l2_all*Y_bbox_helper)/tf.reduce_sum(Y_bbox_helper)

        ## to minimize the 3D volumn of invalid/negative bboxes, it serves as a regularizer to penalize false pred bboxes
        ## it turns out to be quite helpful, but not discussed in the paper
        bbox_pred_neg = tf.tile((1.- Y_bbox_helper)[:,:,None,None], [1,1,2,3])*y_bbvert_pred
        bbox_loss_l2_neg = (bbox_pred_neg[:,:,0,:]-bbox_pred_neg[:,:,1,:])**2
        bbox_loss_l2_neg = tf.reduce_sum(bbox_loss_l2_neg)/(tf.reduce_sum(1.-Y_bbox_helper)+1e-8)

        bbox_loss_l2 = bbox_loss_l2_pos + bbox_loss_l2_neg

        #####
        if label == 'use_all_ce_l2_iou':
            bbox_loss = bbox_loss_ce + bbox_loss_l2 + bbox_loss_iou
        elif label == 'use_both_ce_l2':
            bbox_loss = bbox_loss_ce + bbox_loss_l2
        elif label == 'use_both_ce_iou':
            bbox_loss = bbox_loss_ce + bbox_loss_iou
        elif label == 'use_both_l2_iou':
            bbox_loss = bbox_loss_l2 + bbox_loss_iou
        elif label == 'use_only_ce':
            bbox_loss = bbox_loss_ce
        elif label == 'use_only_l2':
            bbox_loss = bbox_loss_l2
        elif label == 'use_only_iou':
            bbox_loss = bbox_loss_iou
        else:
            bbox_loss = None
            print('bbox loss label error!'); exit()

        return bbox_loss, bbox_loss_l2, bbox_loss_ce, bbox_loss_iou

    @staticmethod
    def get_loss_bbscore(y_bbscore_pred, Y_bbvert):
        bb_num = int(Y_bbvert.shape[1])

        ##### helper -> the valid bbox
        Y_bbox_helper = tf.reduce_sum(tf.reshape(Y_bbvert, [-1, bb_num, 6]), axis=-1)
        Y_bbox_helper = tf.cast(tf.greater(Y_bbox_helper, 0.), tf.float32)

        ##### bbox score loss
        bbox_loss_score = tf.reduce_mean(-Y_bbox_helper * tf.log(y_bbscore_pred + 1e-8)
                                         -(1. - Y_bbox_helper) * tf.log(1. - y_bbscore_pred + 1e-8))
        return bbox_loss_score

    ####################################  pmask loss
    @staticmethod
    def get_loss_pmask(X_pc, y_pmask_pred, Y_pmask):
        points_num = tf.shape(X_pc)[1]
        ##### valid ins
        Y_pmask_helper = tf.reduce_sum(Y_pmask, axis=-1)
        Y_pmask_helper = tf.cast(tf.greater(Y_pmask_helper, 0.), tf.float32)
        Y_pmask_helper = tf.tile(Y_pmask_helper[:, :, None], [1, 1, points_num])

        Y_pmask = Y_pmask * Y_pmask_helper
        y_pmask_pred = y_pmask_pred * Y_pmask_helper

        ##### focal loss
        alpha = 0.75
        gamma = 2
        pmask_loss_focal_all = -Y_pmask*alpha*((1.-y_pmask_pred)**gamma)*tf.log(y_pmask_pred+1e-8)\
                               -(1.-Y_pmask)*(1.-alpha)*(y_pmask_pred**gamma)*tf.log(1.-y_pmask_pred+1e-8)
        pmask_loss_focal = tf.reduce_sum(pmask_loss_focal_all*Y_pmask_helper)/tf.reduce_sum(Y_pmask_helper)

        ## the above "alpha" makes the loss to be small
        ## then use a constant, so it's numerically comparable with other losses (e.g., semantic loss, bbox loss)
        pmask_loss = 30*pmask_loss_focal

        return pmask_loss