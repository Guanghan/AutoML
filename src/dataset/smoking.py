"""
@author: Guanghan Ning
@file: smoking.py
@time: 12/7/20 11:31 下午
@file_desc: Smoking dataset
"""
import numpy as np
import math, os, cv2

from torchvision.datasets import ImageFolder

def validate_file_format(file_in_path, allowed_format):
    if os.path.isfile(file_in_path) and os.path.splitext(file_in_path)[1][1:] in allowed_format:
        return True
    else:
        return False

def is_image(file_in_path):
    if validate_file_format(file_in_path, ['jpg', 'JPEG', 'png', 'JPG']):
        return True
    else:
        return False

def isHtml(file):
    with open(file, 'rb') as f:
        head = f.read(6)
    return head == b'<html>'

def isValid(file):
    if not isHtml(file) and is_image(file):
        return True
    else:
        return False


class Smoking(ImageFolder):
    """
    self.imgs is a tuple of (img_path, cls_id), e.g.,
    [(‘data/dogcat_2/cat/cat.12484.jpg’, 0), (‘data/dogcat_2/cat/cat.12485.jpg’, 0),
    (‘data/dogcat_2/cat/cat.12486.jpg’, 0), (‘data/dogcat_2/cat/cat.12487.jpg’, 0),
    (‘data/dogcat_2/dog/dog.12496.jpg’, 1), (‘data/dogcat_2/dog/dog.12497.jpg’, 1),
    (‘data/dogcat_2/dog/dog.12498.jpg’, 1), (‘data/dogcat_2/dog/dog.12499.jpg’, 1)]
    """
    def __init__(self, root, split, transform=None, target_transform=None):
        if split == "val":
            #split = "val_only_positive"
            #split = "val_only_negative"
            split = "val"
        self.data_dir = os.path.join(root, split)
        print(self.data_dir)
        self.save_dir = os.path.join(root, "save")
        super(Smoking, self).__init__(root=self.data_dir,
                                             transform=transform,
                                             target_transform=target_transform)
        self.class_name = self.class_to_idx.keys()
        self.num_classes = len(self.class_name)
        self.pos_classes = ['smoking']
        self.neg_classes = ['negative']
        self.pos_ids = [self.class_to_idx[pos_class] for pos_class in self.pos_classes]
        self.neg_ids = [self.class_to_idx[neg_class] for neg_class in self.neg_classes]

        print("self.num_classes", self.num_classes)
        print("self.pos_ids", self.pos_ids)
        print("self.neg_ids", self.neg_ids)
        print("Class-ID mapping: {}".format(self.class_to_idx))


    def save_results(self, results, GTs, save_dir):
        import json
        json.dump(results, open('{}/results.json'.format(save_dir), 'w'))
        json.dump(GTs, open('{}/GTs.json'.format(save_dir), 'w'))


    def run_eval(self, preds, GTs, save_dir):
        #self.save_results(preds, GTs, save_dir)
        #print(GTs)
        #print(preds)
        assert(len(GTs) == len(preds))
        num_samples = len(GTs)
        correct_num_top1 = 0
        correct_num_topK = 0

        for sample_id in range(num_samples):
            pred = preds[sample_id]
            pred_top1 = pred[0]
            gt = GTs[sample_id]

            if pred_top1 == gt:
                correct_num_top1 += 1

            if gt in pred:
                correct_num_topK += 1

        top1_accuracy = correct_num_top1 * 1.0 / num_samples
        print("Correct predictions: {}, Total samples: {}".format(correct_num_top1, num_samples))
        print("Top1 Accuracy: {}".format(top1_accuracy))

        topK_accuracy = correct_num_topK * 1.0 / num_samples
        print("TopK Accuracy: {}".format(topK_accuracy))


    def calculate_precision_recall(self, preds, GTs, topK=None):
        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

        assert(len(GTs) == len(preds))
        num_samples = len(GTs)

        for sample_id in range(num_samples):
            assert(topK <= len(preds[sample_id]))
            pred = preds[sample_id][0:topK]
            gt = GTs[sample_id]

            ''' Determine ground truth'''
            censor_gt = False
            if gt in self.pos_ids:
                censor_gt = True

            ''' Determine prediction '''
            censor_pred = False
            if topK:
                for each_pred in pred:
                    if each_pred in self.pos_ids:
                       censor_pred = True
            else:
                if pred[0] in self.pos_ids:
                    censor_pred = True

            ''' Compare GT and Pred '''
            if censor_gt and censor_pred:
                true_pos += 1
            elif censor_gt and not censor_pred:
                false_neg += 1
            elif not censor_gt and not censor_pred:
                true_neg += 1
            elif not censor_gt and censor_pred:
                false_pos += 1

        ''' metrics '''
        precision = 1.0 * true_pos / (true_pos + false_pos) if true_pos + false_pos != 0 else 0
        recall = 1.0 * true_pos / (true_pos + false_neg) if true_pos + false_neg != 0 else 0
        censor_rate = 1.0 * (true_pos + false_pos) / num_samples

        true_pos_rate = recall
        false_pos_rate = 1.0 * false_pos / (false_pos + true_neg) if false_pos + true_neg != 0 else 0 # fallout

        if topK:
            print("Top{} evaluation:".format(topK))
        else:
            print("Top1 evaluation:")
        print("TP: {}, FP: {}, TN: {}, FN: {}".format(true_pos, false_pos, true_neg, false_neg))
        print("Precision: {}".format(precision))
        print("Recall / sensitivity: {}".format(recall))
        print("Censor rate: {}".format(censor_rate))
        print("True positive rate = recall")
        print("False positive rate: {}".format(false_pos_rate))
        return precision, recall, censor_rate, true_pos_rate, false_pos_rate


    def run_censor_eval(self, preds, GTs):
        prec_list = []
        recall_list = []
        true_pos_rate_list = []
        false_pos_rate_list = []

        for k in range(1, self.num_classes):
            print("Working on Top{}:".format(k))
            precision, recall, censor_rate, true_pos_rate, false_pos_rate = self.calculate_precision_recall(
                preds, GTs, k)
            prec_list.append(precision)
            recall_list.append(recall)
            true_pos_rate_list.append(true_pos_rate)
            false_pos_rate_list.append(false_pos_rate)

        # draw precision-recall curve and calculate auc
        print("precision list: {}".format(prec_list))
        print("recall list: {}".format(recall_list))
        fig_save_path = os.path.join(self.save_dir, "prec-recall-curve.png")
        self.plot_precision_recall_curve(fig_save_path, prec_list, recall_list)
        print("precision-recall-curve saved to {}".format(fig_save_path))

        # draw ROC-curve
        print("true_pos_rate_list: {}".format(true_pos_rate_list))
        print("false_pos_rate_list: {}".format(false_pos_rate_list))
        fig_save_path = os.path.join(self.save_dir, "ROC-curve.png")
        self.plot_roc_curve(fig_save_path, true_pos_rate_list, false_pos_rate_list)
        print("ROC-curve saved to {}".format(fig_save_path))
        return


    def plot_precision_recall_curve(self, fig_save_path, prec_list, recall_list):
        import matplotlib.pyplot as plt
        from sklearn import metrics
        auc_score = metrics.auc(recall_list, prec_list)
        plt.clf()
        plt.plot(recall_list, prec_list)
        plt.text(0.5, 0.5, 'AUC(%s)'%(auc_score), fontsize=12)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall')
        plt.legend(loc="upper right")
        plt.savefig(fig_save_path)


    def plot_roc_curve(self, fig_save_path, true_pos_rate_list, false_pos_rate_list):
        import matplotlib.pyplot as plt
        from sklearn import metrics
        auc_score = metrics.auc(false_pos_rate_list, true_pos_rate_list)
        plt.clf()
        plt.plot(false_pos_rate_list, true_pos_rate_list)
        plt.text(0.5, 0.5, 'AUC(%s)'%(auc_score), fontsize=12)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('ROC curve')
        plt.legend(loc="upper right")
        plt.savefig(fig_save_path)

