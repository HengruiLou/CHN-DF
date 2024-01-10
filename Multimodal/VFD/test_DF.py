from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def auc(real, fake):
    label_all = []
    target_all = []
    for ind in real:
        target_all.append(1)
        label_all.append(-ind)
    for ind in fake:
        target_all.append(0)
        label_all.append(-ind)

    #from sklearn.metrics import roc_auc_score
    #return roc_auc_score(target_all, label_all)
    return target_all, label_all



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 4
    opt.batch_size = 1
    opt.serial_batches = False
    opt.no_flip = True
    opt.display_id = -1
    opt.mode = 'test'
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    model.eval()

    dataset_size = len(dataset)
    print('The number of test images dir = %d' % dataset_size)

    total_iters = 0
    label = None
    real = []
    fake = []

    with tqdm(total=dataset_size) as pbar:
        for i, data in enumerate(dataset):
            input_data = {'img_real': data['img_real'],
                          'img_fake': data['img_fake'],
                          'aud_real': data['aud_real'],
                          'aud_fake': data['aud_fake'],
                          }
            model.set_input(input_data)

            dist_AV, dist_VA = model.val()
            real.append(dist_AV.item())
            for i in dist_VA:
                fake.append(i.item())
            total_iters += 1
            pbar.update()
    y_tar, y_pred=auc(real, fake)
    print("y_tar",y_tar)
    print("y_pred",y_pred)
    print('The auc is %.3f'%(roc_auc_score(y_tar, y_pred)))
    y_pred = (np.array(y_pred) > 0.5).astype(int)
    print("y_pred_binary",y_pred)
	# 计算 Precision
    precision = precision_score(y_tar, y_pred)
	# 计算 Recall
    recall = recall_score(y_tar, y_pred)
	# 计算 F1-score
    f1 = f1_score(y_tar, y_pred)
	# 计算 Accuracy
    accuracy = accuracy_score(y_tar, y_pred)
	# 打印结果
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1-score: ' + str(f1))
    print('Accuracy: ' + str(accuracy))