import numpy as np
import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2


def constraint_acc(true_labels, pred_labels):
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    assert true_labels.shape == pred_labels.shape, "真实标签与预测标签的形状必须一致"
    correct = np.sum(true_labels == pred_labels)
    acc = correct / len(true_labels)
    return acc

if __name__ == "__main__":
    img_org_paths = glob.glob("D:\\code\\Diffusion-Models-pytorch-main\\Image\\0\\*.png")
    value_true_paths = glob.glob("D:\\code\\Integrated-Design-Diffusion-Model-main\\results_LDM-4\\result_paper_experiment-npy\\*.npz")
    img_sample_paths = glob.glob("D:\\code\\Integrated-Design-Diffusion-Model-main\\results_LDM-4\\result_paper_experiment\\*.png")
    n = len(value_true_paths)
    con_acc_all = 0
    for i in range(int(n)):
        print(i)
        # 读取
        data = np.load(value_true_paths[i])
        coords_scaled = data["array1"]
        img_org = cv2.imread(img_org_paths[i], cv2.IMREAD_GRAYSCALE)
        img_org = cv2.resize(img_org, (512, 512), interpolation=cv2.INTER_NEAREST)
        img_sample = cv2.imread(img_sample_paths[i], cv2.IMREAD_GRAYSCALE)
        true_values = img_org[coords_scaled[:, 0],coords_scaled[:, 1]]
        sample_values = img_sample[coords_scaled[:, 0],coords_scaled[:, 1]]
        con_acc = constraint_acc(true_values, sample_values)
        print("con_acc--",con_acc)
        con_acc_all = con_acc_all + con_acc
    con_acc_all = con_acc_all/n
    print("con_acc_all--",con_acc_all)




