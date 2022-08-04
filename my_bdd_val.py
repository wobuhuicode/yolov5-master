
# with open("val_list.txt", 'x') as file:
#     for i in range(10000):
#         file.write("/root/nfs/bdd-expr-on-board/bdd_val/images/" + str(i + 1) + ".jpg\n")  

import os
from re import X
import cv2
from regex import P
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
import numpy as np
import torch
from tqdm import tqdm
import prettytable as pt

from utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, emojis, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)

from utils.plots import output_to_target, plot_images, plot_val_study

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def run():
    # model_name = 'fbnet'
    # model_name = 'mobiledets'
    # model_name = 'ofa'
    # model_name = 'ours'

    model_names = ['fbnet', 'mobiledets', 'ofa', 'ours']
    precisions = []
    map50s = []

    # model_name = 'yolov6n_no_mosa'

    for i in range(len(model_names)):
        print("Validing model: ", model_names[i])

        if os.path.exists(f'bdd_val_out/{model_names[i]}_ref') is False:
            os.mkdir(f'bdd_val_out/{model_names[i]}_ref')
            os.mkdir(f'bdd_val_out/{model_names[i]}_ref/preds')


        nc = 1
        iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()

        out_dir = f'bdd_val_out/{model_names[i]}_ref'

        plots = True

        seen = 0
        confusion_matrix = ConfusionMatrix(nc=nc)

        names = {0: "car"}

        s = ('%20s' + '%11s' * 4) % ('Class', 'Images', 'P', 'R', 'mAP@.5')
        dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        loss = torch.zeros(3)
        jdict, stats, ap, ap_class = [], [], [], []

        # pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar


        all_inference_result = np.loadtxt(f"bdd_valid_output/{model_names[i]}_out.txt", dtype=np.float64, delimiter=',')

        for si in tqdm(range(1, 10001, 1)): # 遍历
            # 得到该图片的所有，label 信息
            pred = torch.from_numpy(all_inference_result[np.where(all_inference_result[:, 0] == si)][:, 1:])
            # labels = torch.from_numpy(np.loadtxt(f"/mnt/d/Littro_3519A/ubuntu/bdd-expr-on-board/bdd_val/labels/val/{si}.txt", dtype=np.float64, delimiter=' '))
            labels = torch.from_numpy(np.loadtxt(f"/mnt21t/home/wyh/wyh_project/bdd100k-yolov5/BDD100/yolo_dataset/labels/val/{si}.txt", dtype=np.float64, delimiter=' '))
            if labels.dim() == 1:
                if labels.size()[0] == 0:
                    labels = torch.reshape(labels, (0, 5))
                else :
                    labels = torch.reshape(labels, (1, 5))


            # 得到该图片的所有，pred 信息
            # labels 信息和 pred 信息的长度
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            # 图片的 路径，图片的形状
            # path = f"/mnt/d/Littro_3519A/ubuntu/bdd-expr-on-board/bdd_val/images/val/{si}.jpg"
            shape = [720, 1280]

            # 初始化 correct 张量，全为0，形状 npr * 10，表示 每个 pred 是否正确
            correct = torch.zeros(npr, niou, dtype=torch.bool)  # init
            seen += 1
            # 如果没有预测到任何值
            if npr == 0:
                # 如果 labels 有值
                if nl:
                    # 状态信息新增，correct 矩阵，后面的信息也全给 0
                    stats.append((correct, *torch.zeros((3, 0))))
                continue
            # Predictions
            # 复制 pred
            predn = pred.clone()
            predn[:, 2] = predn[:, 0] + predn[:, 2]
            predn[:, 3] = predn[:, 1] + predn[:, 3]

            scale_coords((640, 640), predn[:, :4], shape)  # native-space labels

            # scale_x = shape[1] / 640
            # scale_y = shape[0] / 640

            # predn[:, 0] = predn[:, 0] * scale_x
            # predn[:, 1] = predn[:, 1] * scale_y
            # predn[:, 2] = predn[:, 2] * scale_x
            # predn[:, 3] = predn[:, 3] * scale_y

            # Evaluate
            if nl:
                # 根据 pred 和 labels 的信息，计算出 correct 和 tp 和 fp
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                tbox[:, 0] = tbox[:, 0] * shape[1]
                tbox[:, 1] = tbox[:, 1] * shape[0]
                tbox[:, 2] = tbox[:, 2] * shape[1]
                tbox[:, 3] = tbox[:, 3] * shape[0]

                if si < 50:
                    img = cv2.imread(f"/mnt21t/home/wyh/wyh_project/bdd100k-yolov5/BDD100/yolo_dataset/images/val/{si}.jpg")
                    for i in range(len(tbox)):
                        cv2.rectangle(img, (int(tbox[i, 0]), int(tbox[i, 1])), (int(tbox[i, 2]), int(tbox[i, 3])), (0, 255, 0), 2)
                    for i in range(len(predn)):
                        cv2.rectangle(img, (int(predn[i, 0]), int(predn[i, 1])), (int(predn[i, 2]), int(predn[i, 3])), (0, 0, 255), 2)

                    cv2.imwrite(out_dir + f"/preds/pred_{si}.jpg", img)


                # 把 标签分类信息 和 像素 target bbox 信息，拼起来
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                # 计算 correct 矩阵
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
            # Save/log
            # if save_txt:
            #     save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            # if save_json:
            #     save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            # callbacks.run('on_val_image_end', pred, predn, path, names, im[si])
            # Plot images

            # Plot images
            # if plots and si < 3:
            #     plot_images(im, targets, paths, save_dir / 'val_batch{batch_i}_labels.jpg', names)  # labels
            #     plot_images(im, output_to_target(out), paths, save_dir / 'val_batch{batch_i}_pred.jpg', names)  # pred

        # Compute metrics
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            # Plot class APs, 计算每个 class 的 AP
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=out_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # print("\t\tclass      seen       nt          mp         mr       map50     map[0.5:0.95]")
        print(s)
        # Print results
        pf = '%20s' + '%11i' + '%11.3g' * 3  # print format
        LOGGER.info(pf % ('all', seen, mp, mr, map50))

        precisions.append(mp)
        map50s.append(map50)

        # Print results per class
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, p[i], r[i], ap50[i]))

        # Plots
        if plots:
            confusion_matrix.plot(save_dir=out_dir, names=list(names.values()))

    ## 按行添加数据
    tb = pt.PrettyTable()
    tb.field_names = ["Model", "P", "Map50"]
    for i in range(len(precisions)):
        tb.add_row([model_names[i], '%.3g' % precisions[i], '%.3g' % map50s[i]])
    
    print(tb)

if __name__ == '__main__':
    run()
