import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import time
import yaml
import argparse
from tqdm import tqdm
from glob import glob
from sklearn import metrics

from models.recapture_classifier import RecaptureDetector

DETECT_CFG = yaml.load(open('./detect_cfg.yaml'), Loader=yaml.FullLoader)
model = RecaptureDetector(device_id=DETECT_CFG['device_id'],
                          weight_path=DETECT_CFG['models']['recapture_classification'])


def classify_video_liveness(vid_cap):
    start = time.time()
    total_frames = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    succ, frame = False, None
    while not succ:
        rand_frame_id = random.randint(0, total_frames)
        vid_cap.set(cv2.CAP_PROP_POS_FRAMES, rand_frame_id)
        succ, frame = vid_cap.read()

    assert succ

    preds = model.predict(frame)[0]
    label = np.argmax(preds)
    return label, preds, time.time() - start


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dir', help='Test directory', default=None)
    args = parser.parse_args()

    total = len(glob(args.test_dir + '/**/*.mp4', recursive=True))
    pbar = tqdm(glob(args.test_dir + '/**/*.mp4', recursive=True), total=total)

    correct = 0
    total_time = 0.

    y_true = []
    y_pred = []
    recap_probs = []

    for vid_path in pbar:
        vid_cap = cv2.VideoCapture(vid_path)
        path = os.path.dirname(vid_path)
        target_label = int(path.split('/')[-1])
        label, prediction, infer_time = classify_video_liveness(vid_cap)

        total_time += infer_time
        y_true.append(target_label)
        y_pred.append(label)
        recap_probs.append(prediction[1])

        correct += int(label == target_label)

    acc = correct * 100 / total
    prec = metrics.precision_score(y_true, y_pred, average='binary') * 100
    rec = metrics.recall_score(y_true, y_pred, average='binary') * 100
    f1 = metrics.f1_score(y_true, y_pred)

    p, r, pr_th = metrics.precision_recall_curve(y_true, recap_probs)
    fscore = (2 * p * r) / (p + r)
    ix = np.argmax(fscore)
    fig, ax = plt.subplots()
    ax.plot(r, p, color='purple')
    ax.scatter(r[ix], p[ix], marker='o', color='orange', label='Best')
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    plt.savefig('pr_curve.png')

    plt.cla()
    plt.clf()

    fpr, tpr, roc_th = metrics.roc_curve(y_true, recap_probs)
    fnr = 1 - tpr
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='purple')
    ax.set_title('ROC Curve')
    ax.set_ylabel('True positive rate')
    ax.set_xlabel('False positive rate')
    plt.savefig('roc_curve.png')
    eer_threshold = roc_th[np.nanargmin(np.absolute((fnr - fpr)))]

    plt.cla()
    plt.clf()

    cm = metrics.confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])
    plot = sns.heatmap(df_cm, annot=True, fmt="d", cmap='Blues').get_figure()
    plot.savefig('cm.jpg')

    print(f'Accuracy: {acc:.3f}%')
    print(f'Precision: {prec:.3f}%')
    print(f'Recall: {rec:.3f}%')
    print(f'F1 Score: {f1:.3f}')
    print('Best F1 Threshold=%f, F-Score=%.3f' % (pr_th[ix], fscore[ix]))
    print(f'EER: {fpr[np.nanargmin(np.absolute((fnr - fpr)))]:.3f}, Threshold: {eer_threshold:.3f}')
    print(f'Total test time: {total_time:.2f}s')
