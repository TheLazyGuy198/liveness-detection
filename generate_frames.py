import argparse
import os
import cv2
from glob import glob
from tqdm import tqdm


def split_by_vid(vid_root, frame_root, frame_freq=15):
    paths = glob(vid_root + '/**/*.mp4', recursive=True)
    pbar = tqdm(paths, total=len(paths))

    for path in pbar:
        vid_cap = cv2.VideoCapture(path)
        vid_name = os.path.basename(path)
        dataset = os.path.dirname(path).split('/')[-2]
        if dataset == 'test':
            continue
        dest = os.path.dirname(path).replace(vid_root, frame_root)
        os.makedirs(dest, exist_ok=True)

        cnt_frames = 0
        succ, frame = vid_cap.read()
        while succ:
            if cnt_frames % frame_freq == 0:
                frame_name = vid_name + '_frame_' + str(cnt_frames) + '.png'
                cv2.imwrite(os.path.join(dest, frame_name), frame)
            succ, frame = vid_cap.read()
            cnt_frames += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid-root', type=str, help='Path to training video dataset', default=None)
    parser.add_argument('--frame-root', type=str, help='Path to save frame', default=None)
    parser.add_argument('--frame-freq', type=int, help='Frame frequency', default=20)
    args = parser.parse_args()

    """
    Video dataset should be organized as following
    vid-path
        |----videos
                |----0.mp4
                |----1.mp4
        label.csv (label file contains name of video and its label (0=non-live video, 1=live video)
    """

    # split_by_image(args.path)
    split_by_vid(args.vid_root, args.frame_root, args.frame_freq)
