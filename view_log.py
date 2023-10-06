import numpy as np
import cv2
import os

gen = 2563
rewards = 15.0
OLOG_DIR = 'vlog'
DAT_NAME = 'episode-{}-reward-{}.npy'.format(gen, rewards)
LOG_DIR = 'videos'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
frames = np.load(OLOG_DIR + '/' + DAT_NAME, allow_pickle=True)

out = cv2.VideoWriter(LOG_DIR + '/' + DAT_NAME + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (160, 210))
for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
out.release()
