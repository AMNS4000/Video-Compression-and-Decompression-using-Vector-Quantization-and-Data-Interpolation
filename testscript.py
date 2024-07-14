import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lbg
import compression
import cv2
import concurrent.futures
import time
import os
def compression1(img):
    frame_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cbf, crf = cv2.split(frame_ycbcr)
    y1 = compression.sim_protocol(y,4,0.1,(8,8),'./Decompressed', 'y')
    cb1 = compression.sim_protocol(cbf,4,0.1,(8,8),'./Decompressed', 'cbf')
    cr1 = compression.sim_protocol(crf,4,0.1,(8,8),'./Decompressed', 'crf')
    y_bl = np.clip(y1, 0, 255).astype(np.uint8)
    cb_bl = np.clip(cb1, 0, 255).astype(np.uint8)
    cr_bl = np.clip(cr1, 0, 255).astype(np.uint8)
    frame_decompressed_ycbcr = cv2.merge([y_bl, cb_bl, cr_bl])
    frame_decompressed_bgr = cv2.cvtColor(frame_decompressed_ycbcr, cv2.COLOR_YCrCb2BGR)
    print("Done")
    return frame_decompressed_bgr
def main():
    cnt = 0
    cp1 = cv2.VideoCapture('test.mp4')
    c1 = []
    print("Started")
    print(len(c1))
    start_time = time.time()
    decomp_img = []
    while True:
        ret,img_dim1= cp1.read()
        if not ret:
            break
        if (cnt%100==0):
            c1.append(img_dim1)
        cnt+=1
    cp1.release()
    indx = 0
    for i in c1:
        decomp_img.append(compression1(i))
        file_name = f"frame_{indx:04d}.png"
        cv2.imwrite(os.path.join("./Decompressed", file_name), decomp_img[indx])
        indx+=1
        print(indx)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken for compressing 14000 frames: {total_time:.2f} seconds")
if __name__ == "__main__":
    main()
