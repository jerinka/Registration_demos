import argparse

import cv2
import numpy as np

def show(img,win='img',time=30, destroy=True):
    cv2.namedWindow(win,cv2.WINDOW_NORMAL)
    cv2.imshow(win,img)
    cv2.waitKey(time)
    if destroy:
        cv2.destroyWindow(win)
        
def fuse(moving, fixed):
    fuz = np.zeros([moving.shape[0],moving.shape[1],3], dtype=np.uint8)
    fuz[:,:,1] = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
    fuz[:,:,2] = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
    show(fuz,win='fused',time=0)
    return fuz

def keypoint_register(fixed, moving):

    MIN_MATCHES = 50

    orb = cv2.ORB_create(nfeatures=5000)
    kp1, des1 = orb.detectAndCompute(moving, None)
    kp2, des2 = orb.detectAndCompute(fixed, None)

    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # As per Lowe's ratio test to filter good matches
    good_matches = []
    
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            matchesMask[i]=[1,0]
            good_matches.append(m)

    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)

    imMatches = cv2.drawMatchesKnn(moving, kp1, fixed, kp2,matches,None,**draw_params) 
    show(imMatches,win='imMatches',time=0)
    
    
    if len(good_matches) > MIN_MATCHES:
        src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        moved = cv2.warpPerspective(moving, m, (fixed.shape[1], fixed.shape[0]))
    else:
        print('registration failed!')
        moved=None
    return moved
    
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--src" , default='images/im2.jpg', help="path for the object image")
    parser.add_argument("--dest", default='images/im1.jpg', help="path for image containing the object")
    args = parser.parse_args()

    moving = cv2.imread(args.src)
    fixed = cv2.imread(args.dest)
    
    moved = keypoint_register(fixed, moving)
    
    fuz = fuse(fixed, moved)
    cv2.imwrite('output/fused.jpg',fuz)
    
    
