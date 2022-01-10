import argparse
import os
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

def blend(fixed, moved):
    #import pdb;pdb.set_trace()
    black = np.where(moved==(0,0,0))
    blended = moved.copy()
    blended[black]=fixed[black]
    show(blended,win='blended',time=0)
    return blended

def concat(fixed, moved):
    #import pdb;pdb.set_trace()
    concated = np.concatenate((fixed,moved),axis=1)
    show(concated,win='concated',time=0)
    return concated
    
def keypoint_register(fixed, moving):
    """ Calcualtes transform matrix(M) bw fixed and moving, returns moved image and M"""
    MIN_MATCHES = 1

    orb = cv2.ORB_create(nfeatures=5000, scaleFactor = 1.2, nlevels = 1,
                         edgeThreshold = 8, firstLevel = 0,
                         WTA_K = 2, scoreType = cv2.FAST_FEATURE_DETECTOR_TYPE_9_16, 	
                         patchSize = 30, fastThreshold = 20 )
                         
    kp1, des1 = orb.detectAndCompute(moving, None)
    kp2, des2 = orb.detectAndCompute(fixed, None)
    #import pdb;pdb.set_trace()
    fixed_kp = cv2.drawKeypoints(moving,kp1,None,color=(0,255,0))
    moving_kp = cv2.drawKeypoints(fixed,kp2,None,color=(0,255,0))
    show(fixed_kp,win='fixed_kp',time=30,destroy=False )
    show(moving_kp,win='moving_kp',time=0)
    cv2.destroyWindow('moving_kp')

    
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
    print('num matches:',len(good_matches))
    
    
    if len(good_matches) > MIN_MATCHES:
        src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        moved = cv2.warpPerspective(moving, M, (fixed.shape[1], fixed.shape[0]))
    else:
        print('registration failed!')
        moved=None
        M=None
    return moved, M
    
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-f',"--fixed"  , default='crops/18.png', help="path for fixed image")
    parser.add_argument('-m',"--moving" , default='crops/19.png', help="path for moving image")

    args = parser.parse_args()
    
    fixed = cv2.imread(args.fixed)
    moving = cv2.imread(args.moving)

    moved, M = keypoint_register(fixed, moving)
    
    outfile = os.path.basename(args.moving)
    outfile = os.path.splitext(outfile)[0]
    outfile = os.path.join('output', outfile)
    
    cv2.imwrite(outfile+'_moved.jpg',moved)
    
    fuz = fuse(fixed, moved)
    cv2.imwrite(outfile+'_fused.jpg',fuz)
    
    blended = blend(fixed, moved)
    cv2.imwrite(outfile+'_blended.jpg',blended)
    
    concated = concat(fixed, moved)
    cv2.imwrite(outfile+'_concated.jpg',concated)
    
