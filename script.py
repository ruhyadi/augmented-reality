import argparse
import cv2
from cv2 import rectangle
import numpy as np
import math
import os

MIN_MATCHES = 20

rectangle = True
matches = True
draw_augmented = True

def main():
    homography = None 
    # create ORB keypoint detector
    orb = cv2.ORB_create()
    # create BFMatcher object based on hamming distance  
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # load the reference surface that will be searched in the video stream
    dir_name = os.getcwd()
    model = cv2.imread(os.path.join(dir_name, 'pattern_dm.png'), 0)
    # Compute model keypoints and its descriptors
    kp_model, des_model = orb.detectAndCompute(model, None)
    # # Load 3D model from OBJ file
    # obj = OBJ(os.path.join(dir_name, 'models/cube.obj'), swapyz=True)
    # init video capture
    cap = cv2.VideoCapture('darkmagician.mp4')

    overlay_image = cv2.imread('darkmagician.jpg')

    while True:
        # read the current frame
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            return
        # find and draw the keypoints of the frame
        kp_frame, des_frame = orb.detectAndCompute(frame, None)
        # match frame descriptors with model descriptors
        matches = bf.match(des_model, des_frame)
        # sort them in the order of their distance
        # the lower the distance, the better the match
        matches = sorted(matches, key=lambda x: x.distance)

        # compute Homography if enough matches are found
        if len(matches) > MIN_MATCHES:
            # differenciate between source points and destination points
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            # compute Homography
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if rectangle:
                # Draw a rectangle that marks the found model in the frame
                h, w = model.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                # project corners into frame
                dst = cv2.perspectiveTransform(pts, homography)
                # print('\n DST', dst.shape)
                # print(np.int32(dst))
                # connect them with lines  
                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            if draw_augmented:
                pts_1 = np.squeeze(dst)
                frame = draw_augmented_overlay(pts_1, overlay_image, frame)

            # if a valid homography matrix was found render cube on model plane
            # if homography is not None:
            #     try:
            #         # obtain 3D projection matrix from homography matrix and camera parameters
            #         projection = projection_matrix(camera_parameters, homography)  
            #         # project cube or model
            #         frame = render(frame, obj, projection, model, False)
            #         #frame = render(frame, model, projection)
            #     except:
            #         pass
            # draw first 10 matches.
            if matches:
                frame = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:10], 0, flags=2)
            # show result
            cv2.imshow('Augmented Reality', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Not enough matches found - %d/%d" % (len(matches), MIN_MATCHES))

    cap.release()
    cv2.destroyAllWindows()
    return 0

def draw_augmented_overlay(pts_1, overlay_image, image):
    """Overlay the image 'overlay_image' onto the image 'image'"""
    # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    # Define the squares of the overlay_image image to be drawn:
    pts_2 = np.float32([[0, 0], 
                        [0, overlay_image.shape[0]],
                        [overlay_image.shape[1], overlay_image.shape[0]],
                        [overlay_image.shape[1], 0]])
    # pts_2 = np.float32([[0, 0], [overlay_image.shape[1], 0], [overlay_image.shape[1], overlay_image.shape[0]],
    #                     [0, overlay_image.shape[0]]])

    # Draw border to see the limits of the image:
    cv2.rectangle(overlay_image, (0, 0), (overlay_image.shape[1], overlay_image.shape[0]), (255, 255, 0), 10)

    # Create the transformation matrix:
    M = cv2.getPerspectiveTransform(pts_2, pts_1)

    # Transform the overlay_image image using the transformation matrix M:
    dst_image = cv2.warpPerspective(overlay_image, M, (image.shape[1], image.shape[0]))
    # cv2.imshow("dst_image", dst_image)

    # Create the mask:
    dst_image_gray = cv2.cvtColor(dst_image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(dst_image_gray, 0, 255, cv2.THRESH_BINARY_INV)

    # Compute bitwise conjunction using the calculated mask:
    image_masked = cv2.bitwise_and(image, image, mask=mask)
    # cv2.imshow("image_masked", image_masked)

    # Add the two images to create the resulting image:
    result = cv2.add(dst_image, image_masked)
    return result

main()