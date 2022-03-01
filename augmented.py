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

def augmented(
    pattern_path,
    overlay_path,
    video_path,
    output_path,
    viz_matches,
    viz,
    notebook_mode
    ):
    homography = None 
    # create ORB keypoint detector
    orb = cv2.ORB_create()
    # create BFMatcher object based on hamming distance  
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # load the reference surface that will be searched in the video stream
    dir_name = os.getcwd()
    model = cv2.imread(os.path.join(dir_name, pattern_path), 0)
    # Compute model keypoints and its descriptors
    kp_model, des_model = orb.detectAndCompute(model, None)

    overlay_image = cv2.imread(overlay_path)

    # init video capture
    cap = cv2.VideoCapture(video_path)
    h = int(cap.get(4))
    # conditional if viz true
    if viz:
        w = int(cap.get(3)) * 2
    else:
        w = int(cap.get(3))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), 30, (w, h))

    while True:
        # read the current frame
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            return

        # copy originial frame
        ori_frame = frame.copy()

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

            # Draw a rectangle that marks the found model in the frame
            h, w = model.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # project corners into frame
            dst = cv2.perspectiveTransform(pts, homography)
            # connect them with lines  
            frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            pts_1 = np.squeeze(dst)
            frame = draw_augmented_overlay(pts_1, overlay_image, frame)

            if viz_matches:
                frame = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:10], 0, flags=2)

            if viz:
                frame = np.hstack((ori_frame, frame))

            if output_path is not None:
                out.write(frame)

            if notebook_mode:
                # show result
                cv2.imshow('Augmented Reality', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            print("Not enough matches found - %d/%d" % (len(matches), MIN_MATCHES))

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return 0

def draw_augmented_overlay(pts_1, overlay_image, image):
    """Overlay the image 'overlay_image' onto the image 'image'"""
    # Define the squares of the overlay_image image to be drawn:
    pts_2 = np.float32([[0, 0], 
                        [0, overlay_image.shape[0]],
                        [overlay_image.shape[1], overlay_image.shape[0]],
                        [overlay_image.shape[1], 0]])

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


if __name__ == '__main__':
    # Check the help parameters to understand arguments
    parser = argparse.ArgumentParser(description='Augmented Reality')
    parser.add_argument('--pattern_path', type=str, required=True, help='image directory path')
    parser.add_argument('--overlay_path', type=str, required=True,  help='image format, png/jpg')
    parser.add_argument('--video_path', type=str, required=True, help='video path')
    parser.add_argument('--output_path', type=str, required=False, help='output video path if required')
    parser.add_argument('--viz_matches', action='store_true', help='matches will be draw, but cannot visualize different')
    parser.add_argument('--viz', action='store_true', help='visualize different, but cannot draw matches')
    parser.add_argument('--notebook', action='store_true', help='Show imshow output, if false worked for colab')

    args = parser.parse_args()

    augmented(args.pattern_path, args.overlay_path, args.video_path, args.output_path, args.viz_matches, args.viz, args.notebook)