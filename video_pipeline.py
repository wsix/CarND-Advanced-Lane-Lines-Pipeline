import os
import cv2
import glob
import numpy as np
import matplotlib.image as mpimg
from datetime import datetime
from image_process import *


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients of the last n iterations
        self.recent_fit = []
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = []
        # distance in meters of vehicle center from the line
        self.line_base_pos = []
        # difference in fit coefficients between last and new fits
        self.diffs = []


# =========== Computed the camera matrix and distortion coefficients ===========
images = glob.glob('camera_cal/calibration*.jpg')
nx = 9  # the number of inside corners in x
ny = 6  # the number of inside corners in y

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

objp = np.zeros((ny * nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

for fname in images:
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret:
        imgpoints.append(corners)
        objpoints.append(objp)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img[:, :, 0].shape[::-1], None, None)
# ==============================================================================

# ================= Compute M & Minv for perspective transform =================
offsetx = 280    # x dim offset for dst points
offsety = 0      # y dim offset for dst points
img_size = (img.shape[1], img.shape[0])
src = np.float32([(594, 449), (687, 449), (239, 703), (1086, 703)])
dst = np.float32([
    [offsetx, offsety], [img_size[0] - offsetx, offsety],
    [offsetx, img_size[1] - offsety],
    [img_size[0] - offsetx, img_size[1] - offsety]
])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
# ==============================================================================

# Read images from video
cap = cv2.VideoCapture('project_video.mp4')
# Define instances track the lane line detection
left_line, right_line = Line(), Line()
count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    undist = cv2.undistort(frame, mtx, dist, None, mtx)
    result = pipeline(undist, s_thresh=(170, 250))
    binary_warped = cv2.warpPerspective(result, M, img_size, flags=cv2.INTER_LINEAR)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    if not (left_line.detected & right_line.detected):
        left_fit, right_fit, left_lane_inds, right_lane_inds = fit_polynomial(binary_warped, nonzero)
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    else:
        margin = 100
        left_lane_inds = (
            (nonzerox > (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
            (nonzerox < (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] + margin))
        )
        right_lane_inds = (
            (nonzerox > (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
            (nonzerox < (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] + margin))
        )
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720   # meters per pixel in y dimension
    xm_per_pix = 3.7 / 720  # meters per pixel in x dimension

    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
    curverad = (left_curverad + right_curverad) / 2

    # Left and Right line only store the last 5 fits result to smooth the current result
    if count < 5:
        if count > 0:
            left_line.diffs = left_line.current_fit - left_fit
            right_line.diffs = right_line.current_fit - right_fit
        left_line.recent_xfitted.append(left_fitx)
        left_line.recent_fit.append(left_fit)
        right_line.recent_xfitted.append(right_fitx)
        right_line.recent_fit.append(right_fit)
    else:
        left_line.diffs = left_line.current_fit - left_fit
        right_line.diffs = right_line.current_fit - right_fit
        # Detect whether the poly coefficients differentes between two frame is too large
        if np.sum(np.square(left_line.diffs)) > 200:
            left_line.detected = False
        else:
            left_line.detected = True
            left_line.recent_xfitted = np.concatenate((left_line.recent_xfitted[1:], [left_fitx]))
            left_line.recent_fit = np.concatenate((left_line.recent_fit[1:], [left_fit]))

        if np.sum(np.square(right_line.diffs)) > 200:
            right_line.detected = False
        else:
            right_line.detected = True
            right_line.recent_xfitted = np.concatenate((right_line.recent_xfitted[1:], [right_fitx]))
            right_line.recent_fit = np.concatenate((right_line.recent_fit[1:], [right_fit]))

    left_line.bestx = np.mean(left_line.recent_xfitted, axis=0)
    right_line.bestx = np.mean(right_line.recent_xfitted, axis=0)
    left_line.best_fit = np.mean(left_line.recent_fit, axis=0)
    right_line.best_fit = np.mean(right_line.recent_fit, axis=0)
    left_line.current_fit = left_fit
    right_line.current_fit = right_fit
    left_line.radius_of_curvature = left_curverad
    right_line.radius_of_curvature = right_curverad
    left_line.linebase_pos = (621 - np.mean(left_fitx[-3:])) * xm_per_pix
    right_line.linebase_pos = (np.mean(right_fitx[-3:]) - 621) * xm_per_pix

    vehicle2center = (621 - (np.mean(left_fitx[-3:]) + np.mean(right_fitx[-3:])) / 2) * xm_per_pix

    # ================= Plotted lane area back down onto the road =================
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_line.bestx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.bestx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (frame.shape[1], frame.shape[0]))
    # Draw curvature and vehicle position onto the image
    cv2.putText(newwarp, 'Radius of Curvature = {:.2f}(m)'.format(curverad), (30, 30), 0, 0.9, (255, 255, 255), 2)
    cv2.putText(newwarp, 'Radius of Left Line Curvature = {:.2f}(m).'.format(left_line.radius_of_curvature), (30, 60), 0, 0.9, (255, 255, 255), 2)
    cv2.putText(newwarp, 'Radius of Right Line Curvature = {:.2f}(m).'.format(right_line.radius_of_curvature), (30, 90), 0, 0.9, (255, 255, 255), 2)
    cv2.putText(newwarp, 'Vehicle is {:.2f}m to left lane line.'.format(left_line.linebase_pos), (30, 120), 0, 0.9, (255, 255, 255), 2)
    cv2.putText(newwarp, 'Vehicle is {:.2f}m to right lane line.'.format(right_line.linebase_pos), (30, 150), 0, 0.9, (255, 255, 255), 2)
    if vehicle2center >= 0:
        cv2.putText(newwarp, 'Vehicle is {:.4f}m right of center'.format(vehicle2center), (30, 180), 0, 0.9, (255, 255, 255), 2)
    else:
        cv2.putText(newwarp, 'Vehicle is {:.4f}m left of center'.format(-vehicle2center), (30, 180), 0, 0.9, (255, 255, 255), 2)
    # Combine the result with the original image
    output = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # =============================================================================

    timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
    image_filename = os.path.join('output_images/project_video_output/', timestamp)
    cv2.imwrite('{}.jpg'.format(image_filename), output)
    if count % 20 == 0:
        print('{} Done!'.format(count))
    count += 1

cap.release()
cv2.destroyAllWindows()
