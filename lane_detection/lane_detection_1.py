#from moviepy.editor import VideoFileClip
# -- coding: UTF-8 --
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import numpy as np
import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

blur_ksize = 5  # Gaussian blur kernel size
canny_lthreshold = 50  # Canny edge detection low threshold
canny_hthreshold = 150  # Canny edge detection high threshold

# Hough transform parameters
rho = 1
theta = np.pi / 180
threshold = 15
min_line_length = 40
max_line_gap = 20

def roi_mask(img, vertices):
  mask = np.zeros_like(img)

  #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
  if len(img.shape) > 2:
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    mask_color = (255,) * channel_count
  else:
    mask_color = 255

  cv2.fillPoly(mask, vertices, mask_color)
  masked_img = cv2.bitwise_and(img, mask)
  return masked_img


def draw_roi(img, vertices):
    cv2.polylines(img, vertices, True, [255, 0, 0], thickness=2)


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
  for line in lines:
    for x1, y1, x2, y2 in line:
      cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    draw_lanes(line_img, lines)
    return line_img


def draw_lanes(img, lines, color=[255, 0, 0], thickness=8):
  left_lines, right_lines = [], []
  for line in lines:
    for x1, y1, x2, y2 in line:
        if x2-x1 != 0:
          k = (y2 - y1) / (x2 - x1)
          if k < 0:
            left_lines.append(line)
          else:
             right_lines.append(line)
        else:
            continue

  if (len(left_lines) <= 0 or len(right_lines) <= 0):
    return img
  
  clean_lines(left_lines, 0.1)
  clean_lines(right_lines, 0.1)
  left_points = [(x1, y1) for line in left_lines for x1,y1,x2,y2 in line]
  left_points = left_points + [(x2, y2) for line in left_lines for x1,y1,x2,y2 in line]
  right_points = [(x1, y1) for line in right_lines for x1,y1,x2,y2 in line]
  right_points = right_points + [(x2, y2) for line in right_lines for x1,y1,x2,y2 in line]
  
  left_vtx = calc_lane_vertices(left_points, 325, img.shape[0])
  right_vtx = calc_lane_vertices(right_points, 325, img.shape[0])
  
  cv2.line(img, left_vtx[0], left_vtx[1], color, thickness)
  cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)
  
def clean_lines(lines, threshold):
  slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
  while len(lines) > 0:
    mean = np.mean(slope)
    diff = [abs(s - mean) for s in slope]
    idx = np.argmax(diff)
    if diff[idx] > threshold:
      slope.pop(idx)
      lines.pop(idx)
    else:
      break
  
  
def calc_lane_vertices(point_list, ymin, ymax):
  x = [p[0] for p in point_list]
  y = [p[1] for p in point_list]
  fit = np.polyfit(y, x, 1)
  fit_fn = np.poly1d(fit)
  
  xmin = int(fit_fn(ymin))
  xmax = int(fit_fn(ymax))
  
  return [(xmin, ymin), (xmax, ymax)]


def process_an_image(img):
  roi_vtx = np.array([[(0, img.shape[0]), (460, 325), (520, 325), (img.shape[1], img.shape[0])]])
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)
  edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)
  roi_edges = roi_mask(edges, roi_vtx)
  cv2.imshow('gray', gray)
  cv2.imshow('blur_gray', blur_gray)
  cv2.imshow('edges', edges)
  cv2.imshow('roi_edges', roi_edges)
  cv2.waitKey(0)
  try:
      line_img = hough_lines(roi_edges, rho, theta, threshold, min_line_length, max_line_gap)
      res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)
      return res_img
  except:
      print ('Sorry,the image can not detect lines')
      return False



  '''
  plt.figure()
  plt.imshow(img)
  plt.savefig('images/lane_original.png', bbox_inches='tight')
  plt.figure()
  plt.imshow(gray, cmap='gray')
  plt.savefig('images/gray.png', bbox_inches='tight')
  plt.figure()
  plt.imshow(blur_gray, cmap='gray')
  plt.savefig('images/blur_gray.png', bbox_inches='tight')
  plt.figure()
  plt.imshow(edges, cmap='gray')
  plt.savefig('images/edges.png', bbox_inches='tight')
  plt.figure()
  plt.imshow(roi_edges, cmap='gray')
  plt.savefig('images/roi_edges.png', bbox_inches='tight')
  plt.figure()
  plt.imshow(line_img, cmap='gray')
  plt.savefig('images/line_img.png', bbox_inches='tight')
  plt.figure()
  plt.imshow(res_img)
  plt.savefig('images/res_img.png', bbox_inches='tight')
  plt.show()
  '''


def video_demo():
    capture = cv2.VideoCapture(0)#打开摄像头，0代表的是设备id，如果有多个摄像头，可以设置其他数值
    while True:
        ret, frame = capture.read() #读取摄像头,它能返回两个参数，第一个参数是bool型的ret，其值为True或False，代表有没有读到图片；第二个参数是frame，是当前截取一帧的图片
        frame = cv2.flip(frame, 1)#翻转 0:上下颠倒 大于0水平颠倒   小于180旋转
        out = process_an_image(frame)
        cv2.imshow("video", out)
        if cv2.waitKey(10) & 0xFF == ord('q'): #键盘输入q退出窗口，不按q点击关闭会一直关不掉 也可以设置成其他键。
            break
            return out


def talker():
    pub = rospy.Publisher('/tutorial/image', Image, queue_size=1)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(30)
    bridge = CvBridge()
    while not rospy.is_shutdown():
        img = video_demo()
        cv2.imshow("talker", img)
        cv2.waitKey(3)
        pub.publish(bridge.cv2_to_imgmsg(img, "bgr8"))
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

 
# img = mplimg.imread("lane.jpg") 
# process_an_image(img)

#output = 'video_4_sol.mp4'
#clip = VideoFileClip("video_2.mp4")
#out_clip = clip.fl_image(process_an_image)
#out_clip.write_videofile(output, audio=False)
clip = video_demo()
cv2.imwrite('out.jpg', clip)
talker()

