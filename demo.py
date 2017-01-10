import numpy as np
import cv2
import sys

cnt = 0
src_dir = "/home/vincent/codebase/testing/leetcode/0.jpg"

def cal_surf_feature(src_img, dst_img):
	orb = cv2.ORB_create()
	k1, des1 = orb.detectAndCompute(src_img, None)
	k2, des2 = orb.detectAndCompute(dst_img, None)
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	try:
		matches = bf.match(des1, des2)
		matches = sorted(matches, key = lambda x:x.distance)
		if len(matches) < 100:
			ret = np.zeros((src_img.shape[0], src_img.shape[1]*2, src_img.shape[2]))
			ret[:, 0:src_img.shape[1]] = src_img
			ret[:, src_img.shape[1]:src_img.shape[1]*2] = dst_img
			ret = np.uint8(ret)
		else:
			ret = cv2.drawMatches(src_img, k1, dst_img, k2, matches[:100], None, flags=2)
	except:
		ret = np.zeros((src_img.shape[0], src_img.shape[1]*2, src_img.shape[2]))
		ret[:, 0:src_img.shape[1]] = src_img
		ret[:, src_img.shape[1]:src_img.shape[1]*2] = dst_img
		ret = np.uint8(ret)
	return ret

def main():
	cap = cv2.VideoCapture(0)
	src_img = cv2.imread(src_dir)
	while True:
		ret, frame = cap.read()
		res = cal_surf_feature(src_img, frame)
		cv2.imshow("Frame", res)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
	sys.exit(0)
