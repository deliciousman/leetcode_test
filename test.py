import cv2
import os, sys
import string, random
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


abnormal_result_dir = "/home/vincent/codebase/shipcontainer/test/model2_analyze/3w_abnormal_model2_result/conv3/"
reduce_iters_dir = "/home/vincent/codebase/shipcontainer/test/model2_analyze/reduce_iterations/conv4/"
reduce_class_amount_dir = "/home/vincent/codebase/shipcontainer/test/model2_analyze/20_class_result/conv5/"
save_pathh = "/home/vincent/codebase/shipcontainer/test/save_noise/"

keys = list(map(str, range(10))) + [i for i in string.ascii_uppercase]
font_list = [f for f in glob.glob('./font_file/**/*.*')]

pure_sample_dir = "/home/vincent/codebase/shipcontainer/test/test_noise/*"
pure_sample = [f for f in glob.glob(pure_sample_dir)]

def read_file(filename):
	ret = []
	with open(filename, 'r') as f:
		ret = f.readlines()
	return ret

def write_file(data):
	with open('test.txt', 'a') as f:
		tmp_data = data + "\n"
		f.write(tmp_data)

def get_imgfile(path):
	filepath = "%s*.jpg" % path
	return [f for f in glob.glob(filepath)]

def combine_img(imgfile, row, col):
	cnt = 0
	for imgname in imgfile:
		img = cv2.imread(imgname)
		gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		plt.subplot(row, col,cnt)
		plt.axis('off')
		plt.imshow(gimg, cmap='gray')
		cnt += 1


def main():
	imgfile = get_imgfile(abnormal_result_dir)
	print len(imgfile)
	combine_img(imgfile, 16, 24)

if __name__ == '__main__':
	main()
	sys.exit(0)
