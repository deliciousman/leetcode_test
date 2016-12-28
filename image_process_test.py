import cv2
import os, sys, shutil
import string, random
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps, ImageFilter
from scipy import ndimage
import glob

imgfiles = [f for f in glob.glob('./new_golden_image/*.jpg')]
save_dir = "/home/vincent/codebase/shipcontainer/test/light/light_result/"

def median_filter(img, imgname, k_size):
	img[:,:,0] = ndimage.median_filter(img[:,:,0], size=(k_size,k_size))
	img[:,:,1] = ndimage.median_filter(img[:,:,1], size=(k_size,k_size))
	img[:,:,2] = ndimage.median_filter(img[:,:,2], size=(k_size,k_size))
	savename = "%s%s/%s" % (save_dir, 'median_filter', imgname)
	try:
		folder_path = "%smedian_filter/" %save_dir
		os.makedirs(folder_path)
	except:
		pass
	print(savename)
	cv2.imwrite(savename, img)
	return img

def mean_filter(img, imgname, k_size):
	img[:,:,0] = ndimage.uniform_filter(img[:,:,0], k_size)
	img[:,:,1] = ndimage.uniform_filter(img[:,:,1], k_size)
	img[:,:,2] = ndimage.uniform_filter(img[:,:,2], k_size)
	savename = "%s%s/%s" % (save_dir, 'mean_filter', imgname)
	try:
		folder_path = "%smean_filter/" % save_dir
		os.makedirs(folder_path)
	except:
		pass
	print(savename)
	cv2.imwrite(savename, img)
	return img

def sharpen_filter(alpha, img, imgname):
	gaus1_b = ndimage.gaussian_filter(img[:,:,0], 5)
	gaus1_g = ndimage.gaussian_filter(img[:,:,1], 5)
	gaus1_r = ndimage.gaussian_filter(img[:,:,2], 5)
	gaus2_b = ndimage.gaussian_filter(img[:,:,0], 3)
	gaus2_g = ndimage.gaussian_filter(img[:,:,1], 3)
	gaus2_r = ndimage.gaussian_filter(img[:,:,2], 3)
	img[:,:,0] = img[:,:,0] + alpha * (gaus1_b - gaus2_b)
	img[:,:,1] = img[:,:,1] + alpha * (gaus1_g - gaus2_g)
	img[:,:,2] = img[:,:,2] + alpha * (gaus1_r - gaus2_r)
	savename = "%ssharpenfilter/%s" % (save_dir, imgname)
	try:
		folder_path = "%ssharpenfilter/" % save_dir
		os.makedirs(folder_path)
	except:
		pass
	print(savename)
	cv2.imwrite(savename, img)
	return img

def contrast_filter(cont_coeff, light_coeff, img, imgname):
	coeff = float(cont_coeff * light_coeff)
	img.astype(float)
	mean_b = np.mean(img[:,:,0])
	mean_g = np.mean(img[:,:,1])
	mean_r = np.mean(img[:,:,2])
	img[:,:,0] = (img[:,:,0] - mean_b) * coeff
	img[:,:,1] = (img[:,:,1] - mean_g) * coeff
	img[:,:,2] = (img[:,:,2] - mean_r) * coeff
	img = np.uint8(img)
	savename = "%scontrastfilter/%s" % (save_dir, imgname)
	try:
		folder_path = "%scontrastfilter/" % save_dir
		os.makedirs(folder_path)
	except:
		pass
	print(savename)
	cv2.imwrite(savename, img)
	return img

def hist_equal(img, imgname):
	ycbcr_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
	hist, bins = np.histogram(ycbcr_img[:,:,0].flatten(), 256, [0, 256])
	cdf = hist.cumsum()
	cdf_m = np.ma.masked_equal(cdf, 0)
	cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
	cdf = np.ma.filled(cdf_m, 0).astype('uint8')
	img2 = cdf[img]
	savename = '%shisteq/%s' % (save_dir, imgname)
	try:
		folder_path = "%shisteq/" % save_dir
		os.makedirs(folder_path)
	except:
		pass
	print(savename)
	cv2.imwrite(savename, img2)
	return img2

def brightness_compensate(img, imgname):
	ycbcr_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(float)
	row , col, ch = img.shape
	brightness = np.mean(ycbcr_img[:,:,0])
	comp = float(255 / brightness)
	ycbcr_img[:,:,0][ycbcr_img[:,:,0]<brightness] = ycbcr_img[:,:,0][ycbcr_img[:,:,0]<brightness] * comp
	ycbcr_img = np.uint8(ycbcr_img)
	ret = cv2.cvtColor(ycbcr_img, cv2.COLOR_YCrCb2BGR)
	savename = '%scompensate_brightness/%s' % (save_dir, imgname)
	try:
		folder_path = "%scompensate_brightness/" % save_dir
		os.makedirs(folder_path)
	except:
		pass
	print(savename)
	cv2.imwrite(savename, ret)
	return ret

def setup_brightness(img, imgname):
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(float)
	row, col, ch = hsv_img.shape
	brightness_val = np.mean(hsv_img[:,:,0])
	if brightness_val < 50:
		add_val = brightness_val * 2
		hsv_img[:,:,0] = hsv_img[:,:,0] + add_val
	elif 50 < brightness_val < 100:
		add_val = brightness_val * 0.2
		hsv_img[:,:,0] = hsv_img[:,:,0] + add_val
	elif brightness_val > 128:
		sub_val = brightness_val * 0.2
		hsv_img[:,:,0] = hsv_img[:,:,0] - sub_val
	hsv_img[:,:,0][hsv_img[:,:,0]>255] = 255

	hsv_img = np.uint8(hsv_img)
	ret = cv2.cvtColor(hsv_img, cv2.COLOR_YCrCb2BGR)
	savename = '%sbrightness_change/%s' % (save_dir, imgname)
	try:
		folder_path = "%sbrightness_change/" % save_dir
		os.makedirs(folder_path)
	except:
		pass
	yy = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
	gr = np.mean(yy)
	print("brightness_val before: %d, after : %d" % (brightness_val, gr))
	print(savename)
	cv2.imwrite(savename, ret)
	return ret

def edge_detect(img, imgname):
	row, col, ch = img.shape
	ycbcr_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
	sobelx = cv2.Sobel(ycbcr_img[:,:,0], cv2.CV_32F, 1, 0, ksize=3)
	sobely = cv2.Sobel(ycbcr_img[:,:,0], cv2.CV_32F, 0, 1, ksize=3)
	bin_res = np.zeros(ycbcr_img[:,:,0].shape, np.uint8)
	edge_val = np.zeros(ycbcr_img[:,:,0].shape, np.uint8)
	edge_val = ((sobelx**2)+(sobely**2))**0.5
	bin_res[edge_val>150] = 255
	res, mark = cv2.connectedComponents(bin_res)
	img[:,:,0][bin_res==255] = 255
	img[:,:,1][bin_res==255] = 255
	img[:,:,2][bin_res==255] = 255
	savename = '%sedge_detect/%s' % (save_dir, imgname)
	try:
		folder_path = "%sedge_detect/" % save_dir
		os.makedirs(folder_path)
	except:
		pass
	print(savename)
	cv2.imwrite(savename, img)
	return img

def get_edge_when_low_brightness(img ,imgname):
	ycbcr_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	otsu_res, th = cv2.threshold(ycbcr_img[:,:,0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	res, mark = cv2.connectedComponents(th)
	th[mark==0] = 0
	k1 = np.ones((3,3), np.uint8)
	k2 = np.ones((9,9), np.uint8)
	ero = cv2.erode(th, k2, iterations = 1)
	dil = cv2.dilate(th, k2, iterations = 1)
	tmp = dil - ero
	row, col, ch = img.shape
	img[:,:,0][tmp==255] = 255
	img[:,:,1][tmp==255] = 255
	img[:,:,2][tmp==255] = 255
	savename = '%smor_edge/%s' % (save_dir, imgname)
	try:
		folder_name = "%smor_edge/" % save_dir
		os.makedirs(folder_name)
	except:
		pass
	print(savename)
	cv2.imwrite(savename, tmp)
	return img

def main():
	for imgitem in imgfiles:
		img = cv2.imread(imgitem)
		tmp = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		brightness = np.mean(tmp[:,:,0])
		tmpname = imgitem.split("./new_golden_image/")
		imgname = tmpname[1]
		print("brightness : %d" % brightness)
#		res = edge_detect(img, imgname)
#		tst1 = hist_equal(img, imgname)
#		tst2 = get_edge_when_low_brightness(img, imgname)
#		if brightness < 50:
#			res1 = brightness_compensate(img, imgname)
#			tst = cv2.imread(imgitem)
#			ress = get_edge_when_low_brightness(tst, imgname)
#			res = brightness_compensate(ress, imgname)
#			res = edge_detect(res1, imgname)
#		else:
#			res1 = hist_equal(img, imgname)
#		if brightness1 < 100:
#			res = brightness_compensate(img, imgname)
		res = edge_detect(img, imgname)
#			res2 = mean_filter(res, imgname, 3)
#		res = hist_equal(img, imgname)
#		img = cv2.imread(imgitem)
#		res4 = setup_brightness(img, imgname)
#		res1 = contrast_filter(1.3, 1, img, imgname)
#		img = cv2.imread(imgitem)
#		res2 = mean_filter(img, imgname)
#		img = cv2.imread(imgitem)
#		res3 = sharpen_filter(5, img, imgname)
#		img = cv2.imread(imgitem)
#		res = mean_filter(img, imgname)
#		res2 = contrast_filter(1.3, 1, res1, imgname)

if __name__ == '__main__':
	main()
	sys.exit(0)
