import sys, string
class Solution:
	def twoSum(self, nums, target):
		if len(nums) <= 1:
			return False
		buff = {}
		for i in range(len(nums)):
			if nums[i] in buff:
				return [buff[nums[i]], i]
			else:
				buff[target - nums[i]] = i

	def reverse(self, x):
		ret = 0
		if x >= sys.maxint or x<-sys.maxint:
			return ret
		neg = (x < 0) and True or False
		x = (x < 0) and abs(x) or x
		y = str(x)
		ret = 0
		for i in range(len(y)):
			ret += int(y[i])*(10**i)
		if abs(ret) > 0x7FFFFFFF:
			ret = 0
		return (neg) and (0 - ret) or ret

	def threeSum(self, nums):
		nums.sort()
		ret = []
		for i in range(len(nums)-2):
			m, n = i + 1, len(nums) - 1
			while m < n:
				s = nums[i] + nums[m] + nums[n]
				if s < 0:
					m += 1
				elif s > 0:
					n -= 1
				else:
					tmp = [nums[i], nums[m], nums[n]]
					tmp.sort()
					if tmp not in ret:
						ret.append(tmp)
					m += 1
					n -= 1
		return ret

#	def fourSum(self, nums, target)

	def myPow(self, x, n):
		if n == 0:
			return 1
		if n < 0:
			x = 1/x
			n = -n
		return ((n % 2) == 0) and self.myPow(x*x, n/2) or x * self.myPow(x*x, n/2)

	def numDecodings(self, s):
		char_list = [f for f in string.uppercase]
		num_list = str([f for f in range(10)])
		ret = []
		if s == "":
			return 1
		for i in s:
			index = int(i)
			ret.append(char_list[index - 1])
		return ret

	def zigzagConvert(self, s, numRows):
		if numRows < 2 or len(s) < numRows:
			return s
		rows = [""] * numRows
		index, step = 0, 1
		for item in s:
			rows[index] += item
			if index == 0:
				step = 1
			elif index == numRows - 1:
				step = -1
			index += step
		return "".join(rows)
	
#	def findMedianSortedArrays(self, nums1, nums2):


arr1 = [1,2]
arr2 = [3,4]

#x = -1234
#res = res1.reverse(x)
#print res

#nums = [0,7,-4,-7,0,14,-6,-4,-12,11,4,9,7,4,-10,8,10,5,4,14,6,0,-9,5,6,6,-11,1,-8,-1,2,-1,13,5,-1,-2,4,9,9,-1,-3,-1,-7,11,10,-2,-4,5,10,-15,-4,-6,-8,2,14,13,-7,11,-9,-8,-13,0,-1,-15,-10,13,-2,1,-1,-15,7,3,-9,7,-1,-14,-10,2,6,8,-6,-12,-13,1,-3,8,-9,-2,4,-2,-3,6,5,11,6,11,10,12,-11,-14]
#nums = [-1, 0, 1, 2, -1, -4]
#res = res1.threeSum(nums)
#print res
#x = float(8.6631)
#n = int(3)
#y = res1.myPow(x, n)
#y = res1.numDecodings(s)
#print y
