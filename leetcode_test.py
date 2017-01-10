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
	
	def findMedianSortedArrays(self, nums1, nums2):
		ret = nums1 + nums2
		ret.sort()
		if len(ret) == 2:
			m = (float(ret[0]) + float(ret[1]))/2
		elif len(ret) != 2 and (len(ret) % 2) == 0:
			m = float(float(ret[(len(ret)/2)-1]) + float(ret[(len(ret)/2)]))/2
		else:
			m = float(ret[(len(ret)/2)])
		return m

	def isPalindrome(self, x):
		if x == 0:
			return True
		elif x < 0:
			return False
		else:
			y = 0
			tmp = x
			while(x != 0):
				y = (y * 10) + (x % 10)
				x = x / 10
			if y == tmp:
				return True
			else:
				return False

	def removeDuplicates(self, nums):
		ret = 1
		i = 1
		end = len(nums)
		if not nums:
			return 0
		while i < end:
			if nums[i] != nums[i - 1]:
				ret += 1
				i += 1
			else:
				del nums[i]
				end -= 1
		return ret

	def removeElement(self, nums, val):
		i = 0
		end = len(nums)
		if not nums:
			return 0
		while i < end:
			if nums[i] == val:
				del nums[i]
				end -= 1
			else:
				i += 1
		return len(nums)

	def searchRange(self, nums, target):
		ret = []
		tmp = []
		for i in range(len(nums)):
			if nums[i] == target:
				tmp.append(i)
		if not tmp:
			ret = [-1,-1]
		else:
			ret = [min(tmp), max(tmp)]
		return ret

	def strStr(self, haystack, needle):
		end = len(haystack) - len(needle) + 1
		for i in range(0, end):
			tmp = haystack[i:i+len(needle)]
			print haystack[i:i+len(needle)]
			if tmp == needle:
				return i
		i = -1
		return i

	def permute(self, nums):
		return [[j] + p for i,j in enumerate(nums) for p in self.permute((nums[:i]+nums[i+1:]))] or [[]]

	def permuteUnique(self, nums):
		ret = self.permute(nums)
		end = len(ret)
		i = 0
		while i < end:
			tmp1 = ret[i]
			tmp2 = ret[i+1:]
			if tmp1 in tmp2:
				del ret[i]
				end -= 1
				i = 0
			else:
				i += 1
		return ret
	def countAndSay(self, n):
		if n < 1:
			return ""
		tmp = str(n)
		ret = ''
		tmp_dict = {}
		if len(tmp) == 1:
			for i in range(int(tmp[0])):
				ret += '1'
			return ret
		for i in range(len(tmp)):

			if tmp[i] not in tmp_dict:
				tmp_dict[tmp[i]] = 1
			else:
				tmp_dict[tmp[i]] += 1
		for k in tmp_dict.keys():
			ret += str(tmp_dict[k])
			ret += k
		return ret
#	def lengthOfLongestSubstring(self, s):

n = 2

sol = Solution()
print sol.countAndSay(n)
