

""" 
Problem:
Given an integer array nums, return all the triplets
[nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, 
and nums[i] + nums[j] + nums[k] == 0 

Note: solution set must not contain duplicate triplets


Alex: 
Brute force would be a worst case time complexity of O(n^3)

However, by sorting it and applying the 2sum problem we can
reduce the complexity to O(n log n) + O(n^2) --> O(n^2)

"""

def threeSum(nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
        """
    if len(nums) < 3: return []
    if all(x == 0 for x in nums): return [[0,0,0]]
    if all(x <= 0 for x in nums): return []
    if all(x >= 0 for x in nums): return []
    
    nums.sort()
    k = 0

    prev = None
    
    triplets = []
    
    while nums[k] <= 0: 
        if nums[k] == prev: 
            k += 1
            continue
        else: 
            l = k+1
            r = len(nums)-1
            target = -nums[k]
            
            #2sum now
            while l < r:
                if nums[l] + nums[r] == target: 
                    triplets.append([nums[k], nums[l], nums[r]])
                    l += 1
                    while nums[l] == nums[l-1] and l < r: 
                        l += 1
                        
                    
                elif nums[l] + nums[r] > target: 
                    r -= 1
                else: 
                    l += 1
                
            prev = nums[k]
            k += 1
        
        
    return triplets
                    

#test
# 
nums = [-1,0,1,2,-1,-4]  # --> [[-1,-1,2],[-1,0,1]]                  

print(threeSum(nums))
                    