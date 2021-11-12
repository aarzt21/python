
""" 
This is one of the most popular interview questions: 
Given an array of numbers [2,7,11,15] and a target [9]
return the indices of the two numbers in the array that add up to the target. 
Assumptions: there is exactly one solution

Alex: 
The brute force approach would be a nested for 
loop which checks for every element where the 
other summand is. This would result in a time
complexity of O(n^2) (worst case.)

However, there is a more efficient approach 
which relies on a hash-map and has time 
complexity of O(n).

"""

def two_sum(arr, target): 
    """
    arr: List[int]
    target: int
    return: [idx1,idx2]
    """

    dic = {}

    for idx, val in enumerate(arr):
        diff = target - val
        if diff in dic: 
            return [dic[diff], idx]
        else: 
            dic[val] = idx


#test 
a = [12,31, 3,2, 12, 19,21,32,99]

print(two_sum(a, 5))







