
""" 
Popular interview question: Write a function that converts integers into roman numeral.

Constraints: 1 <= num <= 3999
"""


def to_roman(num): 
    roman = [[1000,"M"], [900, "CM"], [500,"D"], [400, "CD"], [100,"C"],[90, "XC"], [50,"L"], [40, "XL"],[10,"X"], [9, "IX"], [5, "V"], [4, "IV"],[1,"I"]]
        
    res = ""
        
    while num != 0: 
        for pair in roman: 
            if num//pair[0] >= 1: 
                num -= pair[0]
                res += pair[1]
                break
                
    return res



print(to_roman(3))
print(to_roman(4))
print(to_roman(9))
print(to_roman(58))
print(to_roman(1994))
















