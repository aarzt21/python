# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:40:30 2021

@author: axarz
"""

#One of the early common methods for encrypting text was the
#Playfair cipher. You can read more about the Playfair cipher
#here: https://en.wikipedia.org/wiki/Playfair_cipher
#
#The Playfair cipher starts with a 5x5 matrix of letters,
#such as this one:
#
# D A V I O
# Y N E R B
# C F G H K
# L M P Q S
# T U W X Z
#
#To fit the 26-letter alphabet into 25 letters, I and J are
#merged into one letter. When decrypting the message, it's
#relatively easy to tell from context whether a letter is
#meant to be an i or a j.
#
#To encrypt a message, we first remove all non-letters and
#convert the entire message to the same case. Then, we break
#the message into pairs. For example, imagine we wanted to
#encrypt the message "PS. Hello, worlds". First, we could
#convert it to PSHELLOWORLDS, and then break it into letter
#pairs: PS HE LL OW OR LD S. If there is an odd number of
#characters, we add X to the end.
#
#Then, for each pair of letters, we locate both letters in
#the cipher square. There are four possible orientations
#for the pair of letters: they could be in different rows
#and columns (the "rectangle" case), they could be in the
#same row but different columns (the "row" case), they could
#be in the same column but different rows (the "column"
#case), or they could be the same letter (the "same" case).
#
#Looking at the message PS HE LL OW OR LD SX:
#
# - PS is the Row case: P and S are in the same row.
# - HE is the Rectangle case: H and E are in different rows
#   and columns of the square.
# - LD is the Column case: L and D are in the same column.
# - LL is the Same case as it's two of the same letter.
#
#For the Same case, we replace the second letter in the pair
#with X, and then proceed as normal. When decrypting, it
#would be easy to see the our result was not intended to be
#PS HELXO WORLDSX, and we would thus assume the X is meant to
#repeat the previous letter, becoming PS HELLO WORLDSX.
#
#What we do for each of the other three cases is different:
#
# - For the Rectangle case, we replace each letter with
#   the letter in the same row, but the other letter's
#   column. For example, we would replace HE with GR:
#   G is in the same row as H but the same column as E,
#   and R is in the same row as E but the same column as
#   H. For another example, CS would become KL: K is in
#   C's row but S's column, and L is in C's column but S's
#   row.
# - For the Row case, we pick the letter to the right of
#   each letter, wrapping around the end of the row if we
#   need to. PS becomes QL: Q is to the right of P, and L
#   is to the right of S if we wrap around the end of the
#   row.
# - For the Column case, we pick the letter below each
#   letter, wrapping around if necessary. LD becomes TY:
#   T is below L and Y is below D.
#
#We would then return the resultant encrypted message.
#
#Decrypting a message is essentially the same process.
#You would use the exact same cipher and process, except
#for the Row and Column cases, you would shift left and up
#instead of right and down.
#
#Write two methods: encrypt and decrypt. encrypt should
#take as input a string, and return an encrypted version
#of it according to the rules above.
#
#To encrypt the string, you would:
#
# - Convert the string to uppercase.
# - Replace all Js with Is.
# - Remove all non-letter characters.
# - Add an X to the end if the length if odd.
# - Break the string into character pairs.
# - Replace the second letter of any same-character
#   pair with X (e.g. LL -> LX).
# - Encrypt it.
#
#decrypt should, in turn, take as input a string and
#return the unencrypted version, just undoing the last
#step. You don't need to worry about Js and Is, duplicate
#letters, or odd numbers of characters in decrypt.
#
#For example:
#
# encrypt("PS. Hello, world") -> "QLGRQTVZIBTYQZ"
# decrypt("QLGRQTVZIBTYQZ") -> "PSHELXOWORLDSX"
#
#HINT: You might find it easier if you implement some
#helper functions, like a find_letter function that
#returns the row and column of a letter in the cipher.
#
#HINT 2: Once you've written encrypt, decrypt should
#be trivial: try to think of how you can modify encrypt
#to serve as decrypt.
#
#To make this easier, we've gone ahead and created the
#cipher as a 2D tuple for you:
CIPHER = (("D", "A", "V", "I", "O"),
          ("Y", "N", "E", "R", "B"),
          ("C", "F", "G", "H", "K"),
          ("L", "M", "P", "Q", "S"),
          ("T", "U", "W", "X", "Z"))

def get_idx(string):
    res = []
    for letter in string: 
 
        for i in range(len(CIPHER)):
            for j in range(len(CIPHER)):
                if letter == CIPHER[i][j]: 
                    res.append((i,j))
    return res

def new_idx(indices, mode = "row"): 
    new_idx_w1 = None; new_idx_w2 = None
    
    if mode == "row":
        if indices[0][1] == 4: 
            new_idx_w1 = (indices[0][0], 0)
        else: 
            new_idx_w1 = (indices[0][0], indices[0][1]+1)
            
        if indices[1][1] == 4: 
            new_idx_w2 = (indices[1][0], 0)
        else: 
            new_idx_w2 = (indices[1][0], indices[1][1]+1)
        
        return [new_idx_w1, new_idx_w2]

    if mode == "column": 
        if indices[0][0] == 4: 
            new_idx_w1 = (0, indices[0][1])
        else: 
            new_idx_w1 = (indices[0][0]+1, indices[0][1])
        
        if indices[1][0] == 4: 
            new_idx_w2 = (0, indices[1][1])
        else: 
            new_idx_w2 = (indices[1][0]+1, indices[1][1])
        
        return [new_idx_w1, new_idx_w2]
    
    if mode == "rectangle": 
        new_idx_w1 = (indices[0][0], indices[1][1])
        new_idx_w2 = (indices[1][0], indices[0][1])
        return [new_idx_w1, new_idx_w2]
    

def old_idx(indices, mode = "row"): 
    new_idx_w1 = None; new_idx_w2 = None
    
    if mode == "row":
        if indices[0][1] == 0: 
            new_idx_w1 = (indices[0][0], 4)
        else: 
            new_idx_w1 = (indices[0][0], indices[0][1]-1)
            
        if indices[1][1] == 0: 
            new_idx_w2 = (indices[1][0], 4)
        else: 
            new_idx_w2 = (indices[1][0], indices[1][1]-1)
        
        return [new_idx_w1, new_idx_w2]

    if mode == "column": 
        if indices[0][0] == 0: 
            new_idx_w1 = (4, indices[0][1])
        else: 
            new_idx_w1 = (indices[0][0]-1, indices[0][1])
        
        if indices[1][0] == 0: 
            new_idx_w2 = (4, indices[1][1])
        else: 
            new_idx_w2 = (indices[1][0]-1, indices[1][1])
        
        return [new_idx_w1, new_idx_w2]
    
    if mode == "rectangle": 
        new_idx_w1 = (indices[0][0], indices[1][1])
        new_idx_w2 = (indices[1][0], indices[0][1])
        return [new_idx_w1, new_idx_w2] 
    
    


#Add your code here!
def encrypt(plaintext):
    List = []
    for char in plaintext:
        if char.isalpha(): 
            char = char.upper()
            if char == "J": char = "I"
            List.append(char)
        
    if len(List) % 2 == 1: List.append("X")

    final = []
    for i in range(0,len(List)-1,2): 
        if List[i] != List[i+1]: final.append(List[i] + List[i+1])
        else: final.append(List[i]+"X")
    
    res = ""
    for pair in final: 
        indices = get_idx(pair)
        new_idx_w1 = None; new_idx_w2 = None
        
        # row case
        if indices[0][0] == indices[1][0]: 
            new_idx_w1, new_idx_w2 = new_idx(indices)
        
        #colum case
        elif indices[0][1] == indices[1][1]:
            new_idx_w1, new_idx_w2 = new_idx(indices, mode="column")
        
        #rectangle case
        else: 
            new_idx_w1, new_idx_w2 = new_idx(indices, mode="rectangle")
            
        
        res += CIPHER[new_idx_w1[0]][new_idx_w1[1]]
        res += CIPHER[new_idx_w2[0]][new_idx_w2[1]]
    
    
    return res



def decrypt(string):
    res = ""
    pairs = []
    
    for i in range(0,len(string)-1, 2): 
        pairs.append(string[i]+string[i+1])
    
    for pair in pairs: 
        indices = get_idx(pair)
        new_idx_w1 = None; new_idx_w2 = None
        
        # row case
        if indices[0][0] == indices[1][0]: 
            new_idx_w1, new_idx_w2 = old_idx(indices)
        
        #colum case
        elif indices[0][1] == indices[1][1]:
            new_idx_w1, new_idx_w2 = old_idx(indices, mode="column")
        
        #rectangle case
        else: 
            new_idx_w1, new_idx_w2 = old_idx(indices, mode="rectangle")
            
        
        res += CIPHER[new_idx_w1[0]][new_idx_w1[1]]
        res += CIPHER[new_idx_w2[0]][new_idx_w2[1]]
    
    
    return res
    


#Below are some lines of code that will test your function.
#You can change the value of the variable(s) to test your
#function with different inputs.
#
#If your function works correctly, this will originally
#print: QLGRQTVZIBTYQZ, then PSHELXOWORLDSX
encrypt("PS. Hello, worlds")
decrypt("QLGRQTVZIBTYQZ")









