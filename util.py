import numpy as np 

def decimal_to_base(decimal_num, base = 3, padding = 9):    
    base_num = ""
    
    if decimal_num == 0:
        # Special case for 0 in base
        base_num = "0"
    else: 
        while decimal_num > 0:
            remainder = decimal_num % base
            base_num  = str(remainder) + base_num
            decimal_num //= base

    base_string = base_num.zfill(padding)[::-1]
    base_array  = np.array([int(digit) for digit in base_string])

    return base_array