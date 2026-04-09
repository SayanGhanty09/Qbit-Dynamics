from vision import *
import numpy as np

# We provide fake inputs
# current_mask shape: (240, 640)
# w_roi = 640
def test_calc(x, w):
    cx = x + w // 2
    w_roi = 640
    if cx < w_roi // 3:
        return 'left'
    elif cx > 2 * w_roi // 3:
        return 'right'
    else:
        return 'center'

print("Test: x=10, w=50 ->", test_calc(10, 50))
print("Test: x=300, w=50 ->", test_calc(300, 50))
print("Test: x=500, w=50 ->", test_calc(500, 50))
