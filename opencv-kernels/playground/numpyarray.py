from ctypes import *
import numpy as np

array = np.arange(100)

libcl = CDLL("./libHelloOpenCL.so")
libcl.tryout()
