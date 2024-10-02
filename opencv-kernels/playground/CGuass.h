#pragma once
// C standard includes
// OpenCL includes
#include <CL/cl.h>
#include <CL/cl_platform.h>

// clBLAST matrix for efficient matrix ops
#include <clblast_c.h>

// python 
#include <python3.12/Python.h>

// numpy includes
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

// error helper function
#include "err_code.h"


const char RBF_KERNEL_SOURCE[] = ""; 

void CLSetupAndRun() {
  cl_uint err;
  cl_uint num_platforms;
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  checkError(err, "Finding Platforms");
  
  if(num_platforms == 0){
    fprintf(stderr, "No platforms found please install a openCL vendor");
  }
}

// should return matrix of cl_floats in CLBLAST, will change in future
cl_float* ConverNDArray(PyArrayObject *ndarray){
  // placeholder
  return NULL;
}

PyArrayObject* ToNDArray(cl_float* cmatrix) {
  // placeholder
  return NULL;
}

PyObject* RBFInterpolateCL(PyArrayObject *dummy, PyObject *args) {
  /* convert Python arguments */
  /* do function */
  /* return something */
  // placeholder
  return NULL;
}
