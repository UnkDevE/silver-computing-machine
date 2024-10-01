#pragma once 
// C standard includes
#include <CL/cl_platform.h>
#include <stdio.h>

// OpenCL includes
#include <CL/cl.h>

#include <python3.12/Python.h>

static PyObject* nokeyword (PyArrayObject *dummy, PyObject *args)
{
    /* convert Python arguments */
    /* do function */
    /* return something */
    return NULL;
}

int tryout()
{
    cl_int CL_err = CL_SUCCESS;
    cl_uint numPlatforms = 0;

    CL_err = clGetPlatformIDs( 0, NULL, &numPlatforms );

    if (CL_err == CL_SUCCESS)
        printf("%u platform(s) found\n", numPlatforms);
    else
        printf("clGetPlatformIDs(%i)\n", CL_err);

    return 0;
}
