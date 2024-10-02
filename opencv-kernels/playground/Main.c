#define CGUASS_MODULE
#include "CGuass.h"

/* This initiates the module using the above definitions. */
static struct PyModuleDef moduledef = {
};

PyMODINIT_FUNC PyInit_CGuass(void) {
  PyObject *m;
  m = PyModule_Create(&moduledef);
  if (!m) {
    return NULL;
  }
  return m;
}
