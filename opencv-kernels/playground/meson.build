project('cguass', 'c')
pymod = import('python')
opencl = dependency('libclc')
py3=pymod.find_installation('python3', required:true, modules: ['numpy'])
py_dep = py3.dependency()

#nicked from sicpy lib MIT licensonce
incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
  incdir_numpy = run_command(py3,
    [
      '-c',
      '''import os
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
'''
    ],
    check: true
  ).stdout().strip()

  # We do need an absolute path to feed to `cc.find_library` below
  _incdir_numpy_abs = run_command(py3,
    ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
    check: true
  ).stdout().strip()
else
  _incdir_numpy_abs = incdir_numpy
endif

include_directories(incdir_numpy)
py3.extension_module('cguass', ['Main.c'])
