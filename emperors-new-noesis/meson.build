project('emperors-new-noesis', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++17'] )

# dependents
arrayfire = dependency('ArrayFire')
onnx = dependency('ONNX')

# libraries
poly = library('polynomial', 'src/polynomial/main.cpp', dependencies: [arrayfire, onnx])
four = library('fourier', 'src/fourier/main.cpp', link_with: [poly])

# exec
exe = executable('emperors_new_noesis', 'src/main.cpp',
  install : true, link_with: [poly, four])

test('basic', exe)
