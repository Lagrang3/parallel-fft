project('Parallel FFT','cpp','c',
        default_options : ['cpp_std=c++17','warning_level=3','optimization=3'])

boost=dependency('boost',modules : ['mpi'])
mpi = dependency('mpi',language: 'cpp')
hdf5 = dependency('hdf5')
fftw = dependency('fftw3')
pfft = dependency('pfft')

cube_headers = files(['cube_fill.hpp'])

executable('fftw-cube','fftw-cube.cpp',cube_headers,
    dependencies: [boost,mpi,fftw,hdf5])
executable('pfft-cube','pfft-cube.cpp',cube_headers,
    dependencies: [boost,mpi,fftw,pfft,hdf5])
