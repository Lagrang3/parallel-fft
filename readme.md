Parallel FFT is a repository for testing and benchmarking parallel
multidimensional Discrete Fourier Transforms libraries based on MPI.

We want to test and compare three aspects of each library:
1. correctedness,
2. performance,
3. interface,
4. accuracy.

The aim of this project is to kickstart a baseline code in C++ that will
eventually be ported to boost::mpi library in collaboration with
boost::math::fft. And phase out LATfield as the core of gevolution.

We will test FFTW (http://fftw.org/), LATfield
(https://github.com/daverio/LATfield2) and pFFT (https://github.com/mpip/pfft).

Why do we need another parallel FFT library?

1. FFTW and pFFT are written in C, and there is no C++ interface with support
for templates, strong type checking and exception safety.
LATfield is written in C++, but it is badly design, very hard to use and to
maintain.
2. FFTW does not scale for dimensions higher than 2, because the domain
decomposition is along one dimension. LATfield and pFFT do solve the problem.
3. pFFT is no longer maintained since 2018.
4. LATfield's developers do not accept changes to their bad design choices.
5. We want to provide a clean interface for real to complex transforms instead
of the complicated concept sold by FFTW and re-used in LATfield and pFFT.

In essence we need a C++ library (better designed than LATfield) and with a
generalized N-dimensional approach to FFT like pFFT.
At the core of this library we will be using FFTW to perform one dimensional
FFTs.

We expect to achieve a performance competitive with those of FFTW, pFFT and
LATfield (if any).
