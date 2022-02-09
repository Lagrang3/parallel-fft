/*
    link with FFTW, Boost::MPI and OpenMPI
*/

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
namespace mpi = boost::mpi;
#include <complex>
#include <fftw3-mpi.h>

#include "cube_fill.hpp"

struct fftw_cube
{
    std::size_t global_len;
    cube_fill my_cube_fill;
    mpi::communicator my_com;
    
    fftw_plan my_plan;
    std::size_t local_len,local_start;
    
    fftw_complex *my_data;
    
    fftw_cube(std::size_t N, const MPI_Comm& raw_com):
        global_len{N},
        my_cube_fill(N),
        my_com(raw_com,mpi::comm_duplicate)
    {
        fftw_mpi_init();
        
        ptrdiff_t local_n0, local_0_start;
        ptrdiff_t alloc_local =
            fftw_mpi_local_size_3d(
                global_len,global_len,global_len,
                my_com,&local_n0,&local_0_start);
        
        local_len = local_n0;
        local_start = local_0_start;
            
        my_data = fftw_alloc_complex(alloc_local);
        
        for(std::size_t i=0;i<local_len;++i)
            for(std::size_t j=0;j<global_len;++j)
            for(std::size_t k=0;k<global_len;++k)
            {
                std::complex<double> val = my_cube_fill(local_start+i,j,k);
                
                const std::size_t pos = k + global_len*( j + global_len*i);
                my_data[pos][0] = val.real();
                my_data[pos][1] = val.imag();
            }
        
        my_plan =
        fftw_mpi_plan_dft_3d(
            global_len,global_len,global_len,
            my_data,my_data,my_com,FFTW_FORWARD,FFTW_ESTIMATE);
    }
    
    void execute()
    {
        fftw_execute(my_plan);
    }
    
    ~fftw_cube()
    {
        fftw_free(my_data);
        fftw_destroy_plan(my_plan);
        fftw_mpi_cleanup();
    }    
};

int main()
{
    mpi::environment env;
    mpi::communicator world;
    fftw_cube cube(128,world);
    cube.execute();
    return 0;
}
