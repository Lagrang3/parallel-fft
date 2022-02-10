/*
    link with pFFT, Boost::MPI and OpenMPI
*/

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
namespace mpi = boost::mpi;
#include <exception>
#include <complex>
#include <pfft.h>

#include "cube_fill.hpp"

struct pfft_cube
{
    static std::size_t largest_divisor(std::size_t N)
    {
        std::size_t rootN{1};
        for(std::size_t n=1;n*n <= N;++n)
        {
            if(N%n == 0)
                rootN = n;
        }
        return rootN;
    }

    std::size_t global_len;
    std::array<std::size_t,2> proc_grid;
    cube_fill my_cube_fill;
    mpi::communicator my_com;
    
    MPI_Comm comm_cart_2d;
    pfft_plan my_plan;
    std::array<std::size_t,3> local_len,local_start;
    
    pfft_complex *my_data;
    
    pfft_cube(std::size_t N, const MPI_Comm& raw_com):
        global_len{N},
        my_cube_fill(N),
        my_com(raw_com,mpi::comm_duplicate)
    {
        proc_grid[0] = largest_divisor(my_com.size());
        proc_grid[1] = my_com.size()/proc_grid[0];
        
        pfft_init();
        if(int err = pfft_create_procmesh_2d(
            my_com,proc_grid[0],proc_grid[1],&comm_cart_2d); err!=0)
         {
            throw std::runtime_error("failed to create pfft cartesian communicator");
         }   
        ptrdiff_t local_ni[3], local_i_start[3];
        ptrdiff_t local_no[3], local_o_start[3];
        ptrdiff_t global_n[3];
        
        global_n[0] = global_n[1] = global_n[2] = global_len;
        
        const ptrdiff_t alloc_local =
            pfft_local_size_dft_3d(
                global_n,comm_cart_2d,
                0,
                local_ni,local_i_start,
                local_no,local_o_start);
        
        std::copy(local_ni,local_ni+3,local_len.begin());
        std::copy(local_i_start,local_i_start+3,local_start.begin());
        
        
        my_data = pfft_alloc_complex(alloc_local);
        
        my_plan =
            pfft_plan_dft_3d(
                global_n,
                my_data,my_data,comm_cart_2d,PFFT_FORWARD,PFFT_ESTIMATE);
        
        for(std::size_t i=0;i<local_len[0];++i)
            for(std::size_t j=0;j<local_len[1];++j)
            for(std::size_t k=0;k<local_len[2];++k)
            {
                std::complex<double> val 
                    = my_cube_fill(
                        local_start[0]+i,
                        local_start[1]+j,
                        local_start[2]+k);
                
                const ptrdiff_t pos = 
                    k + local_len[2]*( j + local_len[1]*i);
                if(pos>=alloc_local)
                    throw std::runtime_error("pos out of range");
                my_data[pos][0] = val.real();
                my_data[pos][1] = val.imag();
            }
        
    }
    
    void execute()
    {
        pfft_execute(my_plan);
    }
    
    ~pfft_cube()
    {
        pfft_free(my_data);
        pfft_destroy_plan(my_plan);
        pfft_cleanup();
    }    
};

int main()
{
    mpi::environment env;
    mpi::communicator world;
    pfft_cube cube(128,world);
    cube.execute();
    return 0;
}

