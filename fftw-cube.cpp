/*
    link with FFTW, Boost::MPI and OpenMPI
*/

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
namespace mpi = boost::mpi;
#include <complex>
#include <string>
#include <fftw3-mpi.h>
#include <hdf5.h>

#include "cube_fill.hpp"

struct fftw_cube
{
    std::size_t global_len;
    cube_fill my_cube_fill;
    mpi::communicator my_com;
    
    fftw_plan my_plan;
    std::size_t local_len,local_start;
    
    fftw_complex *my_data;
    
    hid_t complex_id;
    
    fftw_cube(std::size_t N, const MPI_Comm& raw_com):
        global_len{N},
        my_cube_fill(N),
        my_com(raw_com,mpi::comm_duplicate)
    {
        {
            fftw_complex tmp;
            complex_id = H5Tcreate(H5T_COMPOUND,sizeof(tmp));
            H5Tinsert(complex_id,"real",0,H5T_NATIVE_DOUBLE);
            H5Tinsert(complex_id,"imag",sizeof(double),H5T_NATIVE_DOUBLE);
        }
    
        fftw_mpi_init();
        
        ptrdiff_t local_n0, local_0_start;
        ptrdiff_t alloc_local =
            fftw_mpi_local_size_3d(
                global_len,global_len,global_len,
                my_com,&local_n0,&local_0_start);
        
        local_len = local_n0;
        local_start = local_0_start;
            
        my_data = fftw_alloc_complex(alloc_local);
        
        my_plan =
        fftw_mpi_plan_dft_3d(
            global_len,global_len,global_len,
            my_data,my_data,my_com,FFTW_FORWARD,FFTW_ESTIMATE);
        
        for(std::size_t i=0;i<local_len;++i)
            for(std::size_t j=0;j<global_len;++j)
            for(std::size_t k=0;k<global_len;++k)
            {
                auto val = my_cube_fill(local_start+i,j,k);
                
                const std::size_t pos = k + global_len*( j + global_len*i);
                my_data[pos][0] = val.real();
                my_data[pos][1] = val.imag();
            }
        
    }
    
    void save_to_file(std::string path)
    {
        hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
        MPI_Info info = MPI_INFO_NULL;
        H5Pset_fapl_mpio(plist_id,my_com,info);
        hid_t file_id = H5Fcreate(path.c_str(),H5F_ACC_TRUNC,H5P_DEFAULT,plist_id);
        H5Pclose(plist_id);
        
        constexpr hsize_t DIM = 3;
        hsize_t count[DIM],offset[DIM];
        for(auto i=0U;i<DIM;++i)
            count[i] = global_len;
            
        hid_t filespace = H5Screate_simple(DIM,count,NULL);
        hid_t dset_id = H5Dcreate(file_id, "My Dataset", complex_id, filespace,
            H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
        H5Sclose(filespace);
        
        count[0]=local_len;
        count[1]=global_len;
        count[2]=global_len;
        offset[0]=local_start;
        offset[1]=0;
        offset[2]=0;
        hid_t memspace = H5Screate_simple(DIM,count,NULL);
        
        filespace = H5Dget_space(dset_id);
        H5Sselect_hyperslab(filespace,H5S_SELECT_SET,offset,NULL,count,NULL);
        
        plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id,H5FD_MPIO_COLLECTIVE);
        
        herr_t status = H5Dwrite(dset_id, complex_id, memspace, filespace,
            plist_id, my_data);
        
        H5Dclose(dset_id);
        H5Sclose(filespace);
        H5Sclose(memspace);
        H5Pclose(plist_id);
        H5Fclose(file_id);
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
    fftw_cube cube(8,world);
    cube.save_to_file("fftw.h5");
    cube.execute();
    cube.save_to_file("fftw-ft.h5");
    return 0;
}
