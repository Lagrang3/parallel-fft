#pragma once

#include <complex>

struct cube_fill
{
    using value_type = std::complex<double>;
    
    cube_fill( std::size_t /* N */)
    {}
    
    value_type operator() 
        (std::size_t /* x */, 
        std::size_t /* y */,
        std::size_t /* z */) const 
    {
        return {1,0};
    }
};
