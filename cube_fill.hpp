#pragma once

#include <complex>
#include <cmath>

struct cube_fill
{
    using value_type = std::complex<double>;
    
    cube_fill( std::size_t /* N */)
    {}
    
    value_type operator() 
        (std::size_t x, 
        std::size_t y,
        std::size_t z) const 
    {
        return {std::exp(0.-0.5*(x*x+y*y+z*z)),0};
    }
};
