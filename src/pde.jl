#include("ode.jl")
#include("chebyshev.jl")
#using FFTW
#using LinearAlgebra
#using Base.Threads

#this file sets up constructors for solving
#poissons equation: (\Delta - c)y = f 
#stokes equation (\Delta - c)\vec{u} = \nabla p + f
a2 = typeof(zeros(2))
a2c = typeof(zeros(2)*im)
a3 = typeof(zeros(2,2))
a3c = typeof(zeros(2,2)*im)
struct poisson
    sg::Array{greens,1}
    m::typeof(1)
    n::typeof(1)
    function poisson(m::typeof(1),n::typeof(1),kv::a2)
        sg = [greens(m,kv[j]) for j in 1:n];
        new(sg,m,n)         #return values
    end
end

#solves poissons equation in spectral space
#tmpr and tmpi are the real and imaginary compnents of fp
function solve!(p::poisson,tmpr::a3,tmpi::a3,fp::a3c)
    for j in 1:(div(p.n,2)+1)
        tmp1 = solveh(p.sg[j], tmpr[:,j])
        tmp2 = solveh(p.sg[j], tmpi[:,j])
        @. fp[:,j] =  tmp1 + tmp2*im
    end
    for j in (div(p.n,2)+2):n
        fp[:,j] = conj.(fp[:,n+2-j])
    end
end

#solves poissons equation in spectral space, parallel
#tmpr and tmpi are the real and imaginary compnents of fp
function solve_p!(p::poisson,tmpr::a3,tmpi::a3,fp::a3c)
    @threads for j in 1:(div(p.n,2)+1)
        tmp1 = solveh(p.sg[j], tmpr[:,j])
        tmp2 = solveh(p.sg[j], tmpi[:,j])
        @. fp[:,j] =  tmp1 + tmp2*im
    end
    @threads for j in (div(p.n,2)+2):n
        fp[:,j] = conj.(fp[:,n+2-j])
    end
end

#solves poissons eqwuation in real space
function solve_r!(p::poisson,f::a3)
    fp = cf_fft2(f)
    tmpr = real.(fp)
    tmpi = imag.(fp)
    for j in 1:(div(p.n,2)+1)
        tmp1 = solveh(p.sg[j], tmpr[:,j])
        tmp2 = solveh(p.sg[j], tmpi[:,j])
        @. fp[:,j] =  tmp1 + tmp2*im
    end
    for j in (div(p.n,2)+2):n
        fp[:,j] = conj.(fp[:,n+2-j])
    end
    f = icf_fft2(fp)
    return
end

