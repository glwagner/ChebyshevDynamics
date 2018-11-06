#include("ode.jl")
#include("chebyshev.jl")
#using FFTW
#using LinearAlgebra
#using Base.Threads

#this file sets up constructors for solving
#Poissons equation: (\Delta - c)y = f 
#stokes equation (\Delta - c)\vec{u} = \nabla p + f
 a2 = Array{Float64,1}
a2c = Array{Complex{Float64},1}
 a3 = Array{Float64,2}
a3c = Array{Complex{Float64},2}

struct Poisson
  sg::Array{Greens,1}
  m::Int
  n::Int
end

function Poisson(m::Int, n::Int, kv::a2)
 sg = [ Greens(m, kv[j]) for j in 1:n ]
 Poisson(sg, m, n)         #return values
end


""" 
Solve Poissons equation in spectral space.
`tmpr` and `tmpi` are the real and imaginary compnents of fp.
"""
function solve!(p::Poisson, tmpr::a3, tmpi::a3, fp::a3c)
  for j in 1:(div(p.n,2)+1)
    tmp1 = solveh(p.sg[j], tmpr[:, j])
    tmp2 = solveh(p.sg[j], tmpi[:, j])
    @. fp[:, j] = tmp1 + im*tmp2
  end

  for j in (div(p.n, 2)+2):p.n
    fp[:, j] = conj.(fp[:, p.n+2-j])
  end

  nothing
end

#solves Poissons equation in spectral space, parallel
#tmpr and tmpi are the real and imaginary compnents of fp
function solve_p!(p::Poisson, tmpr::a3, tmpi::a3, fp::a3c)
  @threads for j in 1:(div(p.n,2)+1)
    tmp1 = solveh(p.sg[j], tmpr[:,j])
    tmp2 = solveh(p.sg[j], tmpi[:,j])
    @. fp[:,j] =  tmp1 + tmp2*im
  end
  @threads for j in (div(p.n,2)+2):n
    fp[:,j] = conj.(fp[:,n+2-j])
  end

  nothing
end

#solves Poissons eqwuation in real space
function solve_r!(p::Poisson, f::a3)
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

  nothing
end

