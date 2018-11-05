module ChebyshevDynamics

export
  cheb,
  uni,
  cf_fft2,
  poisson,
  solve!,
  icf_fft2

using
  FFTW,
  LinearAlgebra,
  Base.Threads

include("chebyshev.jl")
include("ode.jl")
include("pde.jl")

end # module
