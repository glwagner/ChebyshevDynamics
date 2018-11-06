module ChebyshevDynamics

export
  cheb,
  uni,
  wavevec,
  spectral_matrix,
  coeff!,
  value!,
  anti_deriv,

  icf_fft2,
  cf_fft2,
  icf_fft2!,
  cf_fft2!,

  Greens,
  Poisson,

  solve!

using
  FFTW,
  LinearAlgebra,
  Base.Threads

include("utils.jl")
include("odesolver.jl")
include("pdesolver.jl")

end # module
