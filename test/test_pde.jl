using 
  ChebyshevDynamics,
  LinearAlgebra,
  Base.Threads
  
# Threads.nthreads() = 1 #need to start julia with the same number of threads
# FFTW.set_num_threads(12)
# Here we attempt to solve the heat equation.

m = 2^9 + 1 # number of chebyshev grid points
n = 2^8 # number of uniform grid points

x = uni(n)  #domain is from 0 to 1 (no endpoint)
z = cheb(m) #domain is from -1 to 1
kv = wavevec(n)

tr_sol = zeros(m, n) #initialize the true solution
forcing = zeros(m, n) #initialize the forcing
scale =  -π^2 - 4π^2

@threads for j in 1:n
  tmp = exp(sin(2π*x[j]))
  s = sin(2π*x[j])
  c = cos(2π*x[j])
  scale =  -(π^2 + 4π^2*(s-c^2))
  @. tr_sol[:, j] = sin(π*z)*tmp
  @. forcing[:, j] = scale*tr_sol[:, j]
end

# Initialize by computing transform
fp = cf_fft2(forcing)
tmpr = zeros(m,n)
tmpi = zeros(m,n)
# -- done initializing.

# Utilize Poisson
p = Poisson(m, n, kv)

# Solve Poisson equation
@. tmpr = real(fp)
@. tmpi = imag(fp)
solve!(p, tmpr, tmpi, fp)

# Check solution
comp_sol = icf_fft2(fp)

norm(tr_sol-comp_sol) / norm(tr_sol)
