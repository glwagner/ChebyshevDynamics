#include("pde.jl")
#include("ode.jl")
#include("chebyshev.jl")
using 
  ChebyshevDynamics,
  Base.Threads
  
#Threads.nthreads() = 1 #need to start julia with the same number of threads
#FFTW.set_num_threads(12)
#here we attempt to solve the heat equation in a domain

m = 2^9 + 1 #number of chebyshev grid points
n = 2^8 #number of uniform grid points

z = cheb(m) #domain is from -1 to 1
x = uni(n)  #domain is from 0 to 1 (no endpoint)
kv = zeros(n) #wavevector
for j in 1:div(n,2)
    kv[j] = 2.0*pi*(j-1)
end
for j in (div(n,2)+1):n
    kv[j] = (j-n-1)*2.0*pi
end
#display(kv)


tr_sol = zeros(m,n) #initialize the true solution
forcing = zeros(m,n) #initialize the forcing
sc =  -(pi^2 + 4*pi^2)
@threads for j in 1:n
    tmp = exp(sin(2*pi*x[j]))
    s = sin(2*pi*x[j])
    c = cos(2*pi*x[j])
    sc =  -(pi^2 + 4*pi^2*(s-c^2))
    @. tr_sol[:,j] = sin( pi * z)*tmp
    @. forcing[:,j] = sc*tr_sol[:,j]
end

#compute transform
fp = cf_fft2(forcing);
tmpr = zeros(m,n);
tmpi = zeros(m,n);
###########################################################Done initializing
#now utilize poisson class
p = poisson(m,n,kv)

#solve poissons equation
@. tmpr = real(fp);
@. tmpi = imag(fp);
solve!(p,tmpr,tmpi,fp)

#check solution
comp_sol = icf_fft2(fp);

norm(tr_sol-comp_sol) / norm(tr_sol)
