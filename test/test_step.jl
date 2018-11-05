include("step.jl")
include("pde.jl")
include("ode.jl")
include("chebyshev.jl")
using FFTW
using LinearAlgebra

using Base.Threads
#Threads.nthreads() = 1 #need to start julia with the same number of threads
#FFTW.set_num_threads(12)
#here we attempt to solve the heat equation in a domain
m = 2^9 + 1 #number of chebyshev grid points
n = 2^8 #number of uniform grid points
dt = 0.01
ll = 10000
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
x1 = zeros(m,n)*im #initial condition
sc =  -(pi^2 + 4*pi^2)
for j in 1:n
    tmp = exp(sin(2*pi*x[j]))
    s = sin(2*pi*x[j])
    c = cos(2*pi*x[j])
    sc =  -(pi^2 + 4*pi^2*(s-c^2))
    @. tr_sol[:,j] = sin( pi * z)*tmp
    @. forcing[:,j] = -sc*tr_sol[:,j]
end


#compute transform
fp = cf_fft2(forcing);
tmpr = zeros(m,n);
tmpi = zeros(m,n);
###########################################################Done initializing
#now utilize poisson class
ps = poisson_step(m,n,kv,dt,1.0)

#solve poissons equation
@. tmpr = real(fp);
@. tmpi = imag(fp);

println("the first order solver takes")
@time for k in 1:ll
    y = step1(ps,x1,fp)
    @. x1 = y
end
println("to take $ll timesteps")
comp_sol = icf_fft2(x1);
er=norm(tr_sol-comp_sol)/norm(tr_sol)
println("The error is now")
println(er)

println("Now for the second order solver")
x1 *= 0
y = step1(ps,x1,fp)
x2 = copy(x1)
@. x1 = y
f2 = copy(fp)
ps = poisson_step(m,n,kv,dt,1.5)
@time for k in 1:ll
    y = step2(ps,x1,x2,fp,f2)
    @. x2 = x1
    @. x1 = y
end
println("to take $ll timesteps")
comp_sol = icf_fft2(x1);
er=norm(tr_sol-comp_sol)/norm(tr_sol)
println("The error is now")
println(er)

println("Now for the third order solver")
x1 *= 0
y = step1(ps,x1,fp)
x3 = copy(x1)
@. x2 = x1
@. x1 = y
f2 = copy(fp)
f3 = copy(fp)
y = step2(ps,x1,x2,fp,f2)
@. x2 = x1
@. x1 = y
ps = poisson_step(m,n,kv,dt,11.0/6.0)
@time for k in 1:ll
    y = step3(ps,x1,x2,x3,fp,f2,f3)
    @. x3 = x2
    @. x2 = x1
    @. x1 = y
end
println("to take $ll timesteps")
comp_sol = icf_fft2(x1);
er=norm(tr_sol-comp_sol)/norm(tr_sol)
println("The error is now")
println(er)

