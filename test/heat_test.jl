#include("chebyshev.jl")
#include("ode.jl")
using Base.Threads
#Threads.nthreads() = 1 #need to start julia with the same number of threads
#FFTW.set_num_threads(12)
#here we attempt to solve the heat equation in a domain
m = 2^10 + 1 #number of chebyshev grid points
n = 2^9 #number of uniform grid points

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
    tmp = sin(2*pi*x[j])
    @. tr_sol[:,j] = sin( pi * z)*tmp
    @. forcing[:,j] = sc*tr_sol[:,j]
end

#compute transform
fp = cf_fft2(forcing);
tmpr = zeros(m,n);
tmpi = zeros(m,n);
@. tmpr = real(fp);
@. tmpi = imag(fp);
#compute greens function for the domain
sg = [greens(m,kv[j]) for j in 1:n];
@threads for j in 1:(div(n,2)+1)
    tmp1 = solveh(sg[j], tmpr[:,j])
    tmp2 = solveh(sg[j], tmpi[:,j])
    @. fp[:,j] =  tmp1 + tmp2*im
end
@threads for j in (div(n,2)+2):n
    fp[:,j] = conj.(fp[:,n+2-j])
end

comp_sol = icf_fft2(fp);
norm(tr_sol-comp_sol)/norm(tr_sol)


#=

for j in 1:n
    for i in 1:m
        tmpr[i] = real.(fp[i,j])
        tmpi[i] = imag.(fp[i,j])
    end
end

for j in 1:n
    for i in 1:m
        tmpr[i] = real(fp[i,j])
        tmpi[i] = imag(fp[i,j])
    end
end


for j in 1:n
    tmp1 = solveh(sg[j], tmpr[:,j])
    tmp2 = solveh(sg[j], tmpi[:,j])
end

for j in 1:n
    @. fp[:,j] =  tmpr[:,j] + tmpi[:,j]*im
end

for j in 1:n
    tmp1 = solveh(sg[j], tmpr[:,j])
end


for j in 1:n
    for i in 1:m
        tr_sol[i,j] = 0.0
        forcing[i,j] = 0.0
    end
end

terrible
for j in 1:n
    for i in 1:m
        tr_sol[i,j] = sin( pi * z[i])*sin(2*pi*x[j])
        forcing[i,j] = sc*tr_sol[i,j]
    end
end
@threads for j in 1:n
     for i in 1:m
        tr_sol[i,j] = sin( pi * z[i])*sin(2*pi*x[j])
        forcing[i,j] = sc*tr_sol[i,j]
    end
end

ll = 1000
@time for k in 1:ll
    for j in 1:(div(n,2)+1)
        tmp1 = solveh(sg[j], tmpr[:,j])
        tmp2 = solveh(sg[j], tmpi[:,j])
        @. fp[:,j] =  tmp1 + tmp2*im
    end
    for j in (div(n,2)+2):n
        fp[:,j] = conj.(fp[:,n+2-j])
    end
end
=#

