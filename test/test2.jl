include("chebyshev.jl")
using Pkg
using FFTW
using LinearAlgebra
using Plots
#gr()

println("tests a second order boundary value solver")
n = 2^10
k = 1
x = cheb(n)
sm1 = spectral_matrix(n,k)
sm2 = spectral_matrix(n,-k)

#compute yh1
v = zeros(n)
v[2] = k/2
yh1 = (sm1 \ v)
yh2 = copy(yh1)
yh2[1] += 1.0
value(yh1)
@. yh1 = yh1 + 0.5
#compute yh2
yh2 = anti_deriv(yh2)
yh2 = (sm2 \ yh2)
value(yh2)
#GR.plot(x,yh)

ytruth = zeros(n)
fp = zeros(n)
ycomp = zeros(n)
#test, solve (D+k)(D-k)y = fp
@. ytruth = sin( pi * x) + x 
@. fp = -(pi^2 + k^2) * sin( pi * x) - x
#find particular solution
coeff(fp)
yp = anti_deriv(fp)
yp[n] = 0
yp = sm1 \ yp
yp = anti_deriv(yp)
yp = sm2 \ yp
value(yp)
#Find constants
bcmat = zeros((2,2))
bcmat[1] = yh1[n]
bcmat[2] = yh1[1]
bcmat[3] = yh2[n]
bcmat[4] = yh2[1]
lhs = zeros(2)
lhs[1] = ytruth[n] - yp[n]
lhs[2] = ytruth[1] - yp[1]
c = bcmat \ lhs
@. ycomp = c[1] * yh1 + c[2] * yh2 + yp

#compare truth
plot(x,ycomp - ytruth)
