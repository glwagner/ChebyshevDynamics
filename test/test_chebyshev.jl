include("chebyshev.jl")
using Pkg
using FFTW
using LinearAlgebra
using Plots


println("This tests functions in chebyshev.jl")

n=17
x = cheb(n)
y = exp.(2*x)
nw = nweights(n)
sw = sweights(n)

println("Testing Integral 0.5 int_{-1}^1 exp(2 x) dx")
println("computed")
println(dot(y,nw))
println("Should be")
println( (exp(2) - exp(-2))/4.0 )
#now in spectral space
coeff(y)
iv = dot(sw,y)
println("In spectral space we compute the integral to be $iv")

println("Testing endpoints")
ll = left(y)
el = exp(-2)
rr = right(y)
er = exp(2)
println("The left endpoint is computed to be $ll")
println("It should be $el")
println("The left endpoint is computed to be $rr")
println("It should be $er")


println("Testing derivative of endpoints")
ll = dleft(y)
el = 2*exp(-2)
rr = dright(y)
er = 2*exp(2)
println("The deriv of the left endpoint is computed to be $ll")
println("It should be $el")
println("The deriv of the right endpoint is computed to be $rr")
println("It should be $er")


println("testing the 2d transform")

n = 2^2
m = n
a = rand(n,n)
b = cf_fft2(a)
c = icf_fft2(b)
er = norm(a-c)/norm(c)
print("The error of computing the forward transform followed by the back transform is $er")
x = cheb(n)
y = uni(m)
a = zeros(n,m)
for j in 1:m
    for i in 1:m
        a[i,j] = x[i]*cos(y[j])
    end
end
print("The forward transform of x*cos(y)")
display(a)
println("is")
display(cf_fft2(a))






