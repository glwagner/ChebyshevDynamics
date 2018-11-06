"""
    cheb(n)

Return the Chebyshev gridpoints for grid size = n.

# Examples
julia> cheb(4)
4-element Array{Float64,1}:
  1.0               
  0.5000000000000001
 -0.4999999999999998
 -1.0  
"""
cheb(n) = cos.( (0:(n-1)) .* π/(n-1) )

"""
    uni(n)

Return uniform gridpoints for grid size = n, [0,1).

# Examples
julia> uni(4)
0.0:1.5707963267948966:4.71238898038469
"""
uni(n) = (0:(n-1)) ./ n

"""
    wavevec(n)

Return wavevector corresponding to uniform gridpoints for grid size = n, [0,1).
appropriate for fourier transforms

# Examples
julia> wavevec(8)
8-element Array{Float64,1}:
   0.0              
   6.283185307179586
  12.566370614359172
  18.84955592153876 
 -25.132741228718345
 -18.84955592153876 
 -12.566370614359172
  -6.283185307179586
"""
function wavevec(n)
  kv = zeros(n)
  for j in 1:div(n,2)
    kv[j] = 2π*(j-1)
  end
  for j in (div(n,2)+1):n
    kv[j] = (j-n-1)*2π
  end
  kv
end



"""
requires FFTW

    plan(x)

Plan an in place transform for the vector x

# Examples

"""
function plan(x)
  FFTW.plan_r2r!(x, FFTW.REDFT00)
  nothing
end

"""
    coeff!(u)

Calculate the Chebyshev coefficients of the function `f` and store in `f`.

# Examples
julia> u
4-element Array{Float64,1}:
  1.0               
  0.5000000000000001
 -0.4999999999999998
 -1.0               

julia> coeff(u)
4-element Array{Float64,1}:
  6.661338147750939e-16 
  1.0                   
 -3.3306690738754696e-16
  2.220446049250313e-16 
"""
function coeff!(u)
  n = length(u)
  scale = 1 / (n-1)
  FFTW.r2r!(u, FFTW.REDFT00)
  @. u *= scale
  nothing
end

"""
    value!(a)

Calculate the Chebyshev values of `a` and store in `a`.

# Examples
julia> a
4-element Array{Float64,1}:
  6.661338147750939e-16 
  3.0                   
 -3.3306690738754696e-16
  2.220446049250313e-16 

julia> value(a)

julia> a
4-element Array{Float64,1}:
  1.0               
  0.5000000000000001
 -0.4999999999999998
 -1.0       
"""

function value!(a)
  scale = 0.5
  FFTW.r2r!(a, FFTW.REDFT00)
  @. a *= scale
  nothing
end

"""
    spectral_matrix(x)

Returns the LU factorication of `I - k ∫`.
"""
function spectral_matrix(n, k)
  upper = zeros(n-1)
  lower = zeros(n-1)
  diag = ones(n)
  for i = 2:n-2
    lower[i] = k / (2*(i-1))
    upper[i] =  -k / 2i
  end
  y = Tridiagonal(upper, diag, lower)
  return lu!(y)
end

"""
    anti_deriv!(b, a)

Given Chebyshev coefficients `a`, compute the anti-derivative and store result in `b`
with the zeroth mode set to 0.
"""
function anti_deriv!(b, a)
  n = length(a)
  for i = 2:n-2
    b[i] = (a[i-1] - a[i+1]) * 0.5/(i-1)
  end
  b[n-1] = a[n-2] / (n-1)
  nothing
end

function anti_deriv(a)
  b = zero(a)
  anti_deriv!(b, a)
  b
end


"""
    sweights(n)

Returns the spectral weights for computing integrals of certain quantities
on a Chebyshev grid with length `n`. 
"""
function sweights(n::Int)
  sw = zeros(n)
  nh = div(n, 2)
  norm = 1.0 #normalization factor for chebyshev transform
  for i = 2:2nh-2
    sw[i+1] = norm / (1-i^2) 
  end
  sw[1] = 0.5 * norm
  sw
end


"""
    nweights(n)

Returns the nodal weights for computing integrals of certain quantities
on a Chebyshev grid of length `n`.
"""
function nweights(n::Int)
  nw = zeros(n)
  for i = 0:n-1
	  t = i*π/(n-1)
	  tt = 0
	  for j in 1:(n-2)
	    tt += sin(j*t)/j * (1-cos(π*j))
    end
    tt *= sin(t) * 2/(n-1)
    nw[i+1] = 0.5*tt
  end

  nw
end

"""
    left(a)

Returns the value of a function at its left end point, given its Chebyshev coefficients `a`.
"""

function left(a)
  n = length(a)

  sum = 0.5*a[1]
  for i = 2:(n-1)
    sum += a[i] * (-1)^(i-1)
  end
  sum += 0.5*a[n] * (-1)^(n-1)

  sum
end

"""
    dleft(a)

Returns the derivative of a function at its left endpoint, given its Chebyshev coefficients `a`.
"""
function dleft(a)
  n = length(a)
  sum = 0
  for i in 2:(n-1)
      sum += (-1)^i * (i-1)^2 * a[i]
  end
  sum += 0.5*a[n] * (-1)^(n-1) * (n-1)^2

  sum
end


"""
    right(a)

Returns the value of a function at its right endpoint given its Chebyshev coefficients `a`.
"""
function right(a)
  n = length(a)

  sum = 0.5*a[1]
  for i = 2:n-1
      sum += a[i]
  end
  sum += 0.5*a[n]

  sum
end


"""
    dright(x)

Returns the derivative of a function at its left endpoint, given its Chebyshev coefficients `a`.
"""
function dright(a)
  n = length(a)

  sum = 0
  for i in 2:n-1
    sum += (i-1)^2 * a[i]
  end
  sum += 0.5*a[n] * (n-1)^2

  sum
end

"""
Computes the chebyshev fourier transform in one direction and 
"""
function cf_fft2(x)
  n = length(x[:,1])
  scale = 1/(n-1)
  y = copy(x)
  FFTW.r2r!(y,FFTW.REDFT00, 1)
  @. y *= scale
  y = fft(y, 2)
  y
end

"""
computs the chebyshev fourier transform in one direction and fourier in the other
assumes data is real
"""
function icf_fft2(x)
  scale = 0.5
  y = real.(ifft(x, 2))
  FFTW.r2r!(y, FFTW.REDFT00, 1)
  @. y *= scale
  y
end

"""
computs the chebyshev transform in one direction and the fourier 
transform in the other
"""
function cf_fft2!(x)
  n = size(x, 1)
  scale = 1/(n-1)
  FFTW.r2r!(x, FFTW.REDFT00, 1)
  x *= scale
  fft!(x, 2)
  nothing
end

"""
computs the chebyshev fourier transform in one direction and fourier in the other
assumes data is real
"""

function icf_fft2!(x)
  scale = 0.5
  ifft!(x, 2)
  FFTW.r2r!(x, FFTW.REDFT00, 1)
  x *= scale
  nothing
end

function cf_fft2_p2(x)
  m = size(x, 1)
  sc = 1/(m-1)
  y = fft(x, 2)
  y
end
