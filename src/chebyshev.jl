"""
    cheb(n)

Return the chebyshev gridpoints for grid size = n.

# Examples
julia> cheb(4)
4-element Array{Float64,1}:
  1.0               
  0.5000000000000001
 -0.4999999999999998
 -1.0  
"""
cheb(n) = cos.( (0:(n-1)) .* (π / (n-1)))

"""
    uni(n)

Return uniform gridpoints for grid size = n, [0,1).
appropriate for fourier transforms

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
    return kv
end



"""
requires FFTW
    plan(x)

Plan an in place transform for the vector x

# Examples

"""


function plan(x)
    FFTW.plan_r2r!(x,FFTW.REDFT00)
    return 0
end

"""
    coeff(x)

Return the chebyshev coeffficents of a function evaluated on the chebyshev grid. OVERWRITES x

# Examples
julia> x
4-element Array{Float64,1}:
  1.0               
  0.5000000000000001
 -0.4999999999999998
 -1.0               

julia> coeff(x)
4-element Array{Float64,1}:
  6.661338147750939e-16 
  3.0                   
 -3.3306690738754696e-16
  2.220446049250313e-16 
"""
function coeff(x)
    n = length(x)
    sc = 1 / (n-1)
    FFTW.r2r!(x,FFTW.REDFT00)
    for i = 1:n
        x[i] *= sc
    end
    return
end

"""
    coeff(x)

Return the chebyshev values of a function whose chebyshev coefficients are given. OVERWRITES x

# Examples
julia> x
4-element Array{Float64,1}:
  6.661338147750939e-16 
  3.0                   
 -3.3306690738754696e-16
  2.220446049250313e-16 

julia> value(x)

julia> x
4-element Array{Float64,1}:
  1.0               
  0.5000000000000001
 -0.4999999999999998
 -1.0       
"""

function value(x)
    n = length(x)
    sc = 0.5
    FFTW.r2r!(x,FFTW.REDFT00)
    for i = 1:n
        x[i] *= sc
    end
    return
end

        
    

"""
    spectral_matrix(x)

Creates the operator (I - integral k) in spectral space, factorizes, then returns.

# Examples
   

 hello 
"""

function spectral_matrix(n,k)
    up = zeros(n-1)
    lo = zeros(n-1)
    di = ones(n)
    for i = 2:n-2
        lo[i] = k / ( 2 * (i-1) )
        up[i] =  - k / ( 2 * i )
    end
    y = Tridiagonal(up,di,lo)
    return lu!(y)
end

"""
    anti_deriv(a)

Given the chebyshev coefficients, this operator computes an antiderivative and returns it
The zeroeth mode is given by zero

"""
function anti_deriv!(b, a)
    n = length(a)
    for i=2:(n-2)
        b[i] = (a[i-1] - a[i+1])*0.5 / (i-1)
    end
    b[n-1] = a[n-2] / (n-1.0)
    nothing
end

function anti_deriv(a)
    b = zeros(n)
    anti_deriv!(b, a)
    b
end


"""
    sweights(x)

spectral weights for computing integrals of certain quantities. Take the dot product with the chebyshev coefficients 
to get the integral.

"""

function sweights(n)
    sw = zeros(n)
    nh = div(n,2)
    x = (1:nh-1)*2
    norm = 1.0 #normalization factor for chebyshev transform
    for i in x
        sw[i+1] = norm / (1.0-i*i) 
    end
    sw[1] = 0.5 *  norm
    return sw
end


"""
nweights(x)

nodal weights for computing integrals of certain quantities. Take the dot product with the a function evaluated on the chebyshev node 
to get the integral.

"""

function nweights(n::Integer)
    nw = zeros(n)
    for i in 0:n-1
	t = i*pi/(n-1.0)
	tt = 0
	for j in 1:(n-2)
            jj = 1.0*j
	    tt += sin(jj*t)/jj*(1.0-cos(jj*pi))
        end
	tt *= sin(t)*2.0/(n-1.0);
        nw[i+1] = tt/2.0
    end
    return nw
end

"""
left(x)

Computes the left endpoint given the spectral coefficients

"""

function left(x)
    n = length(x)
    sum = x[1]/2
    for i in 2:(n-1)
        sum += x[i] * (-1)^(i-1)
    end
    sum += x[n] /2.0 * (-1)^(n-1)
    return sum
end

"""
dleft(x)

Computes the derivative of the left endpoint given the spectral coefficients

"""

function dleft(x)
    n = length(x)
    sum = 0
    for i in 2:(n-1)
        c = (i-1)
        sum += x[i] * (-1)^(i) * c * c
    end
    sum += x[n] /2.0 * (-1)^(n-1) * (n-1)^2
    return sum
end


"""
right(x)

Computes the right endpoint given the spectral coefficients

"""


function right(x)
    n = length(x)
    sum = x[1]/2
    for i in 2:(n-1)
        sum += x[i]
    end
    sum += x[n] /2.0
    return sum
end


"""
dright(x)

Computes the derivative of the right endpoint given the spectral coefficients

"""


function dright(x)
    n = length(x)
    sum = 0
    for i in 2:(n-1)
        c = i-1
        sum += x[i]*c*c
    end
    sum += x[n] /2.0 * (n-1)^2
    return sum
end

"""
computs the chebyshev fourier transform in one direction and 
"""

function cf_fft2(x)
    n = length(x[:,1])
    sc = 1.0 / (n-1)
    y = copy(x)
    FFTW.r2r!(y,FFTW.REDFT00,1)
    @. y = sc * y
    y = fft(y,2)
    return y
end

"""
computs the chebyshev fourier transform in one direction and fourier in the other
assumes data is real
"""

function icf_fft2(x)
    sc = 0.5
    y = real.(ifft(x,2))
    FFTW.r2r!(y,FFTW.REDFT00,1)
    @. y = sc * y
    return y
end

    

"""
computs the chebyshev transform in one direction and the fourier 
transform in the other
"""

function cf_fft2!(x)
    n = length(x[:,1])
    sc = 1.0 / (n-1)
    FFTW.r2r!(x,FFTW.REDFT00,1)
    x *= sc
    fft!(x,2)
    nothing
end

"""
computs the chebyshev fourier transform in one direction and fourier in the other
assumes data is real
"""

function icf_fft2!(x)
    sc = 0.5
    ifft!(x,2)
    FFTW.r2r!(x,FFTW.REDFT00,1)
    x *= sc
    return
end

function cf_fft2_p2(x)
    m,n = size(x)
    sc = 1.0 / (m-1)
    y = fft(x,2)
    return y
end


    
