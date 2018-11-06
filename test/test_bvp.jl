testcheb() = isapprox(cheb(4), [1, 0.5, -0.5, -1])
testuni() = isapprox(uni(5), [0, 1/5, 2/5, 3/5, 4/5])

function testcoeff()
  u = [1, 0.5, -0.5, -1]
  coeff!(u)
  isapprox(u, [0, 1, 0, 0])
end

function testvalue()
  a = [0, 1.0, 0, 0]
  value!(a)
  isapprox(a, [1, 0.5, -0.5, -1])
end

function testbvp(; n=2^10, k=1)
  x = cheb(n)
  sm1 = spectral_matrix(n, k)
  sm2 = spectral_matrix(n, -k)

  # Compute yh1
  v = zeros(n)
  v[2] = k/2
  yh1 = (sm1 \ v)
  yh2 = copy(yh1)
  yh2[1] += 1.0
  value!(yh1)
  @. yh1 = yh1 + 0.5

  # compute yh2
  yh2 = anti_deriv(yh2)
  yh2 = (sm2 \ yh2)
  value!(yh2)

  ytruth = zeros(n)
  fp = zeros(n)
  ycomp = zeros(n)

  # test, solve (D+k)(D-k)y = fp
  @. fp = -(π^2 + k^2) * sin(π*x) - x
  @. ytruth = sin(π*x) + x 

  # find particular solution
  coeff!(fp)
  yp = anti_deriv(fp)
  yp[n] = 0
  yp = sm1 \ yp
  yp = anti_deriv(yp)
  yp = sm2 \ yp
  value!(yp)

  # Find constants
  bcmat = zeros(2, 2)
  bcmat[1, 1] = yh1[n]
  bcmat[2, 1] = yh1[1]
  bcmat[1, 2] = yh2[n]
  bcmat[2, 2] = yh2[1]
  lhs = zeros(2)
  lhs[1] = ytruth[n] - yp[n]
  lhs[2] = ytruth[1] - yp[1]
  c = bcmat \ lhs

  # Solution.
  @. ycomp = c[1] * yh1 + c[2] * yh2 + yp

  isapprox(ycomp, ytruth)
end
