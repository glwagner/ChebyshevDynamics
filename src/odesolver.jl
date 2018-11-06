# Types
 a1 = typeof(spectral_matrix(4, 1))
 a2 = typeof(zeros(4))
a2c = typeof(zeros(2)*im)
 a3 = typeof(lu([1 2; 3 4]))

# Greens function for D^2-k^2
# implicitly constructed via two Tridiagonal matrices and
# two homogoneous solutions and the Dirichlet b.c. matrix
# can of course overwite the b.c. matrix for other boundary conditions

#=
This is the form we want to use eventually:
struct GreensFunction{Tm,Ty,Tbc}
  m1::Tm 
  m2::Tm 
  y1::Ty 
  y2::Ty 
  bc::Tbc
end

function GreensFunction(n::Int, k::AbstractFloat)
  # Construct object here.
end
=#

struct Greens
  m1::a1 # spectral coefficient matrix "1"
  m2::a1 # spectral coefficient matrix "2"
  y1::a2 # homogoenous solution "1"
  y2::a2 # homogoenous solution "2"
  bc::a3 # boundary condition matrix
end

function Greens(n::typeof(1),k::typeof(1.0))
  sm1 = spectral_matrix(n,k)         # spectral matrices
  sm2 = spectral_matrix(n,-k)        # spectral matrices

  yh1 = zeros(n)                     # homog solution 1
  yh1[2] = 0.5*k
  yh1 = sm1 \ yh1
  yh1[1] += 1
  yh2 = copy(yh1)                    # homog solution 2
  yh2 = anti_deriv(yh2)
  yh2 = sm2 \ yh2

  bcmat = zeros(2, 2)                # boundary condition matrix
  bcmat[1] = left(yh1)               # left endpoint for first row
  bcmat[2] = right(yh1)              # right endpoint for second row
  bcmat[3] = left(yh2)
  bcmat[4] = right(yh2)
  bcmat = lu(bcmat)                  # lu factorization

  Greens(sm1, sm2, yh1, yh2, bcmat)     # return values
end



# Solve with homogoneous boundary conditions
function solveh(g::Greens, fp::a2)
  yp = anti_deriv(fp)  #find the particular solution
  yp[end] = 0
  yp = g.m1 \ yp
  yp = anti_deriv(yp)
  yp = g.m2 \ yp

  lhs = zeros(2)      #satisfy the boundary conditions
  lhs[1] = - left(yp)
  lhs[2] = - right(yp)

  c = g.bc \ lhs

  @. yp = c[1] * g.y1 + c[2] * g.y2 + yp #construct solution from homog and particular solutions

  yp
end

# Solve with homogoneous boundary conditions with complex values for forcing
function solveh_c(g::Greens, fp::a2c)
  yp = anti_deriv(fp)  #find the particular solution
  yp[end] = 0
  yp = g.m1 \ yp
  yp = anti_deriv(yp)
  yp = g.m2 \ yp

  lhs = zeros(2)      #satisfy the boundary conditions
  lhs[1] = -left(yp)
  lhs[2] = -right(yp)

  c = g.bc \ lhs

  @. yp = c[1] * g.y1 + c[2] * g.y2 + yp #construct solution from homog and particular solutions

  yp
end

# Solve with homogoneous boundary conditions, and return derivative of solution
function solveh_wd(g::Greens, fp::a2)
  yp = anti_deriv(fp)  #find the particular solution
  yp[end] = 0

  ydp = g.m1 \ yp
  yp = anti_deriv(ydp)
  yp = g.m2 \ yp

  lhs = zeros(2)      #satisfy the boundary conditions
  lhs[1] = -left(yp)
  lhs[2] = -right(yp)

  c = g.bc \ lhs

  @. ydp = c[1] *(k * g.y1) + c[2] *(g.y1 - k * g.y2)  - k * yp + ydp
  @. yp = c[1] * g.y1 + c[2] * g.y2 + yp #construct solution from homog and particular solutions

  yp, ydp
end

# Solve with whatever boundary conditions, bc = [1, 0], means
# left boundary condition is 1 and right boundary condition is 0
function solve(g::Greens, fp::a2, bc::a2)
  yp = anti_deriv(fp)  #find the particular solution
  yp[end] = 0
  yp = g.m1 \ yp
  yp = anti_deriv(yp)
  yp = g.m2 \ yp

  lhs = zeros(2)      #satisfy the boundary conditions
  lhs[1] = bc[1] - left(yp)
  lhs[2] = bc[2] - right(yp)

  c = g.bc \ lhs

  @. yp = c[1] * g.y1 + c[2] * g.y2 + yp #construct solution from homog and particular solutions

  yp
end


# Solve with whatever boundary conditions, bc = [1, 0], means
# left boundary condition is 1 and right boundary condition is 0
# also gives the derivative of the solution
function solve_wd(g::Greens, fp::a2, bc::a2)
  yp = anti_deriv(fp)  #find the particular solution
  yp[end] = 0
  ydp = g.m1 \ yp
  yp = anti_deriv(ydp)
  yp = g.m2 \ yp

  lhs = zeros(2)      #satisfy the boundary conditions
  lhs[1] = bc[1]- left(yp)
  lhs[2] = bc[2]- right(yp)

  c = g.bc \ lhs

  @. ydp = c[1] *(k * g.y1) + c[2] *(g.y1 - k * g.y2)  - k * yp + ydp
  @. yp = c[1] * g.y1 + c[2] * g.y2 + yp #construct solution from homog and particular solutions

  yp, ydp
end
