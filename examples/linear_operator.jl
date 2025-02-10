using ForwardDifferentiableExternalOperators
using ForwardDiff
using LinearOperators
using LinearAlgebra
using SparseDiffTools
using Chairmarks

# make this large so the linear operation dominates timings
N = 2048

# generate the operator
γ = rand(N, N);
LO = LinearOperator(γ);
FDLO = ForwardDifferentiableLinearOperator(LO)

# this is an in-place implementation of:
# f(x) = e^x * (γ * sin(x)) + x^3
# where using our ForwardDifferentiableLinearOperator for
# multiplication by γ
function f!(y, x, w, FDLO)
    ww = w(eltype(x))
    @. ww = sin(x)
    mul!(y, FDLO, ww)
    @. y *= exp(x)
    @. y += x^3
    return y
end
# a straightforward implementation of the operator
# to pass to ForwardDiff.jacobian for ground truth
function f(x, γ)
    return exp.(x) .* (γ * sin.(x)) .+ x.^3
end

# generate x, v, and a work vector
x = rand(N);
v = rand(N);
w = CacheVector(x);

# make a function that JacVec likes, and generate Jacobian-Vector product operator
g!(y, x) = f!(y, x, w, FDLO)
J = JacVec(g!, x, tag=nothing);
# test our JacVec
y1 = similar(x);
mul!(y1,J,v);
# compare to taking the Jacobian via ForwardDiff of the straightforward implementation
g(x) = f(x, γ)
y2 = ForwardDiff.jacobian(g, x) * v;
# are these the same?
display(y1 ≈ y2)

# now get a timing test of this version
_y = similar(x);
@b mul!($_y,$J,$v)

# we'll now test an auto-caching version
AFDLO = ForwardDifferentiableLinearOperator(LO, true)
g!(y, x) = f!(y, x, w, AFDLO)
J = JacVec(g!, x, tag=nothing);
y3 = similar(x);
# you MUST evaluate once at x
f!(_y, x, w, AFDLO);
mul!(y3,J,v);
# same as before?
display(y1 ≈ y3)

# now get a timing test, should be about twice as fast
@b mul!($y3,$J,$v)

# since the linear operator should dominate here, compare to
# the timing of a direct matrix-vector multiplication by γ
@b mul!($_y, $γ, $x)

