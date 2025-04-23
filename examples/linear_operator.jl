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
# out of place implementation of the same function
# again, using ForwardDifferentiableLinearOperator for
# multiplication by γ
function f(x, γ, FDLO)
    w = sin.(x)
    w = FDLO*w
    return @. exp(x) * w + x^3
end

# a straightforward implementation of the operator
# to pass to ForwardDiff.jacobian for ground truth
function direct_f(x, γ)
    return exp.(x) .* (γ * sin.(x)) .+ x.^3
end

# generate x, v, and a work vector
x = rand(N);
v = rand(N);
w = CacheVector(x);

# in place version using f! // FDLO
g!(y, x) = f!(y, x, w, FDLO)
J! = JacVec(g!, x, tag=nothing);
y1 = similar(x);
mul!(y1,J!,v);
# out of place version using f // FDLO
J = JacVec(x -> f(x, γ, FDLO), x);
y2 = J*v;
# compare to taking the Jacobian via ForwardDiff of the straightforward implementation
y3 = ForwardDiff.jacobian(x -> direct_f(x, γ), x) * v;
# are these the same?
display(y1 ≈ y3)
display(y2 ≈ y3)

# now get a timing test of this version
_y = similar(x);
@b mul!($_y,$J!,$v)

# we'll now test an auto-caching version
AFDLO = ForwardDifferentiableLinearOperator(LO, true)
g!(y, x) = f!(y, x, w, AFDLO)
J! = JacVec(g!, x, tag=nothing);
J = JacVec(x -> f(x, γ, AFDLO), x);
y4 = similar(x);
# you MUST evaluate once at x
f!(_y, x, w, AFDLO);
mul!(y4,J!,v);
f(x, γ, AFDLO);
y5 = J*v;
# same as before?
display(y3 ≈ y4)
display(y3 ≈ y5)

# now get a timing test, should be about twice as fast
@b mul!($y4,$J!,$v)
@b $J*$v

# since the linear operator should dominate here, compare to
# the timing of a direct matrix-vector multiplication by γ
@b mul!($_y, $γ, $x)

