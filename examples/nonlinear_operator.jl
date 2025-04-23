using Revise
using ForwardDifferentiableExternalOperators
using ForwardDiff
using LinearOperators
using LinearAlgebra
using SparseDiffTools
using Chairmarks

N = 1024

struct MyOperator
    N::Int
end
Base.size(op::MyOperator) = (op.N, op.N)
Base.size(op::MyOperator, i::Int) = op.N
Base.eltype(op::MyOperator) = Float64
function LinearAlgebra.mul!(
    y::AbstractVector{T},
    op::MyOperator,
    x::AbstractVector{T}
) where T <: AbstractFloat
    y .= dot(x, x) .* x
    return y
end
function Base.:*(
    op::MyOperator,
    x::AbstractVector{T}
) where T <: AbstractFloat
    return dot(x, x) .* x
end
# needed for the in-place version
function ForwardDifferentiableExternalOperators.ApplyDerivative!(
    y::AbstractVector{T},
    op::MyOperator,
    x::AbstractVector{T},
    p::AbstractVector{T}
) where T <: AbstractFloat
    dxp = dot(x, p)
    dxx = dot(x, x)
    @. y = 2x * dxp + dxx * p
    return y
end
# needed for the out-of-place version
function ForwardDifferentiableExternalOperators.ApplyDerivative(
    op::MyOperator,
    x::AbstractVector{T},
    p::AbstractVector{T}
) where T <: AbstractFloat
    dxp = dot(x, p)
    dxx = dot(x, x)
    return 2x * dxp + dxx * p
end

Op = MyOperator(N)
FDOp = ForwardDifferentiableNonLinearOperator(Op)

f!(y, x, FDOp) = mul!(y, FDOp, x)
f(x, FDOp) = FDOp * x
direct_f(x) = dot(x, x) .* x

# generate x, v, and a work vector
x = rand(N);
v = rand(N);

# make a function that JacVec likes, and generate Jacobian-Vector product operator
g!(y, x) = f!(y, x, FDOp)
J! = JacVec(g!, x, tag=nothing);
y1 = similar(x);
mul!(y1,J!,v);
J = JacVec(x -> f(x, FDOp), x);
y2 = J*v;
# compare to taking the Jacobian via ForwardDiff of the straightforward implementation
y3 = ForwardDiff.jacobian(f, x) * v;
# are these the same?
display(y1 ≈ y3)
display(y2 ≈ y3)

# now get a timing test of this version
_y = similar(x);
@b mul!($_y,$J!,$v)

# we'll now test an auto-caching version
AFDOp = ForwardDifferentiableNonLinearOperator(Op, true)
g!(y, x) = f!(y, x, AFDOp)
J! = JacVec(g!, x, tag=nothing);
J = JacVec(x -> f(x, AFDOp), x);
y4 = similar(x);
# you MUST evaluate once at x
f!(_y, x, AFDOp);
mul!(y4,J!,v);
f(x, AFDOp);
y5 = J*v;
# same as before?
display(y4 ≈ y3)
display(y5 ≈ y3)

# now get a timing test, should be about twice as fast
@b mul!($y4,$J!,$v)



