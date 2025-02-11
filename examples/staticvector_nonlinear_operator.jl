using Revise
using ForwardDifferentiableExternalOperators
using ForwardDiff
using LinearOperators
using LinearAlgebra
using SparseDiffTools
using StaticArrays
using Chairmarks
using UnsafeArrays

# make this large so the linear operation dominates timings
N = 16

const Pt3 = SVector{3, Float64}

Base.iscontiguous(A::Array) = true
function unsafe_svectorize(A::AbstractVector{T}, ::Type{<:SVector{N, T}}) where {N, T}
    @assert Base.iscontiguous(A)
    Ap = Base.unsafe_convert(Ptr{SVector{N, T}}, A)
    return UnsafeArray(Ap, (length(A)÷N,)) 
end

################################################################################
# Linear Operator

struct MyOperator1
    pos::Vector{Pt3}
end
Base.size(op::MyOperator1) = (length(op.pos), length(op.pos))
Base.size(op::MyOperator1, i::Int) = length(op.pos)
Base.eltype(::MyOperator1) = SVector{3, Float64}
function LinearAlgebra.mul!(
    y::AbstractVector{SVector{3, T}},
    op::MyOperator1,
    x::AbstractVector{SVector{3, T}}
) where T <: AbstractFloat
    positions = op.pos
    reinterpret(Float64, y) .= 0.0
    for j in eachindex(positions, x)
        src = positions[j]
        pot = x[j]
        p1, p2, p3 = pot[1], pot[2], pot[3]
        pa = p1 + p2 + p3
        pb = p1 - p2 + p3
        pc = p1 + p2 - p3
        for i in eachindex(positions)
            if i != j
                d = norm(src - positions[i], 2)
                y[i] = y[i] + SVector{3, T}(pa/d, pb*d, pc*log(1 + d))
            end
        end
    end
    return y
end
pos = rand(SVector{3, Float64}, N);
MyOp1 = MyOperator1(pos);

################################################################################
# NonLinear Operator

struct MyOperator2
    N::Int
end
Base.size(op::MyOperator2) = (op.N, op.N)
Base.size(op::MyOperator2, i::Int) = op.N
Base.eltype(op::MyOperator2) = SVector{3, Float64}
function LinearAlgebra.mul!(
    y::AbstractVector{SVector{3, T}},
    op::MyOperator2,
    x::AbstractVector{SVector{3, T}}
) where T <: AbstractFloat
    @. y = dot(x, x) * x
    return y
end
function ForwardDifferentiableExternalOperators.ApplyDerivative!(
    y::AbstractVector{SVector{3, T}},
    op::MyOperator2,
    x::AbstractVector{SVector{3, T}},
    p::AbstractVector{SVector{3, T}}
) where T <: AbstractFloat
    @. y = 2x * dot(x, p) + dot(x, x) * p
    return y
end

MyOp2 = MyOperator2(N)

################################################################################
# implementation of test functions

# using this packages machinery
function f!(y::AbstractVector{T}, x::AbstractVector{T}, Op1, Op2, w) where T
    # reform as SVectors
    xr = unsafe_svectorize(x, SVector{3, T})
    yr = unsafe_svectorize(y, SVector{3, T})
    # get a cachevector for temporary
    wr = w(eltype(xr))
    # nonlinear operation done using this packages machinery
    mul!(wr, Op2, xr)
    # linear operation done using this packages machinery
    mul!(yr, Op1, wr)
    # nonlinear operation done using ForwardDiff directly
    @. yr = exp(dot(yr, xr)) * yr
    return y
end
# a straightforward implementation of the operator
# to pass to ForwardDiff.jacobian for ground truth
function f(x, pos)
    T = eltype(x)
    xr = reinterpret(SVector{3, T}, x)
    potential = @. dot(xr, xr) * xr
    y = zero(xr)
    for j in eachindex(pos, xr)
        src = pos[j]
        pot = potential[j]
        p1, p2, p3 = pot[1], pot[2], pot[3]
        pa = p1 + p2 + p3
        pb = p1 - p2 + p3
        pc = p1 + p2 - p3
        for i in eachindex(pos)
            if i != j
                d = norm(src - pos[i], 2)
                y[i] = y[i] + SVector{3, T}(pa/d, pb*d, pc*log(1 + d))
            end
        end
    end
    @. y = exp(dot(y, xr)) * y
    return reinterpret(T, y)
end

################################################################################
# generate data and cache vectors

# generate x, v, and a work vector
x = rand(3N);
v = rand(3N);
w = CacheVector(reinterpret(SVector{3, Float64}, x));

################################################################################
# test non-caching version version

FDOp1 = ForwardDifferentiableLinearOperator(MyOp1)
FDOp2 = ForwardDifferentiableNonLinearOperator(MyOp2)

# make a function that JacVec likes, and generate Jacobian-Vector product operator
g!(y, x) = f!(y, x, FDOp1, FDOp2, w)
J = JacVec(g!, x, tag=nothing);
# test our JacVec
y1 = similar(x);
mul!(y1,J,v);
# compare to taking the Jacobian via ForwardDiff of the straightforward implementation
g(x) = f(x, pos)
y2 = ForwardDiff.jacobian(g, x) * v;
# are these the same?
display(y1 ≈ y2)

# now get a timing test of this version
_y = similar(x);
@b mul!($_y,$J,$v)


################################################################################
# test auto-caching version version

# we'll now test an auto-caching version
AFDOp1 = ForwardDifferentiableLinearOperator(MyOp1, true)
AFDOp2 = ForwardDifferentiableNonLinearOperator(MyOp2, true)

g!(y, x) = f!(y, x, AFDOp1, AFDOp2, w)
J = JacVec(g!, x, tag=nothing);
y3 = similar(x);
# you MUST evaluate once at x
f!(_y, x, AFDOp1, AFDOp2, w);
mul!(y3,J,v);
# same as before?
display(y1 ≈ y3)

# now get a timing test, should be about twice as fast
@b mul!($y3,$J,$v)
