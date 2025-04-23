
abstract type WrappedOperator{T, OP} end
_typeof(::WrappedOperator{T, OP}) where {T, OP} = OP
Base.size(WO::WrappedOperator) = size(WO.operator)
Base.size(WO::WrappedOperator, i::Int) = size(WO.operator, i)

function LinearAlgebra.mul!(
    y::AbstractVector{T}, 
    WO::WrappedOperator{T}, 
    x::AbstractVector{T}
) where T
    mul!(y, WO.operator, x)
    return y
end
function Base.:*(
    WO::WrappedOperator{T}, 
    x::AbstractVector{T}
) where T
    return WO.operator * x
end

struct LinearWrappedOperator{T, OP} <: WrappedOperator{T, OP}
    operator::OP
end
function LinearWrappedOperator(operator::OP) where OP
    return LinearWrappedOperator{eltype(operator), OP}(operator)
end
is_linear(::LinearWrappedOperator) = true
function ApplyDerivative!(
    y::AbstractVector{T}, 
    LWO::LinearWrappedOperator{T}, 
    x::AbstractVector{T},
    p::AbstractVector{T}
) where T
    mul!(y, LWO.operator, p)
    return y
end
function ApplyDerivative(
    LWO::LinearWrappedOperator{T}, 
    x::AbstractVector{T},
    p::AbstractVector{T}
) where T
    return LWO.operator * p
end

struct NonLinearWrappedOperator{T, OP} <: WrappedOperator{T, OP}
    operator::OP
end
function NonLinearWrappedOperator(operator::OP) where OP
    return NonLinearWrappedOperator{eltype(operator), OP}(operator)
end
is_linear(::NonLinearWrappedOperator) = false
function ApplyDerivative!(
    y::AbstractVector{T}, 
    NLWO::NonLinearWrappedOperator{T}, 
    x::AbstractVector{T},
    p::AbstractVector{T}
) where T
    ApplyDerivative!(y, NLWO.operator, x, p)
    return y
end
function ApplyDerivative(
    NLWO::NonLinearWrappedOperator{T}, 
    x::AbstractVector{T},
    p::AbstractVector{T}
) where T
    return ApplyDerivative(NLWO.operator, x, p)
end
