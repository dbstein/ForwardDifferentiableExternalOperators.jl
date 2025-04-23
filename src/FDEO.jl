
struct ForwardDifferentiableExternalOperator{
            T,
            DT,
            OP <: WrappedOperator{T}
    }
    operator::OP
    values_cache::Vector{T}
    partials_cache::Vector{T}
    output_cache1::Vector{T}
    output_cache2::Vector{T}
    auto_caching::Bool
end

function Base.show(
    F::ForwardDifferentiableExternalOperator{T, DT}
) where {T, DT}
    println("Forward Differentiable External Operator")
    println("  Is linear:             $(is_linear(F))")
    println("  Is auto_caching:       $(F.auto_caching)")
    println("  Operator type:         $(_typeof(F.operator))")
    println("  Size:                  $(size(F))")
    println("  Base Type:             $(T)")
    println("  Dual Type:             $(DT)")
end
Base.print(F::ForwardDifferentiableExternalOperator) = show(F)
Base.display(F::ForwardDifferentiableExternalOperator) = show(F)
is_linear(F::ForwardDifferentiableExternalOperator) = is_linear(F.operator)
Base.size(F::ForwardDifferentiableExternalOperator) = size(F.operator)
Base.size(F::ForwardDifferentiableExternalOperator, i::Int) = size(F.operator, i)
Base.eltype(F::ForwardDifferentiableExternalOperator) = eltype(F.operator)

function ForwardDifferentiableLinearOperator(operator, autocaching=false)
    return ForwardDifferentiableExternalOperator(operator, autocaching, LinearWrappedOperator)
end
function ForwardDifferentiableNonLinearOperator(operator, autocaching=false)
    return ForwardDifferentiableExternalOperator(operator, autocaching, NonLinearWrappedOperator)
end
function ForwardDifferentiableExternalOperator(operator, auto_caching, wrap_func)
    T = eltype(operator)
    display(T)
    DT = dual_backing(T)
    # DT = ForwardDiff.Dual{Nothing, T, 1}
    values_cache = Vector{T}(undef, size(operator, 2))
    partials_cache = Vector{T}(undef, size(operator, 2))
    output_cache1 = Vector{T}(undef, size(operator, 1))
    output_cache2 = Vector{T}(undef, size(operator, 1))
    wrapped_operator = wrap_func(operator)
    OP = typeof(wrapped_operator)
    return ForwardDifferentiableExternalOperator{T, DT, OP}(
                wrapped_operator,
                values_cache,
                partials_cache,
                output_cache1,
                output_cache2,
                auto_caching
            )
end

# these could possibly use the "underying type" rewrite
# to allow slightly more general operation (i.e from T -> S of different types)
# but that will wait for now...

function LinearAlgebra.mul!(
    y::AbstractVector{T},
    FDEO::ForwardDifferentiableExternalOperator{T, DT},
    x::AbstractVector{T}
) where {T, DT}
    mul!(y, FDEO.operator, x)
    if FDEO.auto_caching
        FDEO.output_cache1 .= y
    end
    return y
end
function LinearAlgebra.mul!(
    y::AbstractVector{DT},
    FDEO::ForwardDifferentiableExternalOperator{T, DT},
    x::AbstractVector{DT}
) where {T, DT}
    (UDT, _) = assess_underlying_type(DT)
    @. FDEO.values_cache = get_value(x)
    @. FDEO.partials_cache = get_partials(x)
    if !FDEO.auto_caching
        mul!(FDEO.output_cache1, FDEO.operator, FDEO.values_cache)
    end
    ApplyDerivative!(FDEO.output_cache2, FDEO.operator, FDEO.values_cache, FDEO.partials_cache)
    @. y = PackGrad(FDEO.output_cache1, FDEO.output_cache2, UDT)
    return y
end

################################################################################
# Out-of-place variants

function Base.:*(
    FDEO::ForwardDifferentiableExternalOperator,
    x::AbstractVector{T}
) where {T}
    (UT, is_dual_type) = assess_underlying_type(T)
    if is_dual_type
        # how we operator on duals
        yv = FDEO.auto_caching ? FDEO.output_cache1 : FDEO.operator * get_value.(x)
        yp = ApplyDerivative(FDEO.operator, get_value.(x), get_partials.(x))
        return PackGrad.(yv, yp, UT)    
    else
        # how we operate on non-duals
        y = FDEO.operator * x
        if FDEO.auto_caching
            FDEO.output_cache1 .= y
        end
        return y
    end
end
