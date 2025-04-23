
# function to extract base type from ForwardDiff.Dual type
@inline base_type(::Type{ForwardDiff.Dual{a, b, c}}) where {a, b, c} = b

# extraction functions on Scalars
@inline get_value(x::ForwardDiff.Dual) = x.value
@inline get_value(x::Real) = x
@inline get_partials(x::ForwardDiff.Dual) = x.partials[1]
@inline get_partials(x::Real) = x

# extraction functions on SVectors
@inline function get_value(x::SVector{N, DT}) where {N, DT <: ForwardDiff.Dual}
    T = base_type(DT)
    return SVector{N, T}(get_value.(x))
end
@inline get_value(x::SVector{N, T}) where {N, T <: Real} = x
@inline function get_partials(x::SVector{N, DT}) where {N, DT <: ForwardDiff.Dual}
    T = base_type(DT)
    return SVector{N, T}(get_partials.(x))
end
@inline get_partials(x::SVector{N, T}) where {N, T <: Real} = x

# packing functions on scalars
@inline PackGrad(v::TT, p::TT, UDT::Type{DT}) where {TT, DT <: ForwardDiff.Dual} = UDT(v, ForwardDiff.Partials{1, TT}((p,)))
# packing functions on StaticVectors
@inline function PackGrad(
    v::SVector{N, TT},
    p::SVector{N, TT},
    UDT::Type{DT}
) where {N, TT, DT <: ForwardDiff.Dual}
    return SVector{N, UDT}(PackGrad.(v, p, UDT))
end

# function to get the dual type
@inline dual_backing(::Type{T}) where T <: AbstractFloat = ForwardDiff.Dual{Nothing, T, 1}
@inline dual_backing(::Type{T}) where T <: SVector = SVector{length(T), dual_backing(eltype(T))}

# checker functions on scalars
@inline is_dual_type(::Type{<:ForwardDiff.Dual}) = true
@inline is_dual_type(::Type{<:Real}) = false
# checker functions on StaticVectors
@inline is_dual_type(::Type{<:SVector{N, TT}}) where {N, TT} = is_dual_type(TT)

# simple implementation of dual caches (with duplicative memory)
struct CacheVector{T, DT}
    FloatCache::Vector{T}
    DualCache::Vector{DT}
end
function CacheVector(T, N)
    DT = dual_backing(T)
    # DT = ForwardDiff.Dual{Nothing, T, 1}
    return CacheVector{T, DT}(
                Vector{T}(undef, N),
                Vector{DT}(undef, N)
            )
end
function CacheVector(V::AbstractVector)
    return CacheVector(eltype(V), length(V))
end
(CV::CacheVector{T, DT})(::Type{T})  where {T, DT} = CV.FloatCache
(CV::CacheVector{T, DT})(::Type{DT}) where {T, DT} = CV.DualCache
Base.length(CV::CacheVector) = length(CV.FloatCache)
function Base.show(
    CV::CacheVector{T, DT}
) where {T, DT}
    println("CacheVector")
    println("  Length is:             $(length(CV))")
    println("  Base Type:             $(T)")
    println("  Dual Type:             $(DT)")
end
Base.print(CV::CacheVector) = show(CV)
Base.display(CV::CacheVector) = show(CV)

# function to recursively get the element type
function get_number_type(T)
    if T <: Number
        return T
    end
    get_number_type(eltype(T))
end
get_number_type(::Type{T}) where T <: Number = T
get_number_type(::Type{T}) where {R <: Number, T <: AbstractVector{R}} = R
get_number_type(::Type{T}) where {R <: Number, S <: AbstractVector{R}, T <: AbstractVector{S}} = R

function assess_underlying_type(T)
    UT = get_number_type(T)
    return UT, UT <: ForwardDiff.Dual
end
