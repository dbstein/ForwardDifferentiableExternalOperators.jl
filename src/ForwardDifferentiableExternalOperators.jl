module ForwardDifferentiableExternalOperators

using ForwardDiff
using StaticArrays
using LinearAlgebra

export ForwardDifferentiableLinearOperator
export ForwardDifferentiableNonLinearOperator
export CacheVector

include("Utilities.jl")
include("WrappedOperators.jl")
include("FDEO.jl")

end
