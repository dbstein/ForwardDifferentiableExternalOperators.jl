# ForwardDifferentiableExternalOperators

Package to provide easy Forward-Differentiable compatible Operators in cases where the operators pass through foreign function barriers or when those operators are expensive and caching is useful for speedup. Probably cleanest and most useful in the case where the External operator is linear; then the Jacobian of A is A itself, and so A can be applied to a Vector{Dual} by simply applying A first to the values, then to the partials.

# What is supported

1. Support for Jacobian-vector products. No other differentiation operations have been tested, and are unlikely to work properly.
2. Linear operators, defined as a struct OP that has:
    a. size(OP) and size(OP, i::Int) defined on it
    b. eltype(OP)
    c. a mul!(y, OP, x) operation defined on it.
Further, it is assumed that x/y are AbstractVector{T} for some type T. If T <: Real or T <: SVector{N, <:Real}, then this should just work. If not, you'll need to define the functions:
    a. get_value
    b. get_partials
    c. PackGrad
    d. is_dual_type
See the definitions for SVector{N, <:Real} in Utilities.jl as a guide.
3. Nonlinear operators, defined as a struct OP that has:
    a. all the same things required of the linear operator, and,
    b. a ApplyDerivative!(y, OP, x, p) function, where x is the value and p is the partial.
3. Automatic-caching. Suppose you have a function f(x) = g(L(h(x)), x), where L is your Linear Operator and g & h may be arbitrary nonlinear functions. You now wish to evaluate Jf(x)\*v for many different values of v, but at the same x. In this case, naive application of ForwardDiff using this package results in execution of L twice per Jacobian vector product, when it should only need to be done once per Jacobian vector product. Instead, given a value x, simply construct an ForwardDifferentiableLinearOperator(L, true), call the funciton f once, and each subsequent call of Jf(x) will now execute L only once. It is imperative that L is only called *once* inside f. Multiple calls to L are not supported. You can always construct multiple ForwardDifferentiableLinearOperator wrappers to L if it is called more than once inside f; just be sure each wrapper is only called once. Automatic caching should also work for nonlinear functions, but has not been tested yet.

## Caveats

1. It is incumbent upon the user to ensure that what they tell this package is actually correct. In particular, they will get incorrect results if:
    a. operators which are formed using ForwardDifferentiableLinearOperator are not actually linear,
    b. the ApplyDerivative! function for a nonlinear operator doesn't actually evaluate the correct partial of the corresponding mul! function.
2. For the Auto-caching versions, each instance of ForwardDifferentiableExternalOperator(OP, true) must be called *only* once in the the function f that is being differentiated.

## Example

[![Build Status](https://github.com/dbstein/ForwardDifferentiableExternalOperators.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/dbstein/ForwardDifferentiableExternalOperators.jl/actions/workflows/CI.yml?query=branch%3Amain)
