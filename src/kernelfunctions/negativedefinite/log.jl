@doc raw"""
    LogKernel([α [,γ]])

The Log Kernel is a negative definite kernel given by the formula:

```
    κ(x,y) = log(1 + α‖x-y‖²ᵞ)   α > 0, γ ∈ (0,1]
```
where `α` is a scaling parameter and `γ` is a shape parameter of the Euclidean distance.

# Examples

```jldoctest; setup = :(using MLKernels)
julia> LogKernel()
LogKernel{Float64}(1.0,1.0)

julia> LogKernel(0.5f0)
LogKernel{Float32}(0.5,1.0)

julia> LogKernel(0.5, 0.5)
LogKernel{Float64}(0.5,0.5)
```
"""
struct LogKernel{T<:Real,A} <: NegativeDefiniteKernel{T}
    α::A
    γ::T
    metric::Metric
    function LogKernel{T}(α::A=T(1), γ::Real=T(1)) where {A<:Union{Real,AbstractVector{<:Real}},T<:Real}
        @check_args(LogKernel, α, all(α .> zero(T)), "α > 0")
        @check_args(LogKernel, γ, one(T) >= γ > zero(T), "γ ∈ (0,1]")
        o
        if A <: Real
            new{T,A}(α,γ,SquaredEuclidean())
        else
            new{T,A}(α,γ,WeightedSqEuclidean(α.^-γ))
        end
    end
end
function LogKernel(α::Union{T₁,AbstractVector{T₁}}=1.0, γ::T₂=T₁(1)) where {T₁<:Real,T₂<:Real}
    LogKernel{promote_float(T₁,T₂)}(α, γ)
end

# @inline basefunction(::LogKernel) = SquaredEuclidean()

@inline kappa(κ::LogKernel{T,<:Real}, d²::T) where {T} = log(one(T) + (κ.α*d²)^(κ.γ))
@inline kappa(κ::LogKernel{T}, d²::T) where {T} = log(one(T) + d²^(κ.γ))

function convert(::Type{K}, κ::LogKernel) where {K>:LogKernel{T,A} where A} where T
    return LogKernel{T}(T.(κ.α), T.(κ.γ))
end
