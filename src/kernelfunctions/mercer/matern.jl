@doc raw"""
    MaternKernel([ν=1 [, θ=1]])

The Matern kernel is a Mercer kernel with parameters `ν > 0` and `ρ > 0`. See the published
documentation for the full definition of the function.

# Examples

```jldoctest; setup = :(using MLKernels)
julia> MaternKernel()
MaternKernel{Float64}(1.0,1.0)

julia> MaternKernel(2.0f0)
MaternKernel{Float32}(2.0,1.0)

julia> MaternKernel(2.0f0, 2.0)
MaternKernel{Float64}(2.0,2.0)
```
"""
struct MaternKernel{T<:Real,A} <: MercerKernel{T}
    ν::T
    ρ::A
    metric::Metric
    function MaternKernel{T}(ν::Real=T(1), ρ::A=T(1)) where {A<:Union{Real,AbstractVector{<:Real}},T<:Real}
        @check_args(MaternKernel, ν, ν > zero(T), "ν > 0")
        @check_args(MaternKernel, ρ, count(ρ .<= zero(T)) == 0, "ρ > 0")
        if A <: Real
            return new{T,A}(ν, ρ, Euclidean())
        else
            return new{T,A}(ν, ρ, WeightedEuclidean(1.0./ρ.^2))
        end
    end
end

MaternKernel(ν::T₁=1.0, ρ::Union{T₂,AbstractVector{T₂}}=one(T₁)) where {T₁<:Real,T₂<:Real} = MaternKernel{promote_float(T₁,T₂)}(ν,ρ)

@inline function kappa(κ::MaternKernel{T,<:Real}, d::T) where {T}
    d = d < eps(T) ? eps(T) : d  # If d is zero, besselk will return NaN
    tmp = √(2κ.ν)*d/κ.ρ
    return (convert(T, 2)^(one(T) - κ.ν))*(tmp^κ.ν)*besselk(κ.ν, tmp)/gamma(κ.ν)
end

@inline function kappa(κ::MaternKernel{T}, d::T) where {T}
    d = d < eps(T) ? eps(T) : d  # If d is zero, besselk will return NaN
    tmp = √(2κ.ν)*d
    return (convert(T, 2)^(one(T) - κ.ν))*(tmp^κ.ν)*besselk(κ.ν, tmp)/gamma(κ.ν)
end

function convert(::Type{K}, κ::MaternKernel) where {K>:MaternKernel{T,A} where A} where T
    return MaternKernel{T}(T(κ.ν), T.(1.0./sqrt.(κ.α)))
end


##TODO Write general form for 1/2, 3/2 and 5/2
