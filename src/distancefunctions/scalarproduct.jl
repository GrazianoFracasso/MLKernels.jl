struct ScalarProduct <: Distances.PreMetric end

struct WeightedScalarProduct{W <: AbstractArray{<:Real}} <: Distances.PreMetric
    weights::W
end

function Distances.evaluate(dist::ScalarProduct,a::AbstractArray,b::AbstractArray)
    LinearAlgebra.dot(a,b)
end

function Distances.evaluate(dist::WeightedScalarProduct,a::AbstractArray,b::AbstractArray)
    LinearAlgebra.dot(dist.weights.*a,b)
end
