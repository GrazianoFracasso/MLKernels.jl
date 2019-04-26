struct ScalarProduct <: Distances.Metric end

struct WeightedScalarProduct{W <: AbtractArray{<:Real}} <: Distances.Metric
    weights::W
end

function evaluate(dist::ScalarProduct,a::AbstractArray,b::AbstractArray)
    dot(a,b)
end

function evaluate(dist::WeightedScalarProduct,a::AbstractArray,b::AbstractArray)
    dot(dist.w.*a,b)
end
