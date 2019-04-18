using Zygote
using ForwardDiff
using Flux.Tracker
using BenchmarkTools
using LinearAlgebra, Distances
using MLKernels

l = 2.0
X = rand(10,2)
Y = rand(100,2)
function approxK(l)
    global k = SquaredExponentialKernel(l)
    Kx = kernelmatrix(k,X)
    Ky = kernelmatrix(k,Y)
    Kxy = kernelmatrix(k,X,Y)
    tr(Ky-Kxy'*inv(Kx)*Kxy)
end
kappa1(x,x2) = exp.(-0.5*evaluate(SqEuclidean(),x,x2))
kappa2(d) = exp.(-0.5*d)

A = rand(1000,100)
function create_matrix1(A)
    P = zeros(size(A,1),size(A,1))
    for i in 1:size(A,1), j in 1:size(A,1)
        @inbounds P[i,j] = kappa1(A[i,:],A[j,:])
    end
    return P
end
function create_matrix2(A)
    P = pairwise(SqEuclidean(),A,dims=1)
    for i in 1:size(A,1), j in 1:size(A,1)
        @inbounds P[i,j] = kappa2(P[i,j])
    end
    return P
end
Aref = kernelmatrix(SquaredExponentialKernel(1.0),A)
A1 = create_matrix1(A)
@show A1 == Aref
A2 = create_matrix2(A)
@show A2 == Aref
# @btime create_matrix1($A);
# @btime create_matrix2($A);
approxK([l,l+0.5])
## Testing
# Zygote.gradient(approxK,l)
# Zygote not working because of mutating arrays

@show ForwardDiff.gradient(approxK,[l,l+0.5])
## Works nicely with everything
@show Tracker.data(Tracker.gradient(approxK,[l,l+0.5]))
# Only works for iso kernels,

##Performance
# @btime Zygote.gradient(trK,l)
@btime ForwardDiff.gradient(approxK,[$l,$l+0.5]);
@btime Tracker.gradient(approxK,[$l,$l+0.5]);
