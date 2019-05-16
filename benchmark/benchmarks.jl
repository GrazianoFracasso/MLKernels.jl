using BenchmarkTools
using MLKernels
using Random: seed!

seed!(42)

dim1 = 100; dim2 = 100
A = rand(dim1,dim2)
B = rand(dim1,dim2)

const SUITE = BenchmarkGroup()

SUITE["iso"] = BenchmarkGroup()
SUITE["ard"] = BenchmarkGroup()
kernels = (ExponentialKernel,SquaredExponentialKernel,GammaExponentialKernel,RationalQuadraticKernel,GammaRationalQuadraticKernel,ExponentiatedKernel,MaternKernel,PolynomialKernel,PowerKernel,LogKernel,SigmoidKernel)
for g in ("iso","ard")
    for k in kernels
        SUITE[g][string(k)] = BenchmarkGroup()
    end
end
obsdim = ["Cols"=>Val(:col),"Rows"=>Val(:row)]
for k in kernels
    kernel = k()
    SUITE["iso"][string(k)]["init"] = @benchmarkable $(k)()
    for (key,val) in obsdim
        SUITE["iso"][string(k)]["K(A)",key] = @benchmarkable  kernelmatrix($val,$kernel,$A)
        SUITE["iso"][string(k)]["K(A,B)",key] = @benchmarkable  kernelmatrix($val,$kernel,$A,$B)
    end
end

input_params = Dict(
    "ExponentialKernel" => Dict("Cols"=>ones(Dim1),"Rows"=>ones(Dim2)),
)
