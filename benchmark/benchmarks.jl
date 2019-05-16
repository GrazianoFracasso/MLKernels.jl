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
    "ExponentialKernel" => Dict("Cols"=>[ones(dim1)],"Rows"=>[ones(dim2)]),
    "SquaredExponentialKernel" => Dict("Cols"=>[ones(dim1)],"Rows"=>[ones(dim2)]),
    "GammaExponentialKernel" => Dict("Cols"=>[ones(dim1)],"Rows"=>[ones(dim2)]),
    "RationalQuadraticKernel" => Dict("Cols"=>[ones(dim1)],"Rows"=>[ones(dim2)]),
    "GammaRationalQuadraticKernel" => Dict("Cols"=>[ones(dim1)],"Rows"=>[ones(dim2)]),
    "ExponentiatedKernel" => Dict("Cols"=>[ones(dim1)],"Rows"=>[ones(dim2)]),
    "MaternKernel" => Dict("Cols"=>([1.0,ones(dim1)]),"Rows"=>([1.0,ones(dim2)])),
    "PolynomialKernel" => Dict("Cols"=>[ones(dim1)],"Rows"=>[ones(dim2)]),
    "PowerKernel" => Dict("Cols"=>[one(Float64)], "Rows"=>[one(Float64)]),
    "LogKernel" => Dict("Cols"=>[ones(dim1)],"Rows"=>[ones(dim2)]),
    "SigmoidKernel" => Dict("Cols"=>[ones(dim1)],"Rows"=>[ones(dim2)]))

for k in kernels
    global kernel = Dict("Rows"=>k(input_params[string(k)]["Rows"]...),"Cols"=>k(input_params[string(k)]["Cols"]...))
    SUITE["ard"][string(k)]["init"] = @benchmarkable $(k)($(input_params[string(k)]["Rows"])...)
    for (key,val) in obsdim
        SUITE["ard"][string(k)]["K(A)",key] = @benchmarkable  kernelmatrix($val,$(kernel[key]),$A)
        SUITE["ard"][string(k)]["K(A,B)",key] = @benchmarkable  kernelmatrix($val,$(kernel[key]),$A,$B)
    end
end

# run(SUITE,verbose=true)
