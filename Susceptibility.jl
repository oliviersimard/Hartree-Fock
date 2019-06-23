module Sus

include("HF_Hubbard.jl")
using ..Hubbard: integrateComplex, initGk, Gk, epsilonk, II, Integral1D, Integral2D
import ..Hubbard: HubbardStruct, MyType


function integrateComplex1DWrapper(SE_funct::Function, ii::Int64, structModel::HubbardStruct, BoundArr::Array{Float64,1}, Gridk::Int64, opt::String)
    function tmp_funct(funct::Function)
        return integrateComplex(funct::Function, SE_funct, ii, structModel, BoundArr, Gridk, opt)
    end
    return tmp_funct
end


function iterationProcess(structModel::Sus.HubbardStruct, BoundArr::Union{Array{Float64,1}, Array{Array{Float64,1},1}}, filename::String; 
    SubLast::Int64=2, Gridk::Int64=100, opt::String="sum")
    @assert (isa(BoundArr,Array{Array{Float64,1},1}) || isa(BoundArr,Array{Float64,1})) "Only treats the 1D and 2D cases for now! Check iterationProcess(...)."
    N_it = structModel.N_it_
    SE_funct = missing
    dictArray = Dict{Int64,Array{Array{Complex{Float64},2},1}}()
    f = open(filename, "a")
    if isa(BoundArr,Array{Float64,1})
        for it in 1:N_it    
            Funct_array = Array{Array{Complex{Float64},2},1}([]) ## SubLast has got to be devided in two, because there are two boundaries.
            arrBoundaries = range(BoundArr[1],stop=BoundArr[2],length=SubLast) |> collect
            @assert (mod(length(arrBoundaries),2) == 0) "The value of SubLast must be an even number!"
            arrBoundaries = reshape( arrBoundaries, ( div( length(arrBoundaries), 2 ), 2 ) )[1,:]
            if it <= 1
                dummyMatrix = Matrix{Complex{Float64}}(undef,(2,2))
                println(it, " iteration_process 1D ", "Boundaries1D: ", arrBoundaries)
                push!(Funct_array, integrateComplex(initGk, dummyMatrix, it, structModel, arrBoundaries, Gridk=Gridk, opt=opt))
            elseif it > 1
                println(it, " iteration_process 1D ", "Boundaries1D: ", arrBoundaries)
                push!(Funct_array, integrateComplex(Gk, dictArray[it-1][1], it, structModel, arrBoundaries, Gridk=Gridk, opt=opt))
            end
            println("Funct_arr: ", Funct_array)
            write(f, "$(it): "*"$(Funct_array)"*"\n")
            dictArray[it] = Funct_array
        end
        close(f)
    elseif isa(BoundArr,Array{Array{Float64,1},1})
        for it in 1:N_it    
            Funct_array = Array{Array{Complex{Float64},2},1}([]) ## SubLast has got to be devided in two, because there are two boundaries.
            if it <= 1
                dummyMatrix = Matrix{Complex{Float64}}(undef,(2,2))
                println(it, " iteration_process 2D ", "Boundaries2D: ", BoundArr[1][1], " ", Gridk, " ", BoundArr[2][1])
                push!(Funct_array, integrateComplex(initGk, dummyMatrix, it, structModel, BoundArr, Gridk=Gridk, opt=opt))
            elseif it > 1
                println(it, " iteration_process 2D ", "Boundaries2D: ", BoundArr)
                push!(Funct_array, integrateComplex(Gk, dictArray[it-1][1], it, structModel, BoundArr, Gridk=Gridk, opt=opt))
            end
            println("Funct_arr: ", Funct_array)
            write(f, "$(it): "*"$(Funct_array)"*"\n")
            dictArray[it] = Funct_array
        end
        close(f)
    end
    return dictArray
end

function Gk_conv(vals::Union{Integral2D,Integral1D}, c_container::Vector{Array{Complex{Float64},2}}, dictFunct::Dict{Int64,Array{Array{Complex{Float64},2},1}}, N_it::Int64; 
    SubLast::Int64=2)
    if isa(vals, Integral1D)
        #println("In gk_conv 1D")
        for idx in 1:div(SubLast,2)
            c_container[idx] = dictFunct[N_it][idx]
        end
        summation_subs = sum(c_container)
        
        Gk = inv(vals.iωn*II - epsilonk(vals.qx)*II - summation_subs)
        
    elseif isa(vals, Integral2D)
        #println("In gk_conv 2D")
        for idx in 1:div(SubLast,2)
            c_container[idx] = dictFunct[N_it][idx]
        end
        summation_subs = sum(c_container)
        
        Gk = inv(vals.iωn*II - epsilonk([vals.qx, vals.qy])*II - summation_subs)
    end

    return Gk
end

function Lambda(HF::HubbardStruct, Gk1::Union{Integral1D,Integral2D}, Gk2::Union{Integral1D,Integral2D}, c_container::Vector{Array{Complex{Float64},2}}, 
    dictFunct::Dict{Int64,Array{Array{Complex{Float64},2},1}}; SubLast::Int64=2)
    #println("IN LAMBDA FUNCTION", "\n")

    kernel = inv( 1 + HF.dict_["U"]*Gk_conv(Gk1,c_container,dictFunct,HF.N_it_)[1,1]*Gk_conv(Gk2,c_container,dictFunct,HF.N_it_)[2,2] )

    #println("kernel value: ", kernel, "\n")
    return kernel
end

function Susceptibility(HF::HubbardStruct, Gk1::Union{Integral1D,Integral2D}, Gk2::Union{Integral1D,Integral2D}, 
    Gks::Union{Array{Integral1D,1},Array{Integral2D,1}}, c_container::Vector{Array{Complex{Float64},2}}, dictFunct::Dict{Int64,Array{Array{Complex{Float64},2},1}}; SubLast::Int64=2)
    #println("IN SUSCEPTIBILITY FUNCTION", "\n")
    sus = Gk_conv(Gks[1],c_container,dictFunct,HF.N_it_)[1,1]*HF.dict_["U"]*Lambda(HF,Gk1,Gk2,c_container,dictFunct)*Gk_conv(Gks[2],c_container,dictFunct,HF.N_it_)[2,2]*Gk_conv(Gks[3],c_container,dictFunct,HF.N_it_)[2,2]*Gk_conv(Gks[4],c_container,dictFunct,HF.N_it_)[1,1]

    #println("Susceptibility: ", sus, "\n\n")
    return sus
end


end ## end of module Sus