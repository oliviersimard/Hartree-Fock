## Main translation unit
# Need to include Precompile.jl before running this unit
#using Distributed

## Some important parameters
dict = Dict{String,Float64}("U" => 4.0, "V" => 1.0)
beta = 200 # For 1D, beta = 100 and Niωn = 50 seems to converge well. For 1D, opt = "integral" gives results fast enough.
           # For 2D, beta = 200 and Niωn = 50 seems to stabilize efficiently the convergence loop. For 2D, opt = "sum" should be specified. (Incresing gap between beta > Niωn)
Niωn = 50 ## Niωn should absolutely be lower than beta value.
Dims = 2
Grid_K = 400
##
SubLast = 2 ## Subdivision of last integral (N_it) to be split in #Sublast to be fed to different cores
N_it = 5 ## Lowest number is 1: one loop in the process. Converges faster for 2D (~15 iterations) while for 1D slower (~30 iterations).
Full = false ## If you want to compute the results of all the iterations, set to true. Set to false otherwise!

filename = "$(Dims)D_HF_Susceptibility_calc_minus_sign_kGrid_$(Grid_K)_N_it_$(N_it)_beta_$(beta)_Niwn_$(Niωn).dat"

if isfile(filename)
    rm(filename)
end


@assert (mod(SubLast,2) == 0) "Variable SubLast must be even for it to be splitable for the dispatch of the jobs to the processors."

include("Precompile.jl")
using .SuperHF

function integrateComplex1DWrapper(SE_funct::Function, ii::Int64, structModel::SuperHF.Hubbard.HubbardStruct, BoundArr::Array{Float64,1}, Gridk::Int64, opt::String)
    function tmp_funct(funct::Function)
        return SuperHF.Hubbard.integrateComplex(funct::Function, SE_funct, ii, structModel, BoundArr, Gridk, opt)
    end
    return tmp_funct
end


function iterationProcess(structModel::SuperHF.Hubbard.HubbardStruct, BoundArr::Union{Array{Float64,1}, Array{Array{Float64,1},1}}; 
    SubLast::Int64=SubLast, Gridk::Int64=100, opt::String="sum")
    @assert (isa(BoundArr,Array{Array{Float64,1},1}) || isa(BoundArr,Array{Float64,1})) "Only treats the 1D and 2D cases for now! Check iterationProcess(...)."
    N_it = structModel.N_it_
    SE_funct = missing
    dictArray = Dict{Int64,Array{Array{Complex{Float64},2},1}}()
    if isa(BoundArr,Array{Float64,1})
        for it in 1:N_it    
            Funct_array = Array{Array{Complex{Float64},2},1}([]) ## SubLast has got to be devided in two, because there are two boundaries.
            arrBoundaries = range(BoundArr[1],stop=BoundArr[2],length=SubLast) |> collect
            @assert (mod(length(arrBoundaries),2) == 0) "The value of SubLast must be an even number!"
            arrBoundaries = reshape( arrBoundaries, ( div( length(arrBoundaries), 2 ), 2 ) )[1,:]
            if it <= 1
                dummyMatrix = Matrix{Complex{Float64}}(undef,(2,2))
                println(it, " iteration_process 1D ", "Boundaries1D: ", arrBoundaries)
                push!(Funct_array, SuperHF.Hubbard.integrateComplex(SuperHF.Hubbard.initGk, dummyMatrix, it, structModel, arrBoundaries, Gridk=Gridk, opt=opt))
            elseif it > 1
                println(it, " iteration_process 1D ", "Boundaries1D: ", arrBoundaries)
                push!(Funct_array, SuperHF.Hubbard.integrateComplex(SuperHF.Hubbard.Gk, dictArray[it-1][1], it, structModel, arrBoundaries, Gridk=Gridk, opt=opt))
            end
            println("Funct_arr: ", Funct_array)
            dictArray[it] = Funct_array
        end
    elseif isa(BoundArr,Array{Array{Float64,1},1})
        for it in 1:N_it    
            Funct_array = Array{Array{Complex{Float64},2},1}([]) ## SubLast has got to be devided in two, because there are two boundaries.
            if it <= 1
                dummyMatrix = Matrix{Complex{Float64}}(undef,(2,2))
                println(it, " iteration_process 2D ", "Boundaries2D: ", BoundArr[1][1], " ", Gridk, " ", BoundArr[2][1])
                push!(Funct_array, SuperHF.Hubbard.integrateComplex(SuperHF.Hubbard.initGk, dummyMatrix, it, structModel, BoundArr, Gridk=Gridk, opt=opt))
            elseif it > 1
                println(it, " iteration_process 2D ", "Boundaries2D: ", BoundArr)
                push!(Funct_array, SuperHF.Hubbard.integrateComplex(SuperHF.Hubbard.Gk, dictArray[it-1][1], it, structModel, BoundArr, Gridk=Gridk, opt=opt))
            end
            println("Funct_arr: ", Funct_array)
            dictArray[it] = Funct_array
        end
    end
    return dictArray
end

## Instantiating HubbardStruct for forthcoming calculations
model = SuperHF.Hubbard.HubbardStruct(Niωn, dict, N_it, beta)

## Boundaries in k-space for the 1D system
Boundaries1D = Array{Float64,1}([-pi,pi])
qp_array1D = Array{Float64,1}([a for a in range(-pi,stop=pi,length=Grid_K)])
qpp_array1D = qp_array1D; k_array1D = qp_array1D; kp_array1D = qp_array1D

## Boundaries in k-space for the 2D system
Boundaries2D = Array{Array{Float64,1},1}([[-pi,-pi],[pi,pi]])
qp_array2D = Array{Array{Float64,1},1}([[a,b] for a in range(-pi,stop=pi,length=Grid_K) for b in range(-pi,stop=pi,length=Grid_K)])
qpp_array2D = qp_array2D; k_array2D = qp_array2D; kp_array2D = qp_array2D

function Gk_conv(vals::Union{SuperHF.Hubbard.Integral2D,SuperHF.Hubbard.Integral1D})
    if isa(vals, SuperHF.Hubbard.Integral1D)
        #println("In gk_conv 1D")
        for idx in 1:div(SubLast,2)
            c_container[idx] = dictFunct[N_it][idx]
        end
        summation_subs = sum(c_container)
        
        Gk = inv(vals.iωn*SuperHF.Hubbard.II - SuperHF.Hubbard.epsilonk(vals.qx)*SuperHF.Hubbard.II - summation_subs)
        
        return Gk
    elseif isa(vals, SuperHF.Hubbard.Integral2D)
        #println("In gk_conv 2D")
        for idx in 1:div(SubLast,2)
            c_container[idx] = dictFunct[N_it][idx]
        end
        summation_subs = sum(c_container)
        
        Gk = inv(vals.iωn*SuperHF.Hubbard.II - SuperHF.Hubbard.epsilonk([vals.qx, vals.qy])*SuperHF.Hubbard.II - summation_subs)
    end
end

function Lambda(HF::SuperHF.Hubbard.HubbardStruct, Gk1::Union{SuperHF.Hubbard.Integral1D,SuperHF.Hubbard.Integral2D}, 
    Gk2::Union{SuperHF.Hubbard.Integral1D,SuperHF.Hubbard.Integral2D})
    #println("IN LAMBDA FUNCTION", "\n")

    kernel = inv( 1 - dict["U"]*Gk_conv(Gk1)[1,1]*Gk_conv(Gk2)[2,2] )

    #println("kernel value: ", kernel, "\n")
    return kernel
end

function Susceptibility(HF::SuperHF.Hubbard.HubbardStruct, Gk1::Union{SuperHF.Hubbard.Integral1D,SuperHF.Hubbard.Integral2D}, 
    Gk2::Union{SuperHF.Hubbard.Integral1D,SuperHF.Hubbard.Integral2D}, Gks::Union{Array{SuperHF.Hubbard.Integral1D,1},Array{SuperHF.Hubbard.Integral2D,1}})
    #println("IN SUSCEPTIBILITY FUNCTION", "\n")
    sus = Gk_conv(Gks[1])[1,1]*dict["U"]*Lambda(HF,Gk1,Gk2)*Gk_conv(Gks[2])[2,2]*Gk_conv(Gks[3])[2,2]*Gk_conv(Gks[4])[1,1]

    #println("Susceptibility: ", sus, "\n\n")
    return sus
end

### Main 
c_container = Vector{Array{Complex{Float64},2}}(undef,div(SubLast,2))
if Dims == 1
    dictFunct = iterationProcess(model, Boundaries1D, Gridk=Grid_K, opt="integral")
    println("Length of function array: ", length(dictFunct[N_it]))
    function main()
        try
            @assert isa(dictFunct,Dict{Int64,Array{Array{Complex{Float64},2},1}}) "Dictionnary holding self-energies must have a given form. Look inside main function."
            Matsubara_array_susceptibility = Array{Complex{Float64},1}()
            q = 0.
            @time for (ii,iωn) in enumerate(model.matsubara_grid_)
                f = open(filename, "a")
                if ii == 1
	                write(f, "#N_it "*"$(N_it)"*" q="*"$(q)"*" Gridk "*"$(Grid_K)"*"\n")
                end
                k_sum = 0.0+0.0im
                println("iwn: ", iωn)
                for qp in qp_array1D
                    #println("In qp: ", qp)
                    for k in k_array1D
                        #println("In k: ", k)
                        for kp in kp_array1D
                            #println("In kp: ", kp)
                            Gk1 = SuperHF.Hubbard.Integral1D(k, iωn); Gk2 = SuperHF.Hubbard.Integral1D(kp+qp, iωn)
                            Gks1 = SuperHF.Hubbard.Integral1D(k, iωn); Gks2 = SuperHF.Hubbard.Integral1D(kp+q, iωn)
                            Gks3 = SuperHF.Hubbard.Integral1D(kp, iωn); Gks4 = SuperHF.Hubbard.Integral1D(k-q, iωn)
                            Matsubara_sus = Susceptibility(model, Gk1, Gk2, [Gks1,Gks2,Gks3,Gks4])
                            k_sum += Matsubara_sus
                        end
                    end
                end
                k_sum = 2.0*(1.0/(Grid_K))^3*k_sum ## 2.0 is for the spin
                push!(Matsubara_array_susceptibility,k_sum)
                write(f, "$(iωn)"*"\t\t"*"$(k_sum)"*"\n")
                close(f)
            end
            tot_susceptibility = (2.0/model.beta_)^3*sum(Matsubara_array_susceptibility)
            println("total Susceptibility for q = $(q): ", tot_susceptibility)
            return tot_susceptibility
        catch err
            if typeof(err) == InterruptException
                println("ALL THE TASKS HAVE BEEN INTERRUPTED","\n")
            else
                println("$(err)")
            end
        end
        println("Program terminated. Have a nice day!")
        return nothing
    end
elseif Dims == 2
    dictFunct = iterationProcess(model, Boundaries2D, Gridk=Grid_K, opt="sum")
    println("Length of function array: ", length(dictFunct[N_it]))
    function main()
        try
            @assert isa(dictFunct,Dict{Int64,Array{Array{Complex{Float64},2},1}}) "Dictionnary holding self-energies must have a given form. Look inside main function."
            Matsubara_array_susceptibility = Array{Complex{Float64},1}()
            q = [0.,0.]
            @time for (ii,iωn) in enumerate(model.matsubara_grid_)
                f = open(filename, "a")
                if ii == 1
	                write(f, "#N_it "*"$(N_it)"*" q="*"$(q)"*" Gridk "*"$(Grid_K)"*"\n")
                end
                k_sum = 0.0+0.0im
                println("iwn: ", iωn)
                for qp in qp_array2D
                    #println("In qp: ", qp)
                    for k in k_array2D
                        #println("In k: ", k)
                        for kp in kp_array2D
                            #println("In kp: ", kp)
                            Gk1 = SuperHF.Hubbard.Integral2D(k[1], k[2], iωn); Gk2 = SuperHF.Hubbard.Integral2D(kp[1]+qp[1], kp[2]+qp[2], iωn)
                            Gks1 = SuperHF.Hubbard.Integral2D(k[1], k[2], iωn); Gks2 = SuperHF.Hubbard.Integral2D(kp[1]+q[1], kp[2]+q[2], iωn)
                            Gks3 = SuperHF.Hubbard.Integral2D(kp[1], kp[2], iωn); Gks4 = SuperHF.Hubbard.Integral2D(k[1]-q[1], k[2]-q[2], iωn)
                            Matsubara_sus = Susceptibility(model, Gk1, Gk2, [Gks1,Gks2,Gks3,Gks4])
                            k_sum += Matsubara_sus
                        end
                    end
                end
                k_sum = 2.0*(1.0/(Grid_K)^2)^3*k_sum ## 2.0 is for the spin
                push!(Matsubara_array_susceptibility,k_sum)
                write(f, "$(iωn)"*"\t\t"*"$(k_sum)"*"\n")
                close(f)
            end
            tot_susceptibility = (2.0/model.beta_)^3*sum(Matsubara_array_susceptibility)
            println("total Susceptibility for q = $(q): ", tot_susceptibility)
            return tot_susceptibility
        catch err
            if typeof(err) == InterruptException
                println("ALL THE TASKS HAVE BEEN INTERRUPTED","\n")
            else
                println("$(err)")
            end
        end
        println("Program terminated. Have a nice day!")
        return nothing
    end
end

main()

