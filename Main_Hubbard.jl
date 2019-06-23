## Main translation unit
# Need to include Precompile.jl before running this unit
using JSON

## Reading JSON file
paramsJSON = "params.json"
params = JSON.parsefile(paramsJSON)

## Some important parameters extracted from params.json file.
dict = Dict{String,Float64}("U" => params["U"], "V" => params["V"])
beta = params["beta"] # For 1D, beta = 100 and Niωn = 50 seems to converge well. For 1D, opt = "integral" gives results fast enough.
           # For 2D, beta = 200 and Niωn = 50 seems to stabilize efficiently the convergence loop. For 2D, opt = "sum" should be specified. (Incresing gap between beta > Niωn)
Niωn = params["Niwn"] ## Niωn should absolutely be lower than beta value.
dims = params["dims"]
Grid_K = params["gridK"] ## Grid_K = 400 for 2D, as example! 2D case needs parallelization!!
N_it = params["N_it"] ## Lowest number is 1: one loop in the process. Converges faster for 2D (~15 iterations) while for 1D slower (~30 iterations).

##
SubLast = 2 ## Subdivision of last integral (N_it) to be split in #Sublast to be fed to different cores

filename = "$(dims)D_HF_Susceptibility_calc_minus_sign_kGrid_$(Grid_K)_N_it_$(N_it)_beta_$(beta)_Niwn_$(Niωn)_U_$(dict["U"]).dat"
filenameConv = "$(dims)D_Convergence_Self_kGrid_$(Grid_K)_N_it_$(N_it)_beta_$(beta)_Niwn_$(Niωn)_U_$(dict["U"]).dat"
dataFolder = pwd()*"/data"; superFilenameConv = dataFolder*"/"*filenameConv

if !isdir(dataFolder)
    mkdir(dataFolder, mode=0o777)
end

if isfile(dataFolder*"/"*filename) || isfile(superFilenameConv)
    try
        rm(dataFolder*"/"*filename); rm(superFilenameConv)
    catch err
        nothing
    end
end


@assert (mod(SubLast,2) == 0) "Variable SubLast must be even for it to be splitable for the dispatch of the jobs to the processors."

include("Precompile.jl")
using .SuperHF


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


### Main 
c_container = Vector{Array{Complex{Float64},2}}(undef,div(SubLast,2))
@assert (dims in [1,2]) "dims must be 1 or 2. Only these dimensions have been implemented."
dictFunct = dims == 1 ? SuperHF.Sus.iterationProcess(model, Boundaries1D, superFilenameConv, Gridk=Grid_K, opt="integral") : SuperHF.Sus.iterationProcess(model, Boundaries2D, superFilenameConv, Gridk=Grid_K, opt="sum")
println("dictFunct: ", dictFunct)
try
    function main()
        if dims == 1
            #dictFunct = iterationProcess(model, Boundaries1D, Gridk=Grid_K, opt="integral")
            println("Length of function array: ", length(dictFunct[N_it]))
            @assert isa(dictFunct,Dict{Int64,Array{Array{Complex{Float64},2},1}}) "Dictionnary holding self-energies must have a given form. Look inside main function."
            Matsubara_array_susceptibility = Array{Complex{Float64},1}()
            q = 0.
            @time for (ii,iωn) in enumerate(model.matsubara_grid_)
                f = open(dataFolder*"/"*filename, "a")
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
                            Matsubara_sus = SuperHF.Sus.Susceptibility(model, Gk1, Gk2, [Gks1,Gks2,Gks3,Gks4],c_container,dictFunct)
                            k_sum += Matsubara_sus
                        end
                    end
                end
                k_sum = 2.0*(1.0/(Grid_K))^3*k_sum ## 2.0 is for the spin
                push!(Matsubara_array_susceptibility,k_sum)
                write(f, "$(iωn)"*"\t\t"*"$(k_sum)"*"\n")
                close(f)
            end
            tot_susceptibility = 2.0*(1.0/model.beta_)^3*sum(Matsubara_array_susceptibility)
            println("total Susceptibility for q = $(q): ", tot_susceptibility)
            f = open(dataFolder*"/"*filename, "a")
            write(f, "total susceptibility at q=$(q): "*"$(tot_susceptibility)"*"\n")
            close(f)
            return nothing
        elseif dims == 2
            #dictFunct = iterationProcess(model, Boundaries2D, Gridk=Grid_K, opt="sum")
            println("Length of function array: ", length(dictFunct[N_it]))
            @assert isa(dictFunct,Dict{Int64,Array{Array{Complex{Float64},2},1}}) "Dictionnary holding self-energies must have a given form. Look inside main function."
            Matsubara_array_susceptibility = Array{Complex{Float64},1}()
            q = [0.,0.]
            @time for (ii,iωn) in enumerate(model.matsubara_grid_)
                f = open(dataFolder*"/"*filename, "a")
                if ii == 1
	                write(f, "#N_it "*"$(N_it)"*" q="*"$(q)"*" Gridk "*"$(Grid_K)"*"\n")
                end
                k_sum = 0.0+0.0im
                println("iwn: ", iωn)
                for qp in qp_array2D
                    println("In qp: ", qp)
                    for k in k_array2D
                        println("In k: ", k)
                        for kp in kp_array2D
                            #println("In kp: ", kp)
                            Gk1 = SuperHF.Hubbard.Integral2D(k[1], k[2], iωn); Gk2 = SuperHF.Hubbard.Integral2D(kp[1]+qp[1], kp[2]+qp[2], iωn)
                            Gks1 = SuperHF.Hubbard.Integral2D(k[1], k[2], iωn); Gks2 = SuperHF.Hubbard.Integral2D(kp[1]+q[1], kp[2]+q[2], iωn)
                            Gks3 = SuperHF.Hubbard.Integral2D(kp[1], kp[2], iωn); Gks4 = SuperHF.Hubbard.Integral2D(k[1]-q[1], k[2]-q[2], iωn)
                            Matsubara_sus = SuperHF.Sus.Susceptibility(model, Gk1, Gk2, [Gks1,Gks2,Gks3,Gks4],c_container,dictFunct)
                            k_sum += Matsubara_sus
                        end
                    end
                end
                k_sum = 2.0*(1.0/(Grid_K)^2)^3*k_sum ## 2.0 is for the spin
                push!(Matsubara_array_susceptibility,k_sum)
                write(f, "$(iωn)"*"\t\t"*"$(k_sum)"*"\n")
                close(f)
            end
            tot_susceptibility = 2.0*(1.0/model.beta_)^3*sum(Matsubara_array_susceptibility)
            println("total Susceptibility for q = $(q): ", tot_susceptibility)
            f = open(dataFolder*"/"*filename, "a")
            write(f, "total susceptibility at q=$(q): "*"$(tot_susceptibility)"*"\n")
            close(f)
            return nothing
        end
    end

    main() ## Running main() here

catch err
    if typeof(err) == InterruptException
        println("ALL THE TASKS HAVE BEEN INTERRUPTED","\n")
    else
        println(err)
    end
    println("Program terminated. Have a nice day!")
end

