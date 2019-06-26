## Main translation unit
# Need to include Precompile.jl before running this unit
using Distributed

const N_procs = 8
@assert N_procs <= Base.Sys.CPU_THREADS "The number of workers must not exceed the amount of processors available."
addprocs(N_procs)

@assert N_procs == nworkers() "Too much workers available. Review recruiting methods."

@everywhere using JSON
@everywhere using SharedArrays

## Reading JSON file
@everywhere paramsJSON = "params.json"
@everywhere params = JSON.parsefile(paramsJSON)

## Some important parameters extracted from params.json file.
@everywhere dict = Dict{String,Float64}("U" => params["U"], "V" => params["V"])
@everywhere beta = params["beta"] # For 1D, beta = 100 and Niωn = 50 seems to converge well. For 1D, opt = "integral" gives results fast enough.
           # For 2D, beta = 200 and Niωn = 50 seems to stabilize efficiently the convergence loop. For 2D, opt = "sum" should be specified. (Incresing gap between beta > Niωn)
@everywhere Niωn = params["Niwn"] ## Niωn should absolutely be lower than beta value.
@everywhere dims = params["dims"]
@everywhere Grid_K = params["gridK"] ## Grid_K = 400 for 2D, as example! 2D case needs parallelization!!
@everywhere N_it = params["N_it"] ## Lowest number is 1: one loop in the process. Converges faster for 2D (~15 iterations) while for 1D slower (~30 iterations).
@everywhere precomputed_enabled = params["precomK"] ## Has effect only in the 2D case. To precompute the dispersion relation.

##
@everywhere SubLast = 2 ## Subdivision of last integral (N_it) to be split in #Sublast to be fed to different cores

@everywhere filename = "$(dims)D_HF_Susceptibility_calc_minus_sign_kGrid_$(Grid_K)_N_it_$(N_it)_beta_$(beta)_Niwn_$(Niωn)_U_$(dict["U"])_inner_loops.dat"
@everywhere filenameConv = "$(dims)D_Convergence_Self_kGrid_$(Grid_K)_N_it_$(N_it)_beta_$(beta)_Niwn_$(Niωn)_U_$(dict["U"])_inner_loops.dat"
@everywhere dataFolder = pwd()*"/data"; @everywhere superFilenameConv = dataFolder*"/"*filenameConv

## End of global parameters' definition

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

@assert (mod(SubLast,2) == 0) "Variable SubLast must be even for it to be splitable for the dispatch of the jobs to the processors. Deprecated."

@everywhere include("Precompile.jl")
@everywhere using .SuperHF

## Instantiating HubbardStruct for forthcoming calculations
@everywhere model = SuperHF.Hubbard.HubbardStruct(Niωn, dict, N_it, beta)

## Boundaries in k-space for the 1D system
@everywhere Boundaries1D = Array{Float64,1}([-pi,pi])
@everywhere qp_array1D = Array{Float64,1}([a for a in range(-pi,stop=pi,length=Grid_K)])
@everywhere k_array1D = qp_array1D; @everywhere kp_array1D = qp_array1D

## Boundaries in k-space for the 2D system
@everywhere Boundaries2D = Array{Array{Float64,1},1}([[-pi,-pi],[pi,pi]])

if !precomputed_enabled && dims == 2
    @everywhere qp_array2D = Array{Array{Float64,1},1}([[a,b] for a in range(-pi,stop=pi,length=Grid_K) for b in range(-pi,stop=pi,length=Grid_K)])
    @everywhere k_array2D = qp_array2D; @everywhere kp_array2D = qp_array2D
elseif precomputed_enabled && dims == 2
    @everywhere qp_array2D = Array{Array{Int64,1},1}([[a,b] for a in range(1,stop=Grid_K,length=Grid_K) for b in range(1,stop=Grid_K,length=Grid_K)])
    @everywhere k_array2D = qp_array2D; @everywhere kp_array2D = qp_array2D
end 

function f_pmap(f::Function, model::SuperHF.Hubbard.HubbardStruct)
    np = nworkers()
    println("nworkers: ", nworkers())
    Matsubara_array_susceptibility = SharedArray{Complex{Float64},1}(model.N_iωn_)
    fil = open(dataFolder*"/"*filename, "a")
    i=1
    nextidx() = (idx = i; i+=1; idx)

    @sync begin
        for pid = 1:np
            if pid != myid() || np == 1
                @async begin
                    while true
                        idx = nextidx()
                        if idx == 1
                            write(fil, "#N_it "*"$(model.N_it_)"*" q="*"$(qq)"*" Gridk "*"$(Grid_K)"*"\n")
                        end
                        if idx > model.N_iωn_
                            break
                        end
                        Matsubara_array_susceptibility[idx] = remotecall_fetch(f, pid, model.matsubara_grid_[idx])
                        println("iwn: ", model.matsubara_grid_[idx], "   ", Matsubara_array_susceptibility[idx])
                        write(fil, "$(model.matsubara_grid_[idx])"*"\t\t"*"$(Matsubara_array_susceptibility[idx])"*"\n")
                    end
                end
            end
        end
    end
    close(fil)
    return Matsubara_array_susceptibility
end

### Main 
@assert (dims in [1,2]) "dims must be 1 or 2. Only these dimensions have been implemented."
@everywhere dictFunct = dims == 1 ? SuperHF.Sus.iterationProcess(model, Boundaries1D, superFilenameConv, Gridk=Grid_K, opt="integral") : SuperHF.Sus.iterationProcess(model, Boundaries2D, superFilenameConv, Gridk=Grid_K, opt="sum")
try
    function main()
        funct_to_use = missing
        if dims == 1
            println("Length of function array: ", length(dictFunct[N_it]))
            @assert isa(dictFunct,Dict{Int64,Array{Array{Complex{Float64},2},1}}) "Dictionnary holding self-energies must have a given form. Look inside main function."
            @everywhere qq = params["q_1D"]

            @everywhere @time function oneDSpinSus(iωn::Complex{Float64})
                k_sum = 0.0 + 0.0im
                c_container = Vector{Array{Complex{Float64},2}}(undef,div(SubLast,2))
                for iqpn in model.matsubara_grid_bosons_
                    for qp in qp_array1D
                        for ikn in model.matsubara_grid_
                            for k in k_array1D
                                for ikpn in model.matsubara_grid_
                                    for kp in kp_array1D
                                        Gk1 = SuperHF.Hubbard.Integral1D(k, ikn); Gk2 = SuperHF.Hubbard.Integral1D(kp+qp, ikpn+iqpn)
                                        Gks1 = SuperHF.Hubbard.Integral1D(k, ikn); Gks2 = SuperHF.Hubbard.Integral1D(kp+qq, ikpn+iωn)
                                        Gks3 = SuperHF.Hubbard.Integral1D(kp, ikpn); Gks4 = SuperHF.Hubbard.Integral1D(k-qq, ikn-iωn)
                                        Matsubara_sus = SuperHF.Sus.Susceptibility(model, Gk1, Gk2, [Gks1,Gks2,Gks3,Gks4],c_container,dictFunct)
                                        k_sum += Matsubara_sus
                                    end
                                end
                            end
                        end
                    end
                end
                return 2.0*(1.0/(Grid_K))^3*k_sum
            end

            funct_to_use = oneDSpinSus

        elseif dims == 2
            println("Length of function array: ", length(dictFunct[N_it]))
            @assert isa(dictFunct,Dict{Int64,Array{Array{Complex{Float64},2},1}}) "Dictionnary holding self-energies must have a given form. Look inside main function."

            if !precomputed_enabled
                @everywhere qq = params["q_2D"]

                @everywhere @time function twoDSpinSus(iωn::Complex{Float64})
                    k_sum = 0.0 + 0.0im
                    c_container = Vector{Array{Complex{Float64},2}}(undef,div(SubLast,2))
                    for iqpn in model.matsubara_grid_bosons_
                        for qp in qp_array2D
                            for ikn in model.matsubara_grid_
                                for k in k_array2D
                                    for ikpn in model.matsubara_grid_
                                        for kp in kp_array2D
                                            Gk1 = SuperHF.Hubbard.Integral2D(k[1], k[2], ikn); Gk2 = SuperHF.Hubbard.Integral2D(kp[1]+qp[1], kp[2]+qp[2], ikpn+iqpn)
                                            Gks1 = SuperHF.Hubbard.Integral2D(k[1], k[2], ikn); Gks2 = SuperHF.Hubbard.Integral2D(kp[1]+q[1], kp[2]+qq[2], ikpn+iωn)
                                            Gks3 = SuperHF.Hubbard.Integral2D(kp[1], kp[2], ikpn); Gks4 = SuperHF.Hubbard.Integral2D(k[1]-q[1], k[2]-qq[2], ikpn-iωn)
                                            Matsubara_sus = SuperHF.Sus.Susceptibility(model, Gk1, Gk2, [Gks1,Gks2,Gks3,Gks4],c_container,dictFunct)
                                            k_sum += Matsubara_sus
                                        end
                                    end
                                end
                            end
                        end
                    end
                    return 2.0*(1.0/(Grid_K))^3*k_sum
                end

                funct_to_use = twoDSpinSus
                # Matsubara_array_susceptibility = f_pmap(twoDSpinSus,model)
                # tot_susceptibility = 2.0*(1.0/model.beta_)^3*sum(Matsubara_array_susceptibility)
                # println("total Susceptibility for q = $(q): ", tot_susceptibility)
                # f = open(dataFolder*"/"*filename, "a")
                # write(f, "total susceptibility at q=$(q): "*"$(tot_susceptibility)"*"\n")
                # close(f)
            else
                @everywhere q = [0,0] ## Has to be integer in order to refer to an element of the precomputed k-space. Equivalent to adding the [0.,0.] vector.
                
                @everywhere @time function twoDSpinSusPrecom(iωn::Complex{Float64})
                    bigMatSpace = SuperHF.Hubbard.BigKArray(SuperHF.Hubbard.epsilonk1, Boundaries2D, Grid_K)
                    k_sum = 0.0 + 0.0im
                    c_container = Vector{Array{Complex{Float64},2}}(undef,div(SubLast,2))
                    for iqpn in model.matsubara_grid_bosons_
                        for qp in qp_array2D
                            for ikn in model.matsubara_grid_
                                for k in k_array2D
                                    for ikpn in model.matsubara_grid_
                                        for kp in kp_array2D
                                            Gk1 = SuperHF.Hubbard.Integral2D(k[1], k[2], ikn); Gk2 = SuperHF.Hubbard.Integral2D(kp[1]+qp[1], kp[2]+qp[2], ikpn+iqpn)
                                            Gks1 = SuperHF.Hubbard.Integral2D(k[1], k[2], ikn); Gks2 = SuperHF.Hubbard.Integral2D(kp[1]+q[1], kp[2]+qq[2], ikpn+iωn)
                                            Gks3 = SuperHF.Hubbard.Integral2D(kp[1], kp[2], ikpn); Gks4 = SuperHF.Hubbard.Integral2D(k[1]-q[1], k[2]-qq[2], ikpn-iωn)
                                            Matsubara_sus = SuperHF.Sus.Susceptibility(model, Gk1, Gk2, [Gks1,Gks2,Gks3,Gks4],c_container,dictFunct,precomputedSpace=bigMatSpace,precom_enabled=precomputed_enabled,Gridk=Grid_K)
                                            k_sum += Matsubara_sus
                                        end
                                    end
                                end
                            end
                        end
                    end
                    return 2.0*(1.0/(Grid_K))^3*k_sum
                end
                
                funct_to_use = twoDSpinSusPrecom
                # Matsubara_array_susceptibility = f_pmap(twoDSpinSusPrecom,model)
                # tot_susceptibility = 2.0*(1.0/model.beta_)^3*sum(Matsubara_array_susceptibility)
                # println("total Susceptibility for q = $(q): ", tot_susceptibility)
                # f = open(dataFolder*"/"*filename, "a")
                # write(f, "total susceptibility at q=$(q): "*"$(tot_susceptibility)"*"\n")
                # close(f)
            end
        end
        Matsubara_array_susceptibility = f_pmap(funct_to_use,model)
        tot_susceptibility = 2.0*(1.0/model.beta_)^3*sum(Matsubara_array_susceptibility)
        println("total Susceptibility for q = $(qq): ", tot_susceptibility)
        f = open(dataFolder*"/"*filename, "a")
        write(f, "total susceptibility at q=$(qq): "*"$(tot_susceptibility)"*"\n")
        close(f)
    end

    main() ## Running main() here

catch err
    if typeof(err) == InterruptException
        println("ALL THE TASKS HAVE BEEN INTERRUPTED","\n")
        for pid in workers()
            interrupt(pid) ## Interrupting the tasks!
        end
    else
        println(err)
    end
    rmprocs(workers()) ## Freeing the workers!
    println("Program terminated. Have a nice day!")
end

