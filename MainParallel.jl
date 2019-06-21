## Main translation unit
# Need to include Precompile.jl before running this unit
using Distributed

## Some important parameters
dict = Dict{String,Float64}("U" => 4.0, "V" => 1.0)
beta = 200; Niωn = 50
N_it = 3 ## Lowest number is 1: one loop in the process
SubLast = 1 ## Subdivision of last integral (N_it) to be split in #Sublast to be fed to different cores
Full = false ## If you want to compute the results of all the iterations, set to true. Set to false otherwise!
Grid_K = 50
Dims = 1

@everywhere include("Precompile.jl")

if Full
    SuperHF.Hubbard.@assert (N_it*SubLast) <= Base.Sys.CPU_THREADS # Maximum number of cores
    addprocs(N_it*SubLast) ## defines the number of workers (cores)
else
    SuperHF.Hubbard.@assert (SubLast) <= Base.Sys.CPU_THREADS # Maximum number of cores
    addprocs(SubLast)
end

function Gk_conv(vals::Union{SuperHF.Green_Utils.Integral,SuperHF.Green_Utils.Integral1D})
    if isa(vals, SuperHF.Green_Utils.Integral)
        @time @sync for (idx,pid) in enumerate(workers())
            @async c_container[idx] = remotecall_fetch(Dict_funct[N_it][idx], pid, vals) ## To test
        end
        summation_subs = sum(c_container)
        println("Summation: ", summation_subs)

        Gk = 1.0/(vals.iωn - SuperHF.Green_Utils.epsilonk(vals.kx, vals.ky) - summation_subs)

        return Gk
    elseif isa(vals, SuperHF.Green_Utils.Integral1D)
        println("In gk_conv 1D")
        @time @sync for (idx,pid) in enumerate(workers())
            @async c_container[idx] = remotecall_fetch(Dict_funct[N_it][idx], pid, vals) ## To test
        end
        println("Elements: ", c_container)
        summation_subs = sum(c_container)
        println("Summation: ", summation_subs)

        Gk = 1.0/(vals.iωn - SuperHF.Green_Utils.epsilonk1D(vals.kx) - summation_subs)

        return Gk
    end
end

function Lambda(q::Array{Float64,1}, qp::Array{Float64,1}, qpp::Array{Float64,1}, k::Array{Float64,1}, kp::Array{Float64,1}, iωn::Complex{Float64}, HF::SuperHF.Green_Utils.HFVec, Gk1::SuperHF.Green_Utils.Integral, Gk2::SuperHF.Green_Utils.Integral)
    println("IN LAMBDA FUNCTION", "\n\n")

    kernel = inv( inv(SuperHF.Green_Utils.Potential(q[1]-qp[1], q[2]-qp[2], values(HF.dict_v_)...)) +  
                inv(SuperHF.Green_Utils.Potential(qpp[1], qpp[2], values(HF.dict_v_)...))*SuperHF.Green_Utils.Potential(q[1]-qp[1]-qpp[1], q[2]-qp[2]-qpp[2], values(HF.dict_v_)...)*Gk_conv(Gk1)*Gk_conv(Gk2) )

    println("kernel value: ", kernel, "\n")
    return kernel
end

function Lambda1D(q::Float64, qp::Float64, k::Float64, kp::Float64, iωn::Complex{Float64}, HF::SuperHF.Green_Utils.HFVec, Gk1::SuperHF.Green_Utils.Integral1D, Gk2::SuperHF.Green_Utils.Integral1D)
    println("IN LAMBDA FUNCTION 1D", "\n\n")

    kernel = inv( inv(SuperHF.Green_Utils.Potential1D(q, values(HF.dict_v_)...)) +  
                inv(SuperHF.Green_Utils.Potential1D(qp, values(HF.dict_v_)...))*SuperHF.Green_Utils.Potential1D(q-qp, values(HF.dict_v_)...)*Gk_conv(Gk1)*Gk_conv(Gk2) )

    println("kernel value 1D: ", kernel, "\n")
    return kernel
end


function Susceptibility(q::Array{Float64,1}, qp::Array{Float64,1}, qpp::Array{Float64,1}, k::Array{Float64,1}, kp::Array{Float64,1}, iωn::Complex{Float64}, HF::SuperHF.Green_Utils.HFVec, Gk1::SuperHF.Green_Utils.Integral, Gk2::SuperHF.Green_Utils.Integral, Gks::Array{SuperHF.Green_Utils.Integral,1})
    println("IN SUSCEPTIBILITY FUNCTION", "\n\n")
    sus = Gk_conv(Gks[1])*SuperHF.Green_Utils.Potential(qp[1], qp[2], values(HF.dict_v_)...)*Gk_conv(Gks[2])*Lambda(q,qp,qpp,k,kp,iωn,HF,Gk1,Gk2)*Gk_conv(Gks[3])*Gk_conv(Gks[4])*Gk_conv(Gks[5])*Gk_conv(Gks[6])

    println("Susceptibility: ", sus, "\n\n")
    return sus
end

function Susceptibility1D(q::Float64, qp::Float64, k::Float64, kp::Float64, iωn::Complex{Float64}, HF::SuperHF.Green_Utils.HFVec, Gk1::SuperHF.Green_Utils.Integral1D, Gk2::SuperHF.Green_Utils.Integral1D, Gks::Array{SuperHF.Green_Utils.Integral1D,1})
    println("IN SUSCEPTIBILITY FUNCTION 1D", "\n\n")
    sus = Gk_conv(Gks[1])*SuperHF.Green_Utils.Potential1D(qp, values(HF.dict_v_)...)*Lambda1D(q,qp,k,kp,iωn,HF,Gk1,Gk2)*Gk_conv(Gks[2])*Gk_conv(Gks[3])*Gk_conv(Gks[4])

    println("Susceptibility 1D: ", sus, "\n\n")
    return sus
end

## Initializing the struct members ##
#####################################
model = SuperHF.Green_Utils.HFVec(Niωn,beta,N_it,dict)
Boundaries = Array{Array{Float64,1},1}([[-pi,-pi],[pi,pi]])
qp_array = Array{Array{Float64,1},1}([[a,b] for a in range(-pi,stop=pi,length=Grid_K) for b in range(-pi,stop=pi,length=Grid_K)])
qpp_array = qp_array; k_array = qp_array; kp_array = qp_array
Boundaries1D = Array{Float64,1}([-pi,pi])
qp_array1D = Array{Float64,1}([a for a in range(-pi,stop=pi,length=Grid_K)])
qpp_array1D = qp_array1D; k_array1D = qp_array1D; kp_array1D = qp_array1D
    
intermediate_integral = SuperHF.Green_Utils.integrate_complex(SuperHF.Green_Utils.Init_Gk, x -> x/2 , 1, model, Boundaries)
println(intermediate_integral(SuperHF.Green_Utils.Integral(pi/4,pi/3,0.01im)))

if Dims == 2
    Dict_funct = SuperHF.Green_Utils.interation_process(model,Boundaries,SubLast)
    if Full
        println("Length of function array: ", length(Dict_funct)*length(Dict_funct[1]), " and number of workers: ", nworkers())
        @assert isa(Dict_funct,Dict{Int64,Array{Function,1}}) && (length(Dict_funct)*length(Dict_funct[1]) == nworkers())
        c_container = Vector{Complex{Float64}}(undef,N_it*SubLast)

        function main()
            ## Part where the convergence (behavior) is checked ##
            ######################################################
            try
                for (ii,iωn) in enumerate(model.Matsubara_G_)
                    if ii == 1
                        println("iwn: ", iωn)
                        val = SuperHF.Green_Utils.Integral(pi, pi/2, iωn)
                        @time @sync for (idx,pid) in enumerate(workers())
                            @async c_container[idx] = remotecall_fetch(Dict_funct[ceil(Int64,idx/N_it)][ceil(Int64,idx/SubLast)], pid, val) ## To test
                            #mapping = pmap((f,x)->f(x), Funct_arr, vals)
                        end
                        summation_subs = sum(c_container[end-SubLast:end])
                        println("Summation last terms: ", summation_subs)
                    end
                end
            catch err
                if typeof(err) == InterruptException
                    println("ALL THE TASKS HAVE BEEN INTERRUPTED","\n")
                    for pid in workers()
                        interrupt(pid)
                    end
                end
            end
            println("Program terminated. Have a nice day!")
            rmprocs(workers())
            return nothing
        end
    else
        c_container = Vector{Complex{Float64}}(undef,SubLast)
        println("Length of function array: ", length(Dict_funct[N_it]), " and number of workers: ", nworkers())
        @assert isa(Dict_funct,Dict{Int64,Array{Function,1}}) && (length(Dict_funct[N_it]) == nworkers())
        function main()
            try
                Matsubara_array_susceptibility = Array{Complex{Float64},1}()
                q = [0.,0.]
                for (ii,iωn) in enumerate(model.Matsubara_G_)
                    k_sum = 0.0+0.0im
                    println("iwn: ", iωn)
                    for qp in qp_array
                        println("In qp: ", qp[1], " ", qp[2])
                        for qpp in qpp_array
                            println("In qpp: ", qpp[1], " ", qpp[2])
                            for k in k_array
                                println("In k: ", k[1], " ", k[2])
                                for kp in kp_array
                                    println("In kp: ", kp[1], " ", kp[2])
                                    Gk1 = SuperHF.Green_Utils.Integral(k[1], k[2], iωn); Gk2 = SuperHF.Green_Utils.Integral(kp[1]+qpp[1]-q[1], kp[2]+qpp[2]-q[2], iωn)
                                    Gks1 = SuperHF.Green_Utils.Integral(k[1]-qpp[1]+q[1], k[2]-qpp[2]+q[2], iωn); Gks2 = SuperHF.Green_Utils.Integral(k[1]-qpp[1]-qp[1]+q[1], k[2]-qpp[2]-qp[2]+q[2], iωn)
                                    Gks3 = SuperHF.Green_Utils.Integral(kp[1]-qp[1], kp[2]-qp[2], iωn); Gks4 = SuperHF.Green_Utils.Integral(kp[1], kp[2], iωn)
                                    Gks5 = SuperHF.Green_Utils.Integral(kp[1]-q[1], kp[2]-q[2], iωn); Gks6 = SuperHF.Green_Utils.Integral(k[1]-qpp[1], k[2]-qpp[2], iωn)
                                    Matsubara_sus = Susceptibility(q, qp, qpp, k, kp, iωn, model, Gk1, Gk2, [Gks1,Gks2,Gks3,Gks4,Gks5,Gks6])
                                    k_sum += Matsubara_sus
                                end
                            end
                        end
                        push!(Matsubara_array_susceptibility,2.0*(1.0/(Grid_K^2))^4*k_sum)
                    end
                end
                tot_susceptibility = (1/model.beta_)^4*sum(Matsubara_array_susceptibility)
                return tot_susceptibility
            catch err
                if typeof(err) == InterruptException
                    println("ALL THE TASKS HAVE BEEN INTERRUPTED","\n")
                    for pid in workers()
                        interrupt(pid)
                    end
                end
            end
            println("Program terminated. Have a nice day!")
            rmprocs(workers())
            return nothing
        end
    end
elseif Dims == 1
    Dict_funct = SuperHF.Green_Utils.interation_process1D(model,Boundaries1D,SubLast,Grid_K)
    c_container = Vector{Complex{Float64}}(undef,SubLast)
    println("Length of function array: ", length(Dict_funct[N_it]), " and number of workers: ", nworkers())
    @assert isa(Dict_funct,Dict{Int64,Array{Function,1}}) && (length(Dict_funct[N_it]) == nworkers())
    function main()
        try
            Matsubara_array_susceptibility = Array{Complex{Float64},1}()
            q = 0.
            for (ii,iωn) in enumerate(model.Matsubara_G_)
                f = open("1D_HF_Susceptibility_calc_first.dat", "a")
                if ii == 1
                    write(f, "#N_it "*"$(N_it)"*" q="*"$(q)"*"\n")
                end
                k_sum = 0.0+0.0im
                println("iwn: ", iωn)
                for qp in qp_array1D
                    println("In qp: ", qp)
                    for k in k_array1D
                        println("In k: ", k)
                        for kp in kp_array1D
                            println("In kp: ", kp)
                            Gk1 = SuperHF.Green_Utils.Integral1D(k, iωn); Gk2 = SuperHF.Green_Utils.Integral1D(kp+qp, iωn)
                            Gks1 = SuperHF.Green_Utils.Integral1D(k-qp+q, iωn); Gks2 = SuperHF.Green_Utils.Integral1D(kp+q, iωn)
                            Gks3 = SuperHF.Green_Utils.Integral1D(kp, iωn); Gks4 = SuperHF.Green_Utils.Integral1D(k-qp, iωn)
                            Matsubara_sus = Susceptibility1D(q, qp, k, kp, iωn, model, Gk1, Gk2, [Gks1,Gks2,Gks3,Gks4])
                            k_sum += Matsubara_sus
                        end
                    end
                end
                k_sum = 2.0*(1.0/(Grid_K))^3*k_sum
                push!(Matsubara_array_susceptibility,k_sum)
                write(f, "$(iωn)"*"\t\t"*"$(k_sum)")
                close(f)
            end
            tot_susceptibility = (1/model.beta_)^3*sum(Matsubara_array_susceptibility)
            return tot_susceptibility
        catch err
            if typeof(err) == InterruptException
                println("ALL THE TASKS HAVE BEEN INTERRUPTED","\n")
                for pid in workers()
                    interrupt(pid)
                end
            end
        end
        println("Program terminated. Have a nice day!")
        rmprocs(workers())
        return nothing
    end
end

main()
