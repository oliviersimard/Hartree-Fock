module Hubbard

using QuadGK
using Cubature: hcubature

II = Matrix{Float64}([1.0 0.0; 0.0 1.0])

abstract type MyType{N} end
struct Integral2D <: MyType{2}
    qx::Float64
    qy::Float64
    iωn::Complex{Float64}
end
struct Integral1D <: MyType{1}
    qx::Float64
    iωn::Complex{Float64}
end

mutable struct HubbardStruct
    N_iωn_::Int64 ## Number of  Matsubara frequencies
    U_::Float64 ## Hubbard interaction
    N_it_::Int64 ## Number of iteration in Hartree-Fock self-consistency loop
    beta_::Int64 ## Inverse temperature
    matsubara_grid_::Array{Complex{Float64},1}  ## Matsubara matsubara_grid_

    function HubbardStruct(N_iωn_::Int64, U_::Float64, N_it_::Int64, beta_::Int64)
        matsubara_grid_ = matsubaraGrid(N_iωn_,beta_)

        return new(N_iωn_, U_, N_it_, beta_, matsubara_grid_)
    end
end


function matsubaraGrid(N_iωn_::Int64, beta_::Int64)
    Niωn_ = mod(Niωn_,2) == 0 ? Niωn_ : Niωn_ + 1
    matsubara_grid_ = Array{Complex{Float64},1}(undef,N_iωn_)
    matsubara_grid_ = 1.0im*[(2*n + 1)*pi/beta_ for n in 1:N_iωn_]

    return matsubara_grid_
end

function swap(a::Matrix{Complex{T}}) where {T<:Number}
    tmp_matrix = Matrix{Complex{T}}(undef,(2,2))
    tmp_matrix[1,1] = a[2,2]; tmp_matrix[2,2] = a[1,1]
    tmp_matrix[1,2] = tmp_matrix[2,1] = Complex{T}(0)

    return tmp_matrix
end

function epsilonk(kk::Union{Float64,Array{Float64,1}}; t::Float64=1.0, tp::Float64=-0.3, tpp::Float64=0.2)
    if isa(kk,Float64)

        return -2.0*t*cos(kk)
    elseif isa(kk,Array{Float64,1})
        kx, ky = kk
        epsilonk1 = -2.0*t*(cos(kx) + cos(ky)) ## Nearest neighbors
        epsilonk2 = -2.0*tp*(cos(kx+ky)+cos(kx-ky)) ## Second-nearest neighbors
        epsilonk3 = -2.0*tpp*(cos(2.0*kx)+cos(2.0*ky)) ## Third-nearest neighbors

        return epsilonk1
    else

        throw(ErrorException("Not a type handled. Check epsilonk function!"))
        exit()
    end

end

function initGk(qq::Union{Float64,Array{Float64,1}}, kk::Union{Float64,Array{Float64,1}}, iωn::Complex{Float64})
    initGkTemp = Matrix{Complex{Float64}}(undef,(2,2)) ## Spin degrees of freedom 
    if isa(kk,Float64) && isa(qq,Float64)
        initGkTemp[1,1] = 1.0/(iωn - epsilonk(kk) - (0.1-0.0im)) ## Setting initial self-energy slightly different between spin components 
        initGkTemp[2,2] = 1.0/(iωn - epsilonk(kk) - (0.11-0.0im))
        initGkTemp[1,2] = initGkTemp[2,1] = 0.0 + 0.0im
    elseif isa(qq,Array{Float64,1}) && isa(kk,Array{Float64,1})
        kx, ky = kk; qx, qy = qq
        initGkTemp[1,1] = 1.0/(iωn - epsilonk(kx, ky) - (0.1-0.0im))
        initGkTemp[2,2] = 1.0/(iωn - epsilonk(kx, ky) - (0.11-0.0im))
        initGkTemp[1,2] = initGkTemp[2,1] = 0.0 + 0.0im
    else
        throw(ErrorException("Not a type handled. Check initGk function!"))
        exit()
    end

    return initGkTemp
end

function Gk(qq::Union{Float64,Array{Float64,1}}, kk::Union{Float64,Array{Float64,1}}, iωn::Complex{Float64}, SE_func::Matrix{Complex{Float64}})
    gkTemp = Matrix{Complex{Float64}}(undef,(2,2)) ## Spin degrees of freedom 
    if isa(kk,Float64) && isa(qq,Float64)
        gkTemp = inv(iωn*II - epsilonk(kk + qq)*II - SE_func)
    else
        throw(ErrorException("Not a type handled. Check Gk function!"))
        exit()
    end
    
    return gkTemp
end


function FunctWrapper1D(funct::Function, other::Complex{Float64})
    function Inner_funct(k::Float64)
        return funct(k, other)
    end
    
    return Inner_funct
end

function FunctWrapper2D(funct::Function, other::Complex{Float64})
    function Inner_funct(k::Array{Float64,1})
        return funct(k[1], k[2], other)
    end

    return Inner_funct
end


function integrateComplex(funct::N, SE_funct::Matrix{Complex{Float64}}, ii::Int64, structModel::HubbardStruct, BoundArr::Union{Array{Float64,1},Array{Array{Float64,1},1}}; Gridk::Int64=80, opt::String="sum") where {N<:Function}
    U = structModel.U_
    if isa(BoundArr,Array{Float64,1})
        if ii <= 1
            println(ii, " integrate_complex 1D")
            dressed_funct = (qx::Float64, kx::Float64, iωn::Complex{Float64})->1.0*U*funct(qx, kx, iωn)
        elseif ii > 1
            println(ii, " integrate_complex 1D")
            dressed_funct = (qx::Float64, kx::Float64, iωn::Complex{Float64})->1.0*U*funct(qx, kx, iωn, SE_funct)
        end
    end


    function real_funct(kx::Float64, ky::Float64, iωn::Complex{Float64})
        return real(dressed_funct(kx, ky, 0.0, 0.0, iωn))
    end

    function real_funct(kx::Float64, iωn::Complex{Float64})
        return real(dressed_funct(kx, 0.0, iωn))
    end

    function imag_funct(kx::Float64, ky::Float64, iωn::Complex{Float64})
        return imag(dressed_funct(kx, ky, 0.0, 0.0, iωn))
    end

    function imag_funct(kx::Float64, iωn::Complex{Float64})
        return imag(dressed_funct(kx, 0.0, iωn))
    end

    if isa(BoundArr,Array{Array{Float64,1},1})
        result_real = (remaining_var::Integral2D) -> 2.0*(2.0*pi)^(-2.0)*hcubature(FunctWrapper2D(real_funct,remaining_var), BoundArr[1], BoundArr[2]; reltol=1.5e-2, abstol=1.5e-2, maxevals=100_000)[1]
        result_imag = (remaining_var::Integral2D) -> 2.0*(2.0*pi)^(-2.0)*hcubature(FunctWrapper2D(imag_funct,remaining_var), BoundArr[1], BoundArr[2]; reltol=1.5e-2, abstol=1.5e-2, maxevals=100_000)[1]
    elseif isa(BoundArr,Array{Float64,1})
        @assert opt == "sum" || opt == "integral"
        if opt == "integral"
            result_real = (remaining_var::Complex{Float64}) -> 2.0*(2.0*pi)^(-1.0)*quadgk(FunctWrapper1D(real_funct,remaining_var), BoundArr[1], BoundArr[2]; rtol=1.5e-4, atol=1.5e-4, maxevals=100_000)[1]
            result_imag = (remaining_var::Complex{Float64}) -> 2.0*(2.0*pi)^(-1.0)*quadgk(FunctWrapper1D(imag_funct,remaining_var), BoundArr[1], BoundArr[2]; rtol=1.5e-4, atol=1.5e-4, maxevals=100_000)[1]
        elseif opt == "sum"
            result_real = (remaining_var::Complex{Float64}) -> 2.0*(Gridk)^(-1.0)*sum(FunctWrapper1D(real_funct,remaining_var), range(BoundArr[1], stop=BoundArr[2], length=Gridk))
            result_imag = (remaining_var::Complex{Float64}) -> 2.0*(Gridk)^(-1.0)*sum(FunctWrapper1D(imag_funct,remaining_var), range(BoundArr[1], stop=BoundArr[2], length=Gridk))
        end
    end

    temp_func = (KK::Complex{Float64}) -> (result_real(KK) + 1.0im*result_imag(KK))

    return temp_func
end

function interationProcess(structModel::HubbardStruct, BoundArr::Union{Array{Float64,1},Array{Array{Float64,1},1}}, SubLast::Int64; Gridk::Int64=80, opt::String="sum")
    @assert isa(BoundArr,Array{Array{Float64,1},1}) || isa(BoundArr,Array{Float64,1})
    N_it = structModel.N_it_
    SE_funct = missing
    dictArray = Dict{Int64,Array{Function,1}}()
    if isa(BoundArr,Array{Float64,1})
        for it in 1:N_it    
            Funct_array = Array{Function,1}([])
            if it <= 1
                for sub in 1:SubLast
                    println(it, " interation_process 1D")
                    to_add_lower = (convert(Float64,sub)-1.)*2.0*pi/SubLast
                    to_add_upper = (convert(Float64,sub))*2.0*pi/SubLast
                    BoundArr = Array{Float64,1}([-pi+to_add_lower,-pi+to_add_upper])
                    println("it: ", it, "BoundArr1D: ", BoundArr)
                    push!(Funct_array, integrateComplex(initGk, x -> x, it, structModel, BoundArr, Gridk=Gridk, opt=opt))
                end
            elseif it > 1
                for sub in 1:SubLast
                    println(it, " interation_process 1D")
                    to_add_lower = (convert(Float64,sub)-1.)*2.0*pi/SubLast
                    to_add_upper = (convert(Float64,sub))*2.0*pi/SubLast
                    BoundArr = Array{Float64,1}([-pi+to_add_lower,-pi+to_add_upper])
                    println("BoundArr1D: ", BoundArr)
                    push!(Funct_array, integrateComplex(Gk, dictArray[it-1][sub], it, structModel, BoundArr, Gridk=Gridk, opt=opt))
                end
            end
            dictArray[it] = Funct_array
        end
    end
    return dictArray
end

end ## End module Hubbard