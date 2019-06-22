module Hubbard

using QuadGK
using Cubature: hcubature

t_ = 1.0; tp_ = -0.3; tpp_ = 0.2

II = Matrix{Float64}([1.0 0.0; 0.0 1.0])

abstract type MyType{N} end
struct Integral2D <: MyType{2}
    qx::Union{Float64,Int64}
    qy::Union{Float64,Int64}
    iωn::Complex{Float64}
end
struct Integral1D <: MyType{1}
    qx::Float64
    iωn::Complex{Float64}
end

mutable struct HubbardStruct
    N_iωn_::Int64 ## Number of  Matsubara frequencies
    dict_::Dict{String,Float64} ## Hubbard interaction
    N_it_::Int64 ## Number of iteration in Hartree-Fock self-consistency loop
    beta_::Int64 ## Inverse temperature
    matsubara_grid_::Array{Complex{Float64},1}  ## Matsubara matsubara_grid_

    function HubbardStruct(N_iωn_::Int64, dict_::Dict{String,Float64}, N_it_::Int64, beta_::Int64)
        matsubara_grid_ = matsubaraGrid(N_iωn_,beta_)

        return new(N_iωn_, dict_, N_it_, beta_, matsubara_grid_)
    end
end


function matsubaraGrid(N_iωn_::Int64, beta_::Int64)
    N_iωn_ = mod(N_iωn_,2) == 0 ? N_iωn_ : N_iωn_ + 1
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

function epsilonk1(kx::Float64, ky::Float64, t::Float64)
    return -2.0*t*(cos(kx) + cos(ky))
end

function epsilonk2(kx::Float64, ky::Float64, tp::Float64)
    return -2.0*tp*(cos(kx+ky)+cos(kx-ky))
end

function epsilonk3(kx::Float64, ky::Float64, tpp::Float64)
    return -2.0*tpp*(cos(2.0*kx)+cos(2.0*ky))
end

function epsilonk(kk::Union{Float64,Array{Float64,1}}; t::Float64=t_, tp::Float64=tp_, tpp::Float64=tpp_)
    if isa(kk,Float64)

        return -2.0*t*cos(kk)
    elseif isa(kk,Array{Float64,1})
        kx, ky = kk
        epsilonk1Var = epsilonk1(kx,ky,t) ## Nearest neighbors
        epsilonk2Var = epsilonk2(kx,ky,tp) ## Second-nearest neighbors
        epsilonk3Var = epsilonk3(kx,ky,tpp) ## Third-nearest neighbors

        return epsilonk1Var
    else

        throw(ErrorException("Not a type handled. Check epsilonk function!"))
        exit()
    end
end

function BigKArray(funct::Function, Boundaries::Array{Array{Float64,1},1}, Gridk::Int64; t::Float64=t_, tp::Float64=tp_, tpp::Float64=tpp_)
    tmpMat = Matrix{Float64}(undef,(Gridk,Gridk))
    for (i,ky) in enumerate(range(Boundaries[1][1],stop=Boundaries[2][1],length=Gridk))
        for (j,kx) in enumerate(range(Boundaries[1][2],stop=Boundaries[2][2],length=Gridk))
            tmpMat[i,j] = funct(kx,ky,t) ## Be careful here, because different dispersion relations would lead to mistakes.
        end
    end
    return tmpMat
end

function initGk(kk::Union{Float64,Array{Float64,1}}, qq::Union{Float64,Array{Float64,1}}, iωn::Complex{Float64})
    initGkTemp = Matrix{Complex{Float64}}(undef,(2,2)) ## Spin degrees of freedom 
    if isa(kk,Float64) && isa(qq,Float64)
        initGkTemp[1,1] = 1.0/(iωn - epsilonk(kk) - (0.1-0.0im)) ## Setting initial self-energy slightly different between spin components 
        initGkTemp[2,2] = 1.0/(iωn - epsilonk(kk) - (0.15-0.0im))
        initGkTemp[1,2] = initGkTemp[2,1] = 0.0 + 0.0im
    elseif isa(qq,Array{Float64,1}) && isa(kk,Array{Float64,1})
        initGkTemp[1,1] = 1.0/(iωn - epsilonk(kk.+qq) - (0.1-0.0im))
        initGkTemp[2,2] = 1.0/(iωn - epsilonk(kk.+qq) - (0.15-0.0im))
        initGkTemp[1,2] = initGkTemp[2,1] = 0.0 + 0.0im
    else
        throw(ErrorException("Not a type handled. Check initGk function!"))
        exit()
    end

    return initGkTemp
end

function Gk(kk::Union{Float64,Array{Float64,1}}, qq::Union{Float64,Array{Float64,1}}, iωn::Complex{Float64}, SE_func::Matrix{Complex{Float64}})
    gkTemp = Matrix{Complex{Float64}}(undef,(2,2)) ## Spin degrees of freedom 
    if (isa(kk,Float64) && isa(qq,Float64)) || (isa(kk,Array{Float64,1}) && isa(qq,Array{Float64,1}))
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
        return funct(k, other)
    end
    
    return Inner_funct
end

function integrateComplex(funct::N, SE_funct::Matrix{Complex{Float64}}, ii::Int64, structModel::HubbardStruct, BoundArr::Union{Array{Float64,1},Array{Array{Float64,1},1}}; 
    Gridk::Int64=100, opt::String="sum") where {N<:Function}
    U = structModel.dict_["U"]
    if isa(BoundArr,Array{Float64,1})
        if ii <= 1
            println(ii, " integrate_complex 1D")
            dressed_funct = (kx::Float64, qx::Float64, iωn::Complex{Float64})->1.0*U*funct(kx, qx, iωn)
        elseif ii > 1
            println(ii, " integrate_complex 1D")
            dressed_funct = (kx::Float64, qx::Float64, iωn::Complex{Float64})->1.0*U*funct(kx, qx, iωn, SE_funct)
        end
    elseif isa(BoundArr,Array{Array{Float64,1},1})
        if ii <= 1
            println(ii, " integrate_complex 2D")
            dressed_funct = (kk::Array{Float64,1}, qq::Array{Float64,1}, iωn::Complex{Float64})->1.0*U*funct(kk, qq, iωn)
        elseif ii > 1
            println(ii, " integrate_complex 2D")
            dressed_funct = (kk::Array{Float64,1}, qq::Array{Float64,1}, iωn::Complex{Float64})->1.0*U*funct(kk, qq, iωn, SE_funct)
        end
    end

    function real_funct(kx::Float64, iωn::Complex{Float64})
        return real.(dressed_funct(kx, 0.0, iωn))
    end

    function real_funct(kk::Array{Float64,1}, iωn::Complex{Float64})
        return real.(dressed_funct(kk, [0.0,0.0], iωn))
    end

    function imag_funct(kx::Float64, iωn::Complex{Float64})
        return imag.(dressed_funct(kx, 0.0, iωn))
    end

    function imag_funct(kk::Array{Float64,1}, iωn::Complex{Float64})
        return imag.(dressed_funct(kk, [0.0,0.0], iωn))
    end

    @assert opt == "sum" || opt == "integral"
    if isa(BoundArr,Array{Float64,1})
        if opt == "integral"
            result_real = (remaining_var::Complex{Float64}) -> 2.0*(2.0*pi)^(-1.0)*quadgk(FunctWrapper1D(real_funct,remaining_var), BoundArr[1], BoundArr[2]; rtol=1.5e-4, atol=1.5e-4, maxevals=100_000)[1]
            result_imag = (remaining_var::Complex{Float64}) -> 2.0*(2.0*pi)^(-1.0)*quadgk(FunctWrapper1D(imag_funct,remaining_var), BoundArr[1], BoundArr[2]; rtol=1.5e-4, atol=1.5e-4, maxevals=100_000)[1]
        elseif opt == "sum"
            result_real = (remaining_var::Complex{Float64}) -> 2.0*(Gridk)^(-1.0)*sum(FunctWrapper1D(real_funct,remaining_var), range(BoundArr[1], stop=BoundArr[2], length=Gridk))
            result_imag = (remaining_var::Complex{Float64}) -> 2.0*(Gridk)^(-1.0)*sum(FunctWrapper1D(imag_funct,remaining_var), range(BoundArr[1], stop=BoundArr[2], length=Gridk))
        end
    elseif isa(BoundArr,Array{Array{Float64,1},1})
        if opt == "integral"
            result_real = (remaining_var::Complex{Float64}) -> 2.0*(2.0*pi)^(-2.0)*hcubature(FunctWrapper2D(real_funct,remaining_var), BoundArr[1], BoundArr[2]; reltol=1.5e-4, abstol=1.5e-4, maxevals=100_000)[1]
            result_imag = (remaining_var::Complex{Float64}) -> 2.0*(2.0*pi)^(-2.0)*hcubature(FunctWrapper2D(imag_funct,remaining_var), BoundArr[1], BoundArr[2]; reltol=1.5e-4, abstol=1.5e-4, maxevals=100_000)[1]
        elseif opt == "sum"
            function result_real(remaining_var::Complex{Float64})
                tmpMat = Matrix{Complex{Float64}}(undef,(2,2))
                for ky in range(BoundArr[1][1], stop=BoundArr[2][1], length=Gridk)
                    for kx in range(BoundArr[1][2], stop=BoundArr[2][2], length=Gridk)
                        #println("real_funct: ", real_funct([ky,kx],remaining_var))
                        tmpMat .+= real_funct([ky,kx],remaining_var)
                    end
                end
                return 2.0*(Gridk)^(-2.0)*tmpMat
            end
            function result_imag(remaining_var::Complex{Float64})
                tmpMat = Matrix{Complex{Float64}}(undef,(2,2)) 
                for ky in range(BoundArr[1][1], stop=BoundArr[2][1], length=Gridk)
                    for kx in range(BoundArr[1][2], stop=BoundArr[2][2], length=Gridk)
                        #println("imag_funct: ", imag_funct([ky,kx],remaining_var))
                        tmpMat .+= imag_funct([ky,kx],remaining_var)
                    end
                end
                return 2.0*(Gridk)^(-2.0)*tmpMat
            end
        end
    end

    temp_func = (KK::Complex{Float64}) -> (result_real(KK) + 1.0im*result_imag(KK))
    
    tmp_self = Matrix{Complex{Float64}}(undef,(2,2))
    
    for iωn in structModel.matsubara_grid_
        tmp_self += real.(temp_func(iωn))
    end
    
    tmp_self = swap(tmp_self) ## Have to swap to have Σ_up = U*n_down

    return (2.0/structModel.beta_)*tmp_self + 1/2*II 
end

end ## End module Hubbard