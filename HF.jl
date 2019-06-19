module Green_Utils

using Cubature: hcubature, hquadrature
using QuadGK

# Defining several global variables
mutable struct HFVec # <--------------------------------------------------------------- Order of declaration IS IMPORTANT 
    Niωn_::Int64 ## Discrtization of the Matsubara_Grid
    beta_::Int64 ## Temperature for the summation
    N_it_::Int64 ## Number of iterations
    dict_v_::Dict{String,Float64} ## Dict containing U and V
    Matsubara_G_::Vector{Complex{Float64}} ## Matsubara grid

    function HFVec(Niωn_::Int64, beta_::Int64, N_it_::Int64, dict_v_::Dict{String,Float64})
        Matsubara_G_ = Matsubara_Grid(beta_,Niωn_)
    
        return new(Niωn_, beta_, N_it_, dict_v_, Matsubara_G_)
    end
end

function Matsubara_Grid(beta_::Int64, Niωn_::Int64)
    Matsubara_G_ = Vector{Complex{Float64}}(undef,Niωn_)
    Niωn_ = mod(Niωn_,2) == 0 ? Niωn_ : Niωn_ + 1
    ## The information in the first Matsubara frequencies are more important
    Matsubara_G_[1:end] = 1.0im*[(2*n + 1)*pi/(beta_) for n in 1:Niωn_]

    return Matsubara_G_
end

struct Integral
    kx::Float64
    ky::Float64
    iωn::Complex{Float64}
end

struct Integral1D
    kx::Float64
    iωn::Complex{Float64}
end

# Defining some functions
function Potential(qx::Float64, qy::Float64, U::Float64, V::Float64)
    UV = U + V*(cos(qx) + cos(qy))

    return UV
end

function epsilonk(kx::Float64, ky::Float64; t::Float64=1.0, tp::Float64=-0.3, tpp::Float64=0.2)
    epsilonk1 = -2.0*t*(cos(kx) + cos(ky))
    epsilonk2 = -2.0*tp*(cos(kx+ky)+cos(kx-ky))
    epsilonk3 = -2.0*tpp*(cos(2.0*kx)+cos(2.0*ky))

    return epsilonk1
end

function Potential1D(qx::Float64, U::Float64, V::Float64)
    UV = U + V*(cos(qx))

    return UV
end

function epsilonk1D(kx::Float64; t::Float64=1.0)
    epsilonk1 = -2.0*t*(cos(kx))
    
    return epsilonk1
end

function Gk(qx::Float64, qy::Float64, kx::Float64, ky::Float64, iωn::Complex{Float64}, SE_func::Function)
    integ = Integral(kx + qx, ky + qy, iωn)
    gk = 1.0/(iωn - epsilonk(kx + qx, ky + qy) - SE_func(integ))
    
    return gk

end

function Init_Gk(qx::Float64, qy::Float64, kx::Float64, ky::Float64, iωn::Complex{Float64})
    gk_init = 1.0/(iωn - epsilonk(kx + qx, ky + qy) - (0.1-0.1im))

    return gk_init
end

function Gk1D(qx::Float64, kx::Float64, iωn::Complex{Float64}, SE_func::Function)
    integ = Integral1D(kx + qx, iωn)
    gk = 1.0/(iωn - epsilonk1D(kx + qx) - SE_func(integ))
    
    return gk

end

function Init_Gk1D(qx::Float64, kx::Float64, iωn::Complex{Float64})
    gk_init = 1.0/(iωn - epsilonk1D(kx + qx) - (0.1-0.1im))

    return gk_init
end

function Funct_Wrapper_one(funct::Function)
    function Inner_funct(k::Array{Float64,1})
        return funct(k[1], k[2])
    end

    return Inner_funct
end

function Funct_Wrapper_two(funct::Function, other::Integral)
    function Inner_funct(k::Array{Float64,1})
        return funct(k[1], k[2], other)
    end

    return Inner_funct
end

function Funct_Wrapper_two1D(funct::Function, other::Integral1D)
    function Inner_funct(k::Float64)
        return funct(k, other)
    end

    return Inner_funct
end


function integrate_complex(funct::Function, SE_funct::Function, ii::Int64, st::HFVec, BoundArr::Array{Array{Float64,1},1})
    dict = st.dict_v_
    if ii <= 1
        println(ii, " integrate_complex")
        dressed_funct = (qx::Float64, qy::Float64, kx::Float64, ky::Float64, iωn::Complex{Float64})->1.0*Potential(qx, qy, values(dict)...)*funct(qx, qy, kx ,ky, iωn)
    elseif ii > 1
        println(ii, " integrate_complex")
        dressed_funct = (qx::Float64, qy::Float64, kx::Float64, ky::Float64, iωn::Complex{Float64})->1.0*Potential(qx, qy, values(dict)...)*funct(qx, qy, kx, ky, iωn, SE_funct)
    end


    function real_funct(qx::Float64, qy::Float64, rest::Integral)
        return real(dressed_funct(qx, qy, rest.kx, rest.ky, rest.iωn))
    end

    function imag_funct(qx::Float64, qy::Float64, rest::Integral)
        return imag(dressed_funct(qx, qy, rest.kx, rest.ky, rest.iωn))
    end

    
    result_real = (remaining_var::Integral) -> hcubature(Funct_Wrapper_two(real_funct,remaining_var), BoundArr[1], BoundArr[2]; reltol=1.5e-2, abstol=1.5e-2, maxevals=100_000)[1]
    result_imag = (remaining_var::Integral) -> hcubature(Funct_Wrapper_two(imag_funct,remaining_var), BoundArr[1], BoundArr[2]; reltol=1.5e-2, abstol=1.5e-2, maxevals=100_000)[1]
    

    temp_func = (KK::Integral) -> 2.0*(2.0*pi)^(-2.0)*(result_real(KK) + 1.0im*result_imag(KK))

    return temp_func
end

function interation_process(st::HFVec, BoundArr::Array{Array{Float64,1},1}, SubLast::Int64)
    N_it = st.N_it_
    SE_funct = nothing
    Dict_array = Dict{Int64,Array{Function,1}}()
    for it in 1:N_it    
        Funct_array = Array{Function,1}([])
        if it <= 1
            for sub in 1:SubLast
                println(it, " interation_process")
                #SE_funct = integrate_complex(Init_Gk, x -> x, it, st, BoundArr) ## SE_funct takes in Integral-type values
                #push!(Funct_array,SE_funct)
                to_add_lower = (convert(Float64,sub)-1.)*2.0*pi/SubLast
                to_add_upper = (convert(Float64,sub))*2.0*pi/SubLast
                BoundArr = Array{Array{Float64,1},1}([[-pi+to_add_lower,-pi+to_add_lower],[-pi+to_add_upper,-pi+to_add_upper]])
                println("it: ", it, "BoundArr: ", BoundArr)
                push!(Funct_array, integrate_complex(Init_Gk, x -> x, it, st, BoundArr))
            end
        elseif it > 1
            for sub in 1:SubLast
                println(it, " interation_process")
                #SE_temp = Array{Function,1}(undef,SubLast)
                to_add_lower = (convert(Float64,sub)-1.)*2.0*pi/SubLast
                to_add_upper = (convert(Float64,sub))*2.0*pi/SubLast
                BoundArr = Array{Array{Float64,1},1}([[-pi+to_add_lower,-pi+to_add_lower],[-pi+to_add_upper,-pi+to_add_upper]])
                println("BoundArr: ", BoundArr)
                push!(Funct_array, integrate_complex(Gk, Dict_array[it-1][sub], it, st, BoundArr))
            end
        end
        Dict_array[it] = Funct_array
    end

    return Dict_array
end

# function funct_wrapper_vector(funct::Function, integ::Integral1D, vec::Vector{Float64})
#     function wrapped_function(k::Float64,vec)
#         return(funct(k[1], integ, vec))
#     end
#     return wrapped_function
# end

function integrate_complex1D(funct::Function, SE_funct::Function, ii::Int64, st::HFVec, BoundArr::Array{Float64,1}, Gridk::Int64; opt::String="sum")
    dict = st.dict_v_
    if ii <= 1
        println(ii, " integrate_complex 1D")
        dressed_funct = (qx::Float64, kx::Float64, iωn::Complex{Float64})->1.0*Potential1D(qx,values(dict)...)*funct(qx, kx, iωn)
    elseif ii > 1
        println(ii, " integrate_complex 1D")
        dressed_funct = (qx::Float64, kx::Float64, iωn::Complex{Float64})->1.0*Potential1D(qx, values(dict)...)*funct(qx, kx, iωn, SE_funct)
    end


    function real_funct(qx::Float64, rest::Integral1D)
        return real(dressed_funct(qx, rest.kx, rest.iωn))
    end

    function imag_funct(qx::Float64, rest::Integral1D)
        return imag(dressed_funct(qx, rest.kx, rest.iωn))
    end

    # vec = Vector{Float64}(undef,2)
    # function vectorized_version(qx::Float64, rest::Integral1D, vec::Vector{Float64})
    #     length(vec) != 2 && throw(ErrorException("Size must be two!!"))
    #     realf = real(dressed_funct(qx, rest.kx, rest.iωn))
    #     imagf = imag(dressed_funct(qx, rest.kx, rest.iωn))
    #     vec[1] = realf; vec[2] = imagf

    #     return nothing
    # end

    # vectorized_funct = (KK::Integral1D) -> hquadrature(2, funct_wrapper_vector(vectorized_version,KK,vec), BoundArr[1], BoundArr[2]; reltol=1.5e-8, abstol=1.5e-8, maxevals=100_000)[1]
    
    # temp_func = (KK::Integral1D) -> 2.0*(2.0*pi)^(-1.0)*(vectorized_funct(KK)[1]+1.0im*vectorized_funct(KK)[2])

    if opt == "integral"
        result_real = (remaining_var::Integral1D) -> 2.0*(2.0*pi)^(-1.0)*quadgk(Funct_Wrapper_two1D(real_funct,remaining_var), BoundArr[1], BoundArr[2]; rtol=1.5e-4, atol=1.5e-4, maxevals=100_000)[1]
        result_imag = (remaining_var::Integral1D) -> 2.0*(2.0*pi)^(-1.0)*quadgk(Funct_Wrapper_two1D(imag_funct,remaining_var), BoundArr[1], BoundArr[2]; rtol=1.5e-4, atol=1.5e-4, maxevals=100_000)[1]
    elseif opt == "sum"
        result_real = (remaining_var::Integral1D) -> 2.0*(Gridk)^(-1.0)*sum(Funct_Wrapper_two1D(real_funct,remaining_var), range(BoundArr[1], stop=BoundArr[2], length=Gridk))
        result_imag = (remaining_var::Integral1D) -> 2.0*(Gridk)^(-1.0)*sum(Funct_Wrapper_two1D(imag_funct,remaining_var), range(BoundArr[1], stop=BoundArr[2], length=Gridk))
    else
        throw(ErrorException("Oups! opt parameter takes as input only: \"sum\" or \"integral\"."))
    end

    temp_func = (KK::Integral1D) -> 1.0*(result_real(KK) + 1.0im*result_imag(KK))

    return temp_func
end

function interation_process1D(st::HFVec, BoundArr::Array{Float64,1}, SubLast::Int64, Gridk::Int64)
    N_it = st.N_it_
    SE_funct = nothing
    Dict_array = Dict{Int64,Array{Function,1}}()
    for it in 1:N_it    
        Funct_array = Array{Function,1}([])
        if it <= 1
            for sub in 1:SubLast
                println(it, " interation_process 1D")
                to_add_lower = (convert(Float64,sub)-1.)*2.0*pi/SubLast
                to_add_upper = (convert(Float64,sub))*2.0*pi/SubLast
                BoundArr = Array{Float64,1}([-pi+to_add_lower,-pi+to_add_upper])
                println("it: ", it, "BoundArr1D: ", BoundArr)
                push!(Funct_array, integrate_complex1D(Init_Gk1D, x -> x, it, st, BoundArr, Gridk))
            end
        elseif it > 1
            for sub in 1:SubLast
                println(it, " interation_process 1D")
                to_add_lower = (convert(Float64,sub)-1.)*2.0*pi/SubLast
                to_add_upper = (convert(Float64,sub))*2.0*pi/SubLast
                BoundArr = Array{Float64,1}([-pi+to_add_lower,-pi+to_add_upper])
                println("BoundArr1D: ", BoundArr)
                push!(Funct_array, integrate_complex1D(Gk1D, Dict_array[it-1][sub], it, st, BoundArr, Gridk))
            end
        end
        Dict_array[it] = Funct_array
    end

    return Dict_array
end


end ## End of module 
