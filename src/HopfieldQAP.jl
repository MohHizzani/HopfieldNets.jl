mutable struct HopfieldQAP <: HopfieldNet
    # s::Array{Float64, N} where {N}
    s::BitArray{N} where {N}
    W::Array{Float64, N} where {N}
    λ::Float64
    # D::Array{Float64, N} where {N}
    # F::Array{Float64, N} where {N}
    # A::Float64
    # B::Float64
    # q::Float64
    ϵ::Float64
end

function δ(ind...)
    all(y -> y==first(ind), ind)
end

function HopfieldQAP(D::Array{Float64, Dim}, F::Array{Float64, Dim}
                    # , A=0.9, B=0.9, q=140
                    , ϵ=0.02
                    , λ = 1.1
    ) where {Dim}
    # A = B = 2

    # D & F are square matrices always
    N = (size(D)[1],size(F)[1]) #dimensions of the Hopfield Network
    
    tnet = DiscreteHopfieldNet(N)

    iters = Base.Iterators.product([collect(1:i) for i in N]...)
    for i in iters
        for j in iters
            # tnet.W[i..., j...] = -2(A * (1-δ(i[1],j[1]) * δ(i[2],j[2])) + B * δ(i[1], j[1]) * (1 - δ(i[2], j[2])) + (D[i[1], j[1]] * F[i[2], j[2]])/q)
            tnet.W[i..., j...] = D[i[1], j[1]] * F[i[2], j[2]]
        end
    end

    return HopfieldQAP(tnet.s , tnet.W
                        # , D, F
                        # , A, B, q
                        , ϵ
                        , λ
                        )
end

# This is a test function to return variable wieghts at each call
function weights(net::HopfieldQAP, i::Tuple)
    N = size(net.s)
    w = Array{Float64}(undef, N...)
    A = net.A
    B = net.B
    q = net.q
    D = net.D
    F = net.F
    iters = Base.Iterators.product([collect(1:i) for i in N]...)
    for j in iters
        w[j...] = -2(A * (1-δ(i[1],j[1]) * δ(i[2],j[2])) + B * δ(i[1], j[1]) * (1 - δ(i[2], j[2])) + (D[i[1], j[1]] * F[i[2], j[2]])/q)
    end
    
    return w
    
end

function energy(net::HopfieldQAP)
    N = size(net.s)
    N_2 = size(net.W)
    iters = Base.Iterators.product([collect(1:i) for i in N]...)
    e = 0.0
    for ind in iters
        a = []
        for i in 1:length(N_2)
            if i <= length(N)
                push!(a, ind[i])
            else
                push!(a, :)
            end
        endMMo
        # e += (dot(net.W[a...], net.s) + net.A + net.B) *net.s[ind...]
        e += dot(net.W[a...], net.s) * net.s[ind...] + net.λ * ((sum(net.s[:,ind[2]]) - 1)^2 + (sum(net.s[ind[1],:]) - 1)^2)
        # e += (dot(weights(net, ind), net.s) + net.A + net.B) * net.s[ind...]
    end
    return e

end


function update!(net::HopfieldQAP, ind::Tuple
                 , θ = 50
                 )
    N = size(net.s)
    N_2 = size(net.W)
    a = []
    for i in 1:length(N_2)
        if i <= length(N)
            push!(a, ind[i])
        else
            push!(a, :)
        end
    end
    # dotp = dot(net.W[a...], net.s) * net.s[ind...] - net.A - net.B
    # dotp = dot(weights(net, ind), net.s) * net.s[ind...] - net.A - net.B
    # net.s[ind...] = (1/ (1 + exp(dotp/net.ϵ)) )#> 0.5
    # net.s[ind...] = dot(net.W[a...], net.s) > θ #? +1 : -1
    fin = dot(net.W[a...], net.s) + net.λ * 2 * (sum(net.s[ind[1], : ]) + sum(net.s[:, ind[2]])) - 4 * net.λ
    net.s[ind...] = round(1 / (1 + exp(fin/net.ϵ)))
    return nothing
end

# Perform an update cycle on the whole network
function cycle!(net::HopfieldQAP
                , θ::Number = 50
                )
    N = size(net.s)
    # N_2 = size(net.W)
    randInd = [shuffle(1:i) for i in N]
    for i in Base.Iterators.product(randInd...)
        update!(net, i
                , θ
                )
    end
    return nothing    
end

function isFeasible(net::HopfieldQAP)
    N = size(net.s)
    for i in 1:N[1]
        if sum(net.s[i,:]) != 1
            return false
        end
    end
    for j in i:N[2]
        if sum(net.s[:,j]) != 1
            return false
        end
    end
    return true
end

function lowEFeasible(net::HopfieldQAP, cyc::Int)
    e = 0.0
    f = isFeasible(net)       
    
    for i in 1:cyc
        e = energy(net)
        cycle!(hnqap1)
        f = isFeasible(hnqap1)    
end