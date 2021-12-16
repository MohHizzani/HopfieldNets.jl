mutable struct HopfieldQAP <: HopfieldNet
    s::Array{Float64, N} where {N}
    # s::BitArray{N} where {N}
    W::Array{Float64, N} where {N}
    # A::Float64
    # B::Float64
    # q::Float64
end

function δ(ind...)
    all(y -> y==first(ind), ind)
end
function HopfieldQAP(D::Array{Float64, Dim}, F::Array{Float64, Dim}
    # , A=0.9, B=0.9, q=140
    ) where {Dim}
    # A = B = 2

    # D & F are square matrices always
    N = (size(D)[1],size(F)[1]) #dimensions of the Hopfield Network
    
    tnet = DiscreteHopfieldNet(N)

    iters = Base.Iterators.product([collect(1:i) for i in N]...)
    for i in iters
        for j in iters
            tnet.W[i..., j...] = -2(A * (1-δ(i[1],j[1]) * δ(i[2],j[2])) + B * δ(i[1], j[1]) * (1 - δ(i[2], j[2])) + (D[i[1], j[1]] * F[i[2], j[2]])/q)
            # tnet.W[i..., j...] = D[i[1], j[1]] * F[i[2], j[2]]
        end
    end

    return HopfieldQAP(tnet.s , tnet.W
                        , A, B, q
                        )
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
        end
        # e += (dot(net.W[a...], net.s) + net.A + net.B) *net.s[ind...]
        e += dot(net.W[a...], net.s) * net.s[ind...]
    end
    return e

end


function update!(net::HopfieldQAP, ind::Tuple)
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
    dotp = dot(net.W[a...], net.s) * net.s[ind...] - net.A - net.B
    net.s[ind...] = (1/ (1 + exp(dotp)) )#> 0.5
    # net.s[ind...] = dot(net.W[a...], net.s) > 0.0 ? +1 : -1

    return nothing
end

# Perform an update cycle on the whole network
function cycle!(net::HopfieldQAP)
    N = size(net.s)
    # N_2 = size(net.W)
    randInd = [shuffle(1:i) for i in N]
    for i in Base.Iterators.product(randInd...)
        update!(net, i)
    end
    return nothing    
end