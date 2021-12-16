using LinearAlgebra
using Random

mutable struct DiscreteHopfieldNet <: HopfieldNet
    s::BitArray{N} where {N} # State -- could have used Int's
    W::Array{Float64, N} where {N} # Weights
end

function DiscreteHopfieldNet(n::Integer)
    s = BitArray(rand(Bool, n))
    W = rand(n, n)
    DiscreteHopfieldNet(s, W)
end

function DiscreteHopfieldNet(N::Tuple)
    s = BitArray(rand(Bool, N))
    a = Vector{Int}()
    # for i in N
    #     push!(a, repeat([i], length(N))...)
    # end
    
    W = rand(Float64, repeat([N...],2)...)
    DiscreteHopfieldNet(s, W)
end

# Perform update on a certain neuron
# function update!(net::HN, ind::Tuple, θ::Float64 = 0.0) where {HN <: HopfieldNet}
#     N = size(net.s)
#     N_2 = size(net.W)
#     a = []
#     for i in 1:length(N_2)
#         if i <= length(N)
#             push!(a, ind[i])
#         else
#             push!(a, :)
#         end
#     end
#     dotp = dot(net.W[a...], net.s)
#     net.s[ind...] = dotp > θ ? +1 : -1
#     # net.s[ind...] = dot(net.W[a...], net.s) > 0.5 ? 1 : 0

#     return nothing
# end

# # Perform an update cycle on the whole network
# function cycle!(net::HN, θ::Float64 = 50.0) where {HN <: HopfieldNet}
#     N = size(net.s)
#     # N_2 = size(net.W)
#     randInd = [shuffle(1:i) for i in N]
#     for i in Base.Iterators.product(randInd...)
#         update!(net, i, θ)
#     end
#     return nothing    
# end

function Base.show(io::IO, net::DiscreteHopfieldNet)
    # @printf io "A discrete Hopfield net with %d neurons\n" length(net.s)
    print(io, "A discrete Hopfield net with $(size(net.s)) neurons")
end




#=
# Perform one asynchronous update on randomly selected neuron
function update!(net::DiscreteHopfieldNet)
    i = rand(1:length(net.s))
    net.s[i] = dot(net.W[:, i], net.s) > 0 ? +1 : -1
    return nothing
end
=#