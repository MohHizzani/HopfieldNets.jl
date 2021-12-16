include("../src/includes.jl")


F = rand(2,2)
D = rand(3,3)

qaph = HopfieldQAP(D, F)


# using Main.HopfieldNets
using Test
using LinearAlgebra
# using Pkg
# include(Pkg.dir("Main.HopfieldNets", "demo", "letters.jl"))
include("../demo/letters.jl")

patterns = hcat(X, O)

n = size(patterns, 1)

dnet = DiscreteHopfieldNet(n)
cnet = ContinuousHopfieldNet(n)

threex3 = DiscreteHopfieldNet((3,3))
twox3 = DiscreteHopfieldNet((2,3))

# N = (2,3)

# a = Vector{Int}()

# for i in N
#      push!(a, repeat([i], length(N))...)
# end
# a

cycle!(twox3)



print(threex3.s[1,1])
update!(threex3, (1,1))
print(threex3.s[1,1])

# N = size(threex3.s)
# N_2 = size(threex3.W)
# randInd = [shuffle(1:i) for i in N]

# D = rand(Float64, 2, 2)
# F = rand(Float64, 3, 3)
# A = B = 2

# # D & F are square matrices always
# N = (size(D)[1],size(F)[1]) #dimensions of the Hopfield Network
# tnet = DiscreteHopfieldNet(N)

# iters = Base.Iterators.product([collect(1:i) for i in N]...)
# for i in iters
#     for j in iters
#           tnet.W[i...,j...] = -2()


# N = size(twox3.s)
# N_2 = size(twox3.W)
# randInd = [shuffle(1:i) for i in N]



for net in [dnet, cnet]
    train!(net, patterns)

    e0 = energy(net)
    settle!(net, 1_000, false)
    eFinal = energy(net)
    @assert e0 != eFinal

    Xcorrupt = copy(X)
    for i = 2:7
         Xcorrupt[i] = 1
    end
    Xrestored = associate!(net, Xcorrupt)
    @test norm(Xcorrupt - Xrestored) > 1e-4
    @test norm(X - Xrestored) < 1e-4

    Ocorrupt = copy(O)
    for i = 2:7
         Ocorrupt[i] = -1
    end
    Orestored = associate!(net, Ocorrupt)
    @test norm(Ocorrupt - Orestored) > 1e-4
    @test norm(O - Orestored) < 1e-4
end
