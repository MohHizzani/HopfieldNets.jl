module HopfieldNets
    export HopfieldNet, DiscreteHopfieldNet, ContinuousHopfieldNet
    export update!, energy, settle!, train!, associate!, storkeytrain!

    include("includes.jl")
end
