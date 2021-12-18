include("../src/includes.jl")
include("../demo/qapBM.jl")

hnqap1 = HopfieldQAP(D[1], F[1])

hnqap1.s = [0 0 1;
            0 1 0;
            1 0 0]
energy(hnqap1)
hnqap1.s = [1 0 0; 0 1 0; 0 0 1]
energy(hnqap1)
energy(hnqap1)/91.4

# println(hnqap1.s[1,1])
# update!(hnqap1, (1,1))
# println(hnqap1.s[1,1])


# hnqap1.s = [1 -1 -1; -1 1 -1; -1 -1 1]
# θ = energy(hnqap1) / 3
hnqap1.s = [1 0 0; 0 1 0; 0 0 1]
hnqap1.λ = 0.5
hnqap1.ϵ = 0.05
θ = 60
for i in 1:5
    e = energy(hnqap1)
    # println("Before update cycle")
    # println(hnqap1.s)
    cycle!(hnqap1, θ)
    f = isFeasible(hnqap1)
    println("The solution is $(map(x-> x ? " " : "not ", f))feasible")
    println("After update cycle")
    println(hnqap1.s)
    println("Energy now ", e)
end
energy(hnqap1)

