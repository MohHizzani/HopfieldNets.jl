include("../src/includes.jl")
include("../demo/qapBM.jl")

hnqap1 = HopfieldQAP(D[1], F[1])

hnqap1.s = [0 0 1; 0 1 0; 1 0 0]
hnqap1.s = [1 0 0; 0 1 0; 0 0 1]

energy(hnqap1)/91.4
energy(hnqap1)
println(hnqap1.s[1,1])
update!(hnqap1, (1,1))
println(hnqap1.s[1,1])


# hnqap1.s = [1 -1 -1; -1 1 -1; -1 -1 1]
for i in 1:1000
    e = energy(hnqap1)
    println("Energy now ", e)
    println("Before update cycle")
    println(hnqap1.s)
    cycle!(hnqap1)
    println("After update cycle")
    println(hnqap1.s)
end
energy(hnqap1)

