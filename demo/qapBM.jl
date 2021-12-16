#=
the online version of QAPLIB – A Quadratic Assignment Problem Library by R.E. Burkard, S.E. Karisch and F. Rendl, (Journal of Global Optimization 10:391-403, 1997.)
https://coral.ise.lehigh.edu/data-sets/qaplib/
=#

D = Vector{Matrix{Float64}}()
F = Vector{Matrix{Float64}}()
push!(D, [0 5 6; 5 0 3.6; 6 3.6 0])
push!(F, [0 10 3; 10 0 6.5; 3 6.5 0])