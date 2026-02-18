using PDMPFlux, Plots, Statistics, StatsPlots, JLD2, CategoricalArrays, Random
using Bootstrap, DataFrames, Distributions, LaTeXStrings

const d = 100

ν_list = [10, 50, 100, 500, 1000, 10000]
SE_FECMC, SE_BPS, t_FECMC, t_BPS = load(joinpath(@__DIR__, "Data", "2-Student.jld2"), "SE_FECMC", "SE_BPS", "t_FECMC", "t_BPS")

df = DataFrame(ν=Float64[], method=String[], mean=Float64[], lo=Float64[], hi=Float64[], γ=Float64[])
for (j,d) in enumerate(ν_list)
    bs = bootstrap(mean, SE_BPS[:,j], BasicSampling(5000))
    bci = confint(bs, BCaConfInt(0.95))
    push!(df, (d,"BPS (optimally tuned)", 1/bci[1][1], 1/bci[1][2], 1/bci[1][3], d^(-1)))
    bs = bootstrap(mean, SE_FECMC[:,j], BasicSampling(5000))
    bci = confint(bs, BCaConfInt(0.95))
    push!(df, (d,"FECMC",1/bci[1][1], 1/bci[1][2], 1/bci[1][3], d^(-1)))
end
df.ν = categorical(df.ν; ordered=true, levels=ν_list)
γ_list = ν_list .^ (-1)
df.γ = categorical(df.γ; ordered=true, levels=γ_list)

ν_list = ν_list[Not([2,4])]
df = df[Not([3,4,7,8]),:]

ratios = vec(mean(SE_BPS, dims=1)) ./ vec(mean(SE_FECMC, dims=1))  # (FECMC mean) / (BPS mean)
xticklabels = ["$(γ_list[i])\n($(round(ratios[i], digits=2)))" for i in eachindex(reverse(γ_list))]

p = plot(
    xlabel=L"Tail Heaviness $\nu^{-1}$",
    ylabel="ESS (mean ± 95% CI)",
    xticks=(1:length(γ_list), string.(reverse(γ_list))),
    legend=(0.72, 0.77),
    title="Student Target",
    framestyle=:box,
    palette = palette(:okabe_ito)[[2, 6]]
)

methods = ["FECMC", "BPS (optimally tuned)"] 
pal = Dict("BPS (optimally tuned)" => "#0096FF", "FECMC" => "#E95420")
markershape = Dict("BPS (optimally tuned)" => :circle, "FECMC" => :star5)
markersize = Dict("BPS (optimally tuned)" => 5, "FECMC" => 7)

xpos = Dict(d => i for (i,d) in enumerate(reverse(γ_list)))
offset = Dict("FECMC" => +0.1, "BPS (optimally tuned)" => -0.1)

for meth in methods
    sub = df[df.method .== meth, :]

    xs = [xpos[d] + offset[meth] for d in sub.γ]
    ys = sub.mean
    yerr_low  = ys .- sub.lo
    yerr_high = sub.hi .- ys

    scatter!(p, xs, ys;
        yerror=(yerr_low, yerr_high),
        color=pal[meth],
        markerstrokecolor=pal[meth],
        markershape=markershape[meth],
        markersize=markersize[meth],
        label=meth
    )

    plot!(p, xs, ys; color=pal[meth], label=false)
end

hline!(p, [39.894]; color=:black, lw=1, label=:none)
hline!(p, [22.979]; color=:black, style=:dash, lw=1, label=:none)

display(p)
