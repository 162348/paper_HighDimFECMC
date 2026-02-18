using PDMPFlux, Plots, Statistics, StatsPlots, JLD2, CategoricalArrays, Random
using Bootstrap, DataFrames, Distributions

const d = 100

γ_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SE_FECMC, SE_BPS, t_FECMC, t_BPS = load(joinpath(@__DIR__, "Data", "1-AnisoGauss.jld2"), "SE_FECMC", "SE_BPS", "t_FECMC", "t_BPS")
ratios = vec(mean(SE_BPS, dims=1)) ./ vec(mean(SE_FECMC, dims=1))  # (FECMC mean) / (BPS mean)
xticklabels = ["$(γ_list[i])\n($(round(ratios[i], digits=2)))" for i in eachindex(γ_list)]

df = DataFrame(γ=Float64[], method=String[], mean=Float64[], lo=Float64[], hi=Float64[])
for (j,d) in enumerate(γ_list)
    bs = bootstrap(mean, SE_BPS[:,j], BasicSampling(5000))
    bci = confint(bs, BCaConfInt(0.95))
    push!(df, (d,"BPS (optimally tuned)",  2/bci[1][1], 2/bci[1][2], 2/bci[1][3]))
    bs = bootstrap(mean, SE_FECMC[:,j], BasicSampling(5000))
    bci = confint(bs, BCaConfInt(0.95))
    push!(df, (d,"FECMC",2/bci[1][1], 2/bci[1][2], 2/bci[1][3]))
end
df.γ = categorical(df.γ; ordered=true, levels=γ_list)

p = plot(
    xlabel="Correlation γ",
    ylabel="ESS (mean ± 95% CI)",
    xticks=(1:length(γ_list), string.(γ_list)),
    legend=:topleft,
    title="Anisotropic Gaussian Target",
    framestyle=:box,
    palette = palette(:okabe_ito)[[2, 6]]
)

methods = ["FECMC", "BPS (optimally tuned)"] 
pal = Dict("BPS (optimally tuned)" => "#0096FF", "FECMC" => "#E95420")
markershape = Dict("BPS (optimally tuned)" => :circle, "FECMC" => :star5)
markersize = Dict("BPS (optimally tuned)" => 5, "FECMC" => 7)

xpos = Dict(d => i for (i,d) in enumerate(γ_list))
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
