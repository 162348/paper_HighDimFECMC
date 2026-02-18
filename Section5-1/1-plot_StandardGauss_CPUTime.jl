using PDMPFlux, Plots, Statistics, StatsPlots, JLD2, CategoricalArrays, Random
using Bootstrap, DataFrames, Distributions
default(palette = palette(:okabe_ito)[[2, 6]])

d_list = [10, 20, 40, 80, 160, 320]
dic = load(joinpath(@__DIR__, "Data", "1-StandardGauss.jld2"))
SE_FECMC = dic["SE_FECMC"]
SE_BPS = dic["SE_BPS"]
elpt_FECMC = dic["elpt_FECMC"]
elpt_BPS = dic["elpt_BPS"]

SE_FECMC_per_time = SE_FECMC .* elpt_FECMC
SE_BPS_per_time = SE_BPS .* elpt_BPS

df = DataFrame(dim=Int[], method=String[], mean=Float64[], lo=Float64[], hi=Float64[])
for (j,d) in enumerate(d_list)
    bs = bootstrap(mean, SE_BPS_per_time[:,j], BasicSampling(5000))
    bci = confint(bs, BCaConfInt(0.95))
    push!(df, (d,"BPS (optimally tuned)",  2/bci[1][1], 2/bci[1][2], 2/bci[1][3]))
    bs = bootstrap(mean, SE_FECMC_per_time[:,j], BasicSampling(5000))
    bci = confint(bs, BCaConfInt(0.95))
    push!(df, (d,"FECMC",2/bci[1][1], 2/bci[1][2], 2/bci[1][3]))
end
df.dim = categorical(df.dim; ordered=true, levels=d_list)

ratios = df.mean[2:2:end] ./ df.mean[1:2:end-1]  # (FECMC mean) / (BPS mean)
xticklabels = ["$(d_list[i])\n($(round(ratios[i], digits=2)))" for i in eachindex(d_list)]


p = plot(
    xlabel="Dimension\n(FECMC/BPS ratio)",
    ylabel="ESS / Time (mean ± 95% CI)",
    xticks=(1:length(d_list), xticklabels),
    legend=:topright,
    title="Standard Gaussian Target",
    bottom_margin=2Plots.mm,
    yscale = :log10,
)

methods = ["FECMC", "BPS (optimally tuned)"] 
pal = Dict("BPS (optimally tuned)" => "#0096FF", "FECMC" => "#E95420")
markershape = Dict("BPS (optimally tuned)" => :circle, "FECMC" => :star5)
markersize = Dict("BPS (optimally tuned)" => 5, "FECMC" => 7)

xpos = Dict(d => i for (i,d) in enumerate(d_list))
offset = Dict("FECMC" => +0.1, "BPS (optimally tuned)" => -0.1)

for meth in methods
    sub = df[df.method .== meth, :]

    xs = [xpos[d] + offset[meth] for d in sub.dim]
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

display(p)