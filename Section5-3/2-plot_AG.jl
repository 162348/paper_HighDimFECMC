using PDMPFlux, Plots, Statistics, StatsPlots, JLD2, CategoricalArrays, Random
using Bootstrap, DataFrames, Distributions, LaTeXStrings
default(palette = palette(:okabe_ito)[[6, 2]])
pal = palette(:okabe_ito)[[6, 2]]

d_list = [10, 20, 40, 80, 160, 320]
BMs_h, BMs_g = load(joinpath(@__DIR__, "Data", "2-AnisoGauss.jld2"), "BMs_h", "BMs_g")
data = [8 ./ BMs_h 4 .* BMs_g]

x = 1:length(d_list)
iter  = size(data, 1)
n_dim = length(d_list)
grps = repeat(d_list, inner=iter, outer=2)
grps = categorical(grps; ordered=true, levels=d_list)
groups = vcat(fill("slow", iter*n_dim), fill("fast", iter*n_dim))

df = DataFrame(d=Float64[], method=String[], mean=Float64[], lo=Float64[], hi=Float64[])
for (j,d) in enumerate(d_list)
    bs = bootstrap(mean, BMs_h[:,j], BasicSampling(5000))
    bci = confint(bs, BCaConfInt(0.95))
    push!(df, (d,"BM for h",  8/bci[1][1], 8/bci[1][2], 8/bci[1][3]))
    bs = bootstrap(mean, BMs_g[:,j], BasicSampling(5000))
    bci = confint(bs, BCaConfInt(0.95))
    push!(df, (d,"BM for g",4*bci[1][1], 4*bci[1][2], 4*bci[1][3]))
end
df.d = categorical(df.d; ordered=true, levels=d_list)

p_main = groupedboxplot(grps, vec(data);
    group = groups,
    legend = :bottomright,
    legend_column = -1,
    legendfontsize = 12,
    ylabel = "Estimate",
    title = "Anisotropic Gaussian Target",
    linewidth = 1.0,
    outliers = false,
)
δ = 0.33
hline!(p_main, [sqrt(32/π)]; color=:black, lw=2, label=:none)
display(p_main)
plot!(p_main; xticks=:none, bottom_margin=-3Plots.mm)

p_mse = plot(
    xlabel = "Dimension",
    ylabel = "MSE",
    xticks = (x .- δ, string.(d_list)),
    framestyle = :box,
    legend = :none,
    ylims = (0.0, 0.9),
    yticks = 0.0:0.3:0.9,
)

SE_FECMC, SE_BPS, t_FECMC, t_BPS = load(
  abspath(joinpath(@__DIR__, "..", "Section5-1", "Data", "3-AnisoGauss.jld2")),
  "SE_FECMC", "SE_BPS", "t_FECMC", "t_BPS",
)
estimated_MSE = mean(SE_FECMC[:,6]) * 100 / 2
estimated_sigma2 = 8/estimated_MSE
MSE_h = vec(mean((8 ./ BMs_h .- estimated_sigma2).^2, dims=1))
MSE_g = vec(mean((4 .* BMs_g .- estimated_sigma2).^2, dims=1))
hline!(p_main, [estimated_sigma2]; color=:black, lw=2, label=:none)

scatter!(p_mse, x .- δ,  MSE_h; label=L"8/\widehat{σ_h^2}", color=pal[2], markersize = 5)
scatter!(p_mse, x .- δ, MSE_g; label=L"4\widehat{σ_g^2}",
    marker=:star,
    color=pal[1], markersize=7
)
plot!(p_mse, x .- δ,  MSE_h; lw=1, label=false, color=pal[2])
plot!(p_mse, x .- δ, MSE_g; lw=1, label=false, color=pal[1])

p_all = plot(
    p_main, p_mse;
    layout = grid(2, 1, heights=[0.82, 0.18]),
    link = :x,
)
display(p_all)