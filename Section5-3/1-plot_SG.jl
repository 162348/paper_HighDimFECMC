using PDMPFlux, Plots, Statistics, StatsPlots, JLD2, CategoricalArrays, Random
using Bootstrap, DataFrames, Distributions, LaTeXStrings
default(palette = palette(:okabe_ito)[[6, 2]])
pal = palette(:okabe_ito)[[6, 2]]

d_list = [10, 20, 40, 80, 160, 320]
BMs_h, BMs_g = load(joinpath(@__DIR__, "Data", "1-StandardGauss.jld2"), "BMs_h", "BMs_g")
data = [BMs_h 2 ./ BMs_g]

x = 1:length(d_list)
iter  = size(data, 1)
n_dim = length(d_list)
grps = repeat(d_list, inner=iter, outer=2)
grps = categorical(grps; ordered=true, levels=d_list)
groups = vcat(fill("slow", iter*n_dim), fill("fast", iter*n_dim))

p_main = groupedboxplot(grps, vec(data);
    group = groups,
    legend = (0.62, 0.92),
    legend_column = -1,
    legendfontsize = 12,
    ylabel = "Estimate",
    title = "Standard Gaussian Target",
    linewidth = 1.0,
    outliers = false,
)
δ = 0.33
hline!(p_main, [8/sqrt(32/π)]; color=:black, lw=2, label=:none)
display(p_main)
plot!(p_main; xticks=:none, bottom_margin=-3Plots.mm)

p_mse = plot(
    xlabel = "Dimension",
    ylabel = "MSE",
    xticks = (x .- δ, string.(d_list)),
    framestyle = :box,
    legend = :none,
    ylims = (0.0, 0.12),
    yticks = 0.0:0.1:0.2,
)

MSE_h = vec(mean((BMs_h .- 8/sqrt(32/π)).^2, dims=1))
MSE_g = vec(mean((2 ./ BMs_g .- 8/sqrt(32/π)).^2, dims=1))
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