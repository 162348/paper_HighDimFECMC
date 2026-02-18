using Plots, LaTeXStrings, SpecialFunctions
default(palette = palette(:okabe_ito)[[2, 6]])

function Ω(ρ)
  return sqrt(π/2) * ρ * erfcx(ρ/sqrt(2))
end

function σ_FECMC(ρ)
  factor = (ρ^2 - ρ * sqrt(π/2) + Ω(ρ))^2 / (ρ^4 * Ω(ρ) * (2-Ω(ρ)))
  return (1/sqrt(2π)) - (1/sqrt(2π)) * factor
end

function σ_BPS(ρ)
  if abs(ρ) < 1e-8
    return 0.0
  end
  
  Ω_ρ = Ω(ρ)
  Ω_2ρ = Ω(2ρ)
  
  if abs(Ω_2ρ) < 1e-12
    return 0.0
  end
  
  block = (Ω_ρ * (1+ρ^2) - ρ^2)^2 / Ω_2ρ
  factor = ρ^3 - ρ^2 * sqrt(8/π) + ρ - sqrt(8/π) * block
  
  result = factor / ρ^4
  
  if abs(result) < 1e-10
    return 0.0
  end
  
  return result
end

cols = palette(:okabe_ito)[[2, 6]]
c_bps, c_fecmc = cols[1], cols[2]

p = plot(ρ -> 8 * σ_BPS(ρ),
    range(0.0, 10.0, length=1000),
    label="BPS", primary=false, color="#0096FF",
    xlabel=L"\rho", ylabel=L"\sigma^2(\rho)",
    linewidth=3, left_margin=2Plots.mm,
    guidefontsize=18, legendfontsize=12, tickfontsize=10,
    # background_color = "#F0F1EB",
)

plot!(p, ρ -> 8 * σ_FECMC(ρ),
    label="FECMC", primary=false, color="#E95420",
    linewidth=3,
)

plot!(p, [NaN], [NaN], label="FECMC", color=c_fecmc, linewidth=3)
plot!(p, [NaN], [NaN], label="BPS",   color=c_bps,   linewidth=3)

savefig("sigma/fig-1.svg")