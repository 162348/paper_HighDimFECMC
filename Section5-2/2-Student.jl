using PDMPFlux, LinearAlgebra, Plots, Statistics, ProgressBars, StatsPlots, JLD2, Distributions, CategoricalArrays, SpecialFunctions

const IS_CI = get(ENV, "CI", "false") == "true" || get(ENV, "GITHUB_ACTIONS", "false") == "true"
const d = IS_CI ? 10 : 100
const OUTPUT_PATH = joinpath(@__DIR__, "Data", "2-Student.jld2")

function h_estimate_online(
  sampler::PDMPFlux.AbstractPDMP,
  ν::Float64,
  T_end::Float64,
  xinit::Vector{Float64},
  vinit::Vector{Float64};
  seed::Union{Int, Nothing}=nothing,
)::Float64
  if !(isfinite(T_end)) || T_end < 0
    throw(ArgumentError("T_end must be finite and non-negative. Current value: $T_end"))
  end
  d = length(xinit)
  if d == 0
    throw(ArgumentError("xinit must be non-empty"))
  end
  if T_end == 0.0
    return 0.0
  end

  state = PDMPFlux.init_state(sampler, xinit, vinit, seed)

  estimate = 0.0
  factor = (d+ν) / 2

  t_prev = state.t
  x_prev = copy(state.x)
  v_prev = copy(state.v)

  while t_prev < T_end
    PDMPFlux.get_event_state!(state, sampler)
    t_next = state.t

    if t_next <= T_end
      Δt = t_next - t_prev
      x_next = copy(state.x)
      b = 2 * (x_prev ⋅ v_prev)
      c = ν + sum(abs2, x_prev)
      sqrtΔ = sqrt(4 * c - b^2)
      Q = t -> t^2 + b * t + c
      F = t -> (t + b / 2) * log(Q(t)) - 2 * (t + b / 2) + sqrtΔ * atan(2 * t + b, sqrtΔ)
      estimate += factor * (F(Δt) - F(0) - Δt * log(ν))

      t_prev = t_next
      copyto!(x_prev, state.x)
      copyto!(v_prev, state.v)
    else
      Δt = T_end - t_prev
      x_next = x_prev .+ Δt .* v_prev

      b = 2 * (x_prev ⋅ v_prev)
      c = ν + sum(abs2, x_prev)
      sqrtΔ = sqrt(4 * c - b^2)
      Q = t -> t^2 + b * t + c
      F = t -> (t + b / 2) * log(Q(t)) - 2 * (t + b / 2) + sqrtΔ * atan(2 * t + b, sqrtΔ)
      estimate += factor * (F(Δt) - F(0) - Δt * log(ν))
      break
    end
  end

  mean_value = factor * (digamma(factor) - digamma(ν/2))
  variance_value = factor^2 * (trigamma(ν/2) - trigamma(factor))
  return (estimate - mean_value * T_end) / T_end / sqrt(variance_value)
end

function sample_symmetric_t(ν::Float64, d::Int)
  W = rand(Chisq(ν))
  Z = randn(d)
  return sqrt(ν/W) * Z
end

function Experiment_once(ν::Float64, T::Float64, FECMC, BPS_sampler)
  xinit, vinit = sample_symmetric_t(ν, d), randn(d)
  vinit = vinit ./ sqrt(sum(vinit.^2))

  T_end = d * T
  t_FECMC = @elapsed estimate_FECMC = h_estimate_online(FECMC, ν, T_end, xinit, vinit)

  xinit, vinit = sample_symmetric_t(ν, d), randn(d)
  vinit = vinit ./ sqrt(sum(vinit.^2))

  t_BPS = @elapsed estimate_BPS = h_estimate_online(BPS_sampler, ν, T_end, xinit, vinit)
  return abs2(estimate_FECMC), abs2(estimate_BPS), t_FECMC, t_BPS
end

function Experiment(ν_list; iter::Int, T::Float64)
  SE_FECMC = fill(-Float64(10^5), iter, length(ν_list))
  SE_BPS = fill(-Float64(10^5), iter, length(ν_list))
  t_FECMC = fill(-Float64(10^5), iter, length(ν_list))
  t_BPS = fill(-Float64(10^5), iter, length(ν_list))
  mkpath(dirname(OUTPUT_PATH))
  for i in ProgressBar(1:length(ν_list))
    ν = ν_list[i]
    function ∇U(x::AbstractVector)
      d = length(x)
      factor = (d + ν) / ν
      denominator = 1 + sum(abs2, x) / ν
      return factor * x ./ denominator
    end
    FECMC = ForwardECMC(d, ∇U, mix_p=1.0, switch=true)
    BPS_sampler = BPS(d, ∇U, refresh_rate = 1.424)
    for j in 1:iter
      SE_FECMC[j,i], SE_BPS[j,i], t_FECMC[j,i], t_BPS[j,i] = Experiment_once(ν, T, FECMC, BPS_sampler)
    end
  end
  save(
    OUTPUT_PATH,
    "SE_FECMC", SE_FECMC,
    "SE_BPS", SE_BPS,
    "t_FECMC", t_FECMC,
    "t_BPS", t_BPS,
  )
  return SE_FECMC, SE_BPS, t_FECMC, t_BPS
end

ν_list = [10.0, 100.0, 1000.0, 10000.0]
T = Float64(10^2)
iter = 10^3
if IS_CI
  ν_list = [10000.0]
  T = 1.0
  iter = 1
end
SE_FECMC, SE_BPS, t_FECMC, t_BPS = Experiment(ν_list; T=T, iter=iter)