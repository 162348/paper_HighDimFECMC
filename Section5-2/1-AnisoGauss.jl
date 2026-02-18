using PDMPFlux, LinearAlgebra, Plots, ProgressBars, Statistics, StatsPlots, Distributions, JLD2, CategoricalArrays

const IS_CI = get(ENV, "CI", "false") == "true" || get(ENV, "GITHUB_ACTIONS", "false") == "true"
const d = IS_CI ? 10 : 100
const OUTPUT_PATH = joinpath(@__DIR__, "Data", "1-AnisoGauss.jld2")

function h_estimate_online(
  sampler::PDMPFlux.AbstractPDMP,
  γ::Float64,
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

  t_prev = state.t
  x_prev = copy(state.x)
  v_prev = copy(state.v)

  while t_prev < T_end
    PDMPFlux.get_event_state!(state, sampler)  # mutates state in-place; advances state.t to next accepted event
    t_next = state.t

    if t_next <= T_end
      Δt = t_next - t_prev
      nx = dot(x_prev, x_prev)
      xv = dot(x_prev, v_prev)
      estimate += (nx * Δt + (Δt^3) / 3 + (Δt^2) * xv) / (1 - γ)

      sx = sum(x_prev)
      sv = sum(v_prev)
      factor = γ / (1-γ) / (1-γ+γ*d)
      estimate -= factor * (sx^2 * Δt + sv * sx * Δt^2 + sv^2 * Δt^3 / 3)

      t_prev = t_next
      copyto!(x_prev, state.x)
      copyto!(v_prev, state.v)
    else
      Δt = T_end - t_prev
      nx = dot(x_prev, x_prev)
      xv = dot(x_prev, v_prev)
      estimate += (nx * Δt + (Δt^3) / 3 + (Δt^2) * xv) / (1 - γ)

      sx = sum(x_prev)
      sv = sum(v_prev)
      factor = γ / (1-γ) / (1-γ+γ*d)
      estimate -= factor * (sx^2 * Δt + sv * sx * Δt^2 + sv^2 * Δt^3 / 3)
      break
    end
  end

  return (estimate - d * T_end) / sqrt(d) / T_end  # CAUTION: inprecise denominator (intentionally)
end

function Experiment_once(γ::Float64, T::Float64, FECMC, BPS_sampler)
  xinit = sqrt(1 - γ) .* randn(d)
  xinit .+= sqrt(γ) * randn()
  vinit = randn(d)
  vinit ./= sqrt(dot(vinit, vinit))

  T_end = d * T
  t_FECMC = @elapsed estimate_FECMC = h_estimate_online(FECMC, γ, T_end, xinit, vinit)

  xinit = sqrt(1 - γ) .* randn(d)
  xinit .+= sqrt(γ) * randn()
  vinit = randn(d)
  vinit ./= sqrt(dot(vinit, vinit))

  t_BPS = @elapsed estimate_BPS = h_estimate_online(BPS_sampler, γ, T_end, xinit, vinit)
  return abs2(estimate_FECMC), abs2(estimate_BPS), t_FECMC, t_BPS
end

function Experiment(γ_list; T::Float64, iter::Int, dt::Float64=0.1)
  SE_FECMC = fill(-Float64(10^5), iter, length(γ_list))
  SE_BPS = fill(-Float64(10^5), iter, length(γ_list))
  t_FECMC = fill(-Float64(10^5), iter, length(γ_list))
  t_BPS = fill(-Float64(10^5), iter, length(γ_list))
  mkpath(dirname(OUTPUT_PATH))
  for i in ProgressBar(1:length(γ_list))
    γ = γ_list[i]
    @inline function ∇U(x::AbstractVector)
      d = length(x)
      a = 1 - γ
      inv_a = 1 / a
      c = γ / (a * (a + γ * d))
      return inv_a .* x .- (c .* sum(x))
    end
    FECMC = ForwardECMC(d, ∇U, mix_p=0.05, switch=true)
    BPS_sampler = BPS(d, ∇U, refresh_rate = 1.424)
    for j in 1:iter
      if j % 100 == 0
        @info "Experiments in γ = $γ: progress: $j/$iter"
      end
      SE_FECMC[j,i], SE_BPS[j,i], t_FECMC[j,i], t_BPS[j,i] = Experiment_once(γ, T, FECMC, BPS_sampler)
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

T, iter = Float64(10^2), 10^3
γ_list = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
if IS_CI
  T, iter = 1.0, 1
  γ_list = [0.5]
end
SE_FECMC, SE_BPS, t_FECMC, t_BPS = Experiment(γ_list; T=T, iter=iter)