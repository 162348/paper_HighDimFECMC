using PDMPFlux, LinearAlgebra, Plots, ProgressBars, Statistics, StatsPlots, JLD2, CategoricalArrays

const IS_CI = get(ENV, "CI", "false") == "true" || get(ENV, "GITHUB_ACTIONS", "false") == "true"
const OUTPUT_PATH = joinpath(@__DIR__, "Data", "1-StandardGauss.jld2")

@inline function ∇U(x::AbstractVector)
  return x
end

function h_estimate_online(
  sampler::PDMPFlux.AbstractPDMP,
  T_end::Float64,
  xinit::Vector{Float64},
  vinit::Vector{Float64};
  seed::Union{Int, Nothing}=nothing,
)
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
  N = 0

  t_prev = state.t
  x_prev = copy(state.x)
  v_prev = copy(state.v)

  while t_prev < T_end
    PDMPFlux.get_event_state!(state, sampler)
    t_next = state.t
    N += 1

    if t_next <= T_end
      Δt = t_next - t_prev
      nx = dot(x_prev, x_prev)
      xv = dot(x_prev, v_prev)
      estimate += nx * Δt + (Δt^3) / 3 + (Δt^2) * xv

      t_prev = t_next
      copyto!(x_prev, state.x)
      copyto!(v_prev, state.v)
    else
      Δt = T_end - t_prev
      nx = dot(x_prev, x_prev)
      xv = dot(x_prev, v_prev)
      estimate += nx * Δt + (Δt^3) / 3 + (Δt^2) * xv
      break
    end
  end

  return ((estimate - d * T_end) / sqrt(d) / T_end, N)  # CAUTION: inprecise denominator (intentionally); will be compensated before plotting
end

function Experiment_once(d::Int, T::Float64, FECMC, BPS_sampler)
  xinit, vinit = randn(d), randn(d)
  vinit = vinit ./ sqrt(sum(vinit.^2))

  t_FECMC = @elapsed estimate_FECMC, N_FECMC = h_estimate_online(FECMC, d * T, xinit, vinit)

  xinit, vinit = randn(d), randn(d)
  vinit = vinit ./ sqrt(sum(vinit.^2))

  t_BPS = @elapsed estimate_BPS, N_BPS = h_estimate_online(BPS_sampler, d * T, xinit, vinit)
  return abs2(estimate_FECMC), abs2(estimate_BPS), t_FECMC, t_BPS, N_FECMC, N_BPS
end

function Experiment(d_list; T::Float64, iter::Int)
  SE_FECMC = fill(-Float64(10^5), iter, length(d_list))
  SE_BPS = fill(-Float64(10^5), iter, length(d_list))
  elpt_FECMC = fill(-Float64(10^5), iter, length(d_list))
  elpt_BPS = fill(-Float64(10^5), iter, length(d_list))
  N_FECMC = fill(-Float64(10^5), iter, length(d_list))
  N_BPS = fill(-Float64(10^5), iter, length(d_list))
  mkpath(dirname(OUTPUT_PATH))
  for i in ProgressBar(1:length(d_list))
    d = d_list[i]
    FECMC = ForwardECMC(d, ∇U, mix_p=0.05, switch=true)
    BPS_sampler = BPS(d, ∇U, refresh_rate = 1.42)
    for j in 1:iter
      if j % 100 == 0
        @info "Experiments in dimension $d: progress: $j/$iter"
      end
      SE_FECMC[j,i], SE_BPS[j,i], elpt_FECMC[j,i], elpt_BPS[j,i], N_FECMC[j,i], N_BPS[j,i] = Experiment_once(d, T, FECMC, BPS_sampler)
    end
    save(
      OUTPUT_PATH,
      "SE_FECMC", SE_FECMC,
      "SE_BPS", SE_BPS,
      "elpt_FECMC", elpt_FECMC,
      "elpt_BPS", elpt_BPS,
      "N_FECMC", N_FECMC,
      "N_BPS", N_BPS,
    )
  end
  return SE_FECMC, SE_BPS, elpt_FECMC, elpt_BPS, N_FECMC, N_BPS
end

d_list = [10, 20, 40, 80, 160, 320]
T, iter = Float64(10^2), 10^3  # paper version: will take several hours
T, iter = Float64(10^2), 10  # test version: will take several minutes
if IS_CI
  d_list = [10]
  T, iter = 1.0, 1
end
SE_FECMC, SE_BPS, elpt_FECMC, elpt_BPS, N_FECMC, N_BPS = Experiment(d_list; T=T, iter=iter)

data = [SE_FECMC SE_BPS]
iter  = size(data, 1)
n_dim = length(d_list)
grps = repeat(d_list, inner=iter, outer=2)
grps = categorical(grps; ordered=true, levels=d_list)
groups = vcat(fill("FECMC", iter*n_dim), fill("BPS", iter*n_dim))

if !IS_CI
  groupedboxplot(grps, vec(data);
      group = groups,
      legend = :bottomleft,
      xlabel = "Dimension",
      ylabel = "Squared Error",
      yscale = :log10,
      title = "Standard Gaussian Target"
  )
end