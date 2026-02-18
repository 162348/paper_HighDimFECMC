using PDMPFlux, LinearAlgebra, Plots, ProgressBars, Statistics, StatsPlots, JLD2, CategoricalArrays

const IS_CI = get(ENV, "CI", "false") == "true" || get(ENV, "GITHUB_ACTIONS", "false") == "true"
const OUTPUT_PATH = joinpath(@__DIR__, "Data", "1-StandardGauss.jld2")

@inline function ∇U(x::AbstractVector)
  return x
end

function BM_estimate_online(
  sampler::PDMPFlux.AbstractPDMP,
  T_end::Float64,
  xinit::Vector{Float64},
  vinit::Vector{Float64};
  seed::Union{Int, Nothing}=nothing,
  B::Int64=50,
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

  T_batch = T_end / B
  Ts = range(0.0, T_end; length=B+1)
  Ts = [Ts; Inf]
  Batches_h_unnormalised = fill(0.0, B)
  Batches_g = fill(0.0, B)
  estimate = 0.0
  index = 2

  # Integrate the deterministic flow between accepted events and split contributions at batch boundaries.
  t_prev = state.t
  x_prev = copy(state.x)
  v_prev = copy(state.v)

  while t_prev < T_end
    PDMPFlux.get_event_state!(state, sampler)
    t_next = state.t

    if t_next <= T_end
      while t_next >= Ts[index]
        Δt = Ts[index] - t_prev
        nx = dot(x_prev, x_prev)
        xv = dot(x_prev, v_prev)
        estimate += nx * Δt + (Δt^3) / 3 + (Δt^2) * xv
        Batches_h_unnormalised[index-1] += nx * Δt + (Δt^3) / 3 + (Δt^2) * xv
        Batches_g[index-1] += (xv * Δt + (Δt^2) / 2) / T_batch

        t_prev = Ts[index]
        x_prev += v_prev * Δt
        index += 1
      end
      Δt = t_next - t_prev
      nx = dot(x_prev, x_prev)
      xv = dot(x_prev, v_prev)
      estimate += nx * Δt + (Δt^3) / 3 + (Δt^2) * xv
      Batches_h_unnormalised[index-1] += nx * Δt + (Δt^3) / 3 + (Δt^2) * xv
      Batches_g[index-1] += (xv * Δt + (Δt^2) / 2) / T_batch
      t_prev = t_next
      copyto!(x_prev, state.x)
      copyto!(v_prev, state.v)
    else
      while t_next >= Ts[index]
        Δt = Ts[index] - t_prev
        nx = dot(x_prev, x_prev)
        xv = dot(x_prev, v_prev)
        estimate += nx * Δt + (Δt^3) / 3 + (Δt^2) * xv
        Batches_h_unnormalised[index-1] += nx * Δt + (Δt^3) / 3 + (Δt^2) * xv
        Batches_g[index-1] += (xv * Δt + (Δt^2) / 2) / T_batch

        t_prev = Ts[index]
        x_prev += v_prev * Δt
        index += 1
      end
      Δt = T_end - t_prev
      nx = dot(x_prev, x_prev)
      xv = dot(x_prev, v_prev)
      estimate += nx * Δt + (Δt^3) / 3 + (Δt^2) * xv
      Batches_h_unnormalised[end] += nx * Δt + (Δt^3) / 3 + (Δt^2) * xv
      Batches_g[end] += (xv * Δt + (Δt^2) / 2) / T_batch
      break
    end
  end

  Batches_h = (Batches_h_unnormalised .- d * (T_batch)) ./ (sqrt(2d) * T_batch)
  BM_h = var(Batches_h) * T_batch
  BM_g = var(Batches_g) * T_batch

  return BM_h / d, BM_g
end

function Experiment_once(d::Int, T::Float64, FECMC)
  xinit, vinit = randn(d), randn(d)
  vinit = vinit ./ sqrt(sum(vinit.^2))

  BM_h, _ = BM_estimate_online(FECMC, d * T, xinit, vinit; B=400)
  _, BM_g = BM_estimate_online(FECMC, d * T, xinit, vinit; B=Int(floor(1.5 * 10^4 * sqrt(d))))

  return BM_h, BM_g
end

function Experiment(d_list; T::Float64, iter::Int)
  BMs_h = fill(-Float64(10^5), iter, length(d_list))
  BMs_g = fill(-Float64(10^5), iter, length(d_list))
  mkpath(dirname(OUTPUT_PATH))
  for i in ProgressBar(1:length(d_list))
    d = d_list[i]
    FECMC = ForwardECMC(d, ∇U, mix_p=1.0, switch=true)
    for j in 1:iter
      if j % 100 == 0
        @info "Experiments in dimension $d: progress: $j/$iter"
      end
      BMs_h[j,i], BMs_g[j,i] = Experiment_once(d, T, FECMC)
    end
    save(OUTPUT_PATH, "BMs_h", BMs_h, "BMs_g", BMs_g)
  end
  return BMs_h, BMs_g
end

d_list = [10, 20, 40, 80, 160, 320]
T = Float64(10^2)
iter = 10
if IS_CI
  d_list = [10]
  T = 1.0
  iter = 1
end
BMs_h, BMs_g = Experiment(d_list; T=T, iter=iter)

data = [BMs_h BMs_g]
iter  = size(data, 1)
n_dim = length(d_list)
grps = repeat(d_list, inner=iter, outer=2)
grps = categorical(grps; ordered=true, levels=d_list)
groups = vcat(fill("BM for h", iter*n_dim), fill("BM for g", iter*n_dim))

if !IS_CI
  groupedboxplot(grps, vec(data);
      group = groups,
      legend = :right,
      xlabel = "Dimension",
      ylabel = "Batch Means Estimate",
      # yscale = :log10,
      title = "Standard Gaussian Target"
  )
end