
using Distributions
using Plots
using Random
using Lux
using StatsFuns
using Optimisers
using Zygote
using AbstractGPs

# Create toy data as in 
# https://juliagaussianprocesses.github.io/AbstractGPs.jl/dev/examples/2-deep-kernel-learning/
rng = Random.default_rng()
xmin, xmax = (-3, 3)
N = 150
noise_std = 0.01
x_train = rand(Uniform(xmin, xmax), N)
target_f(x) = sinc(abs(x)^abs(x))
y_train = target_f.(x_train) + randn(N) * noise_std
x_test = range(xmin, xmax; length=200)
plot(xmin:0.01:xmax, target_f; label="ground truth")
scatter!(x_train, y_train; label="training data")


# Create a Lux GP layer
struct LuxGP <: Lux.AbstractExplicitLayer
    init_θ
end
LuxGP(θ::AbstractArray) = LuxGP(() -> copy(θ))
Lux.initialparameters(rng::AbstractRNG, layer::LuxGP) = (θ=layer.init_θ(),)
Lux.initialstates(rng::AbstractRNG, layer::LuxGP) = (;)
k(θ) = softplus(θ[1])*(Matern52Kernel() ∘ ScaleTransform(softplus(θ[2])))
σ(θ) = softplus(θ)
(l::LuxGP)(x, ps, st) = begin
    θ = ps.θ
    f = GP(k(θ[2:end]))
    fx = f(x, σ(θ[1]))    
    fx, st
end

make_model(hidden_dim, latent_dim) = Chain(
    Dense(1, hidden_dim, tanh),
    Dense(hidden_dim, latent_dim),
    LuxGP(rand(3))
)

xs = Matrix(reduce(vcat, x_train)')
model = make_model(20, 5)
ps, st = Lux.setup(rng, model)
opt = Optimisers.AdamW(5e-2)
opt_state = Optimisers.setup(opt, ps)
nll(ps, st, x, y) = -logpdf(first(model(x, ps, st)), y)

losses = Float64[]
for i in 1:300
    gs = first(gradient(ps -> nll(ps, st, xs, y_train), ps))
    opt_state, ps = Optimisers.update(opt_state, ps, gs)
    push!(losses, nll(ps, st, xs, y_train))
end
plot(losses)

test_xs = collect(minimum(x_train):0.001:maximum(x_train))'
opt_fx, _ = model(xs, ps, st)
opt_p_fx = posterior(opt_fx, y_train)

# Since our GP now lives in the latent space of our network
# we need to pass new observations through the network
# before we can compute their posterior marginals

"""Assumes the GP is the last layer in the chain 
and that all the ps and st are ordered according to their 
corresponding layers."""
make_encoder(model, ps, st) = begin    
    n = length(model) - 1
    encoder = model[1:n]
    ps_ = NamedTuple(first(pairs(ps), n))
    st_ = NamedTuple(first(pairs(st), n))
    x -> first(encoder(x, ps_, st_))
end
encode = make_encoder(model, ps, st)
test_opt_p_fx = opt_p_fx(encode(test_xs))

scatter(
    x_train,
    y_train;
    xlabel="x",
    ylabel="y",
    title="posterior (optimized parameters)",
    label="Train Data",
)
py = marginals(test_opt_p_fx)
plot!(test_xs[:], mean.(py), ribbon=2*std.(py), label="Posterior")
