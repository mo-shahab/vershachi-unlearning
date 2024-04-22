from vershachi.variational_bayesian import plot_kl_distance_mean

plot_kl_distance_mean.plot_likelihood_diffs(
    folder="plot_data",
    exper="moon",
    appr="gauss_fullcov",
    plot_retrain_vs_full=True,
    plot_legend=True,
    plot_eubo=True,
)
