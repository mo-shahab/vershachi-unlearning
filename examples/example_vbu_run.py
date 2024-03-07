from vershachi.variational_bayesian.run import run_variational_bayesian_unlearning

run_variational_bayesian_unlearning(
    experiment="moon", approximate="gauss_fullcov", nsample=100, ntrain=300
)
