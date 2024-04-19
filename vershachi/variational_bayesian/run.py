import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from .unlearn import Unlearnizable  # unlearn.py
from . import approximate_post
import pickle
from . import utils
import os

from .model_config import get_experiment


def make_approximate(approximate, nbijector, nhidden):

    if approximate == "gauss_fullcov":
        approximate_dist = approximate_post.Gaussian
        approximate_config = {}

    elif approximate == "gauss_diag":
        approximate_dist = approximate_post.GaussianDiag
        approximate_config = {}

    elif approximate == "maf":
        approximate_dist = approximate_post.MAF
        approximate_config = {
            "nbijectors": nbijector,
            "hidden_layers": [nhidden, nhidden],
        }

    elif approximate == "gaussmixture":
        approximate_dist = approximate_post.GaussianMixture
        approximate_config = {"xdim": 1, "k": 3, "independent": True}

    else:
        raise Exception("Unknown approximate distribution: {}".format(approximate))

    return approximate_dist, approximate_config


def compute_KL_divergence(
    nparam,
    approximate_dist,
    approximate_config,
    learned_param_full,
    learned_param_remain,
    all_unlearned_param,
    mode_percentages,
    nsample_to_compute_kl=20000,
):
    # compute KL divergence between unlearned distribution and q(theta|full_data)
    full_post_dist = approximate_dist(nparam, **approximate_config)
    remain_post_dist = approximate_dist(nparam, **approximate_config)
    all_unlearned_post_dist = {}
    kl_unlearned2full = {}
    kl_unlearned2remain = {}

    for mode_percentage in mode_percentages:
        unlearned_post_dist = approximate_dist(nparam, **approximate_config)
        all_unlearned_post_dist[mode_percentage] = unlearned_post_dist

        unlearned_samples = unlearned_post_dist.sample(nsample_to_compute_kl)
        unlearned_lprobs = unlearned_post_dist.log_prob(unlearned_samples)
        full_lprobs = full_post_dist.log_prob(unlearned_samples)
        remain_lprobs = remain_post_dist.log_prob(unlearned_samples)

        kl_unlearned2full[mode_percentage] = tf.reduce_mean(
            unlearned_lprobs - full_lprobs
        )
        kl_unlearned2remain[mode_percentage] = tf.reduce_mean(
            unlearned_lprobs - remain_lprobs
        )

    with tf.compat.v1.Session() as sess:
        full_post_dist.load_param(learned_param_full, sess)
        remain_post_dist.load_param(learned_param_remain, sess)

        kl_unlearned2full_np = {}
        kl_unlearned2remain_np = {}

        for i, mode_percentage in enumerate(mode_percentages):
            all_unlearned_post_dist[mode_percentage].load_param(
                all_unlearned_param[mode_percentage], sess
            )

            kl_unlearned2full_np[mode_percentage] = sess.run(
                kl_unlearned2full[mode_percentage]
            )
            kl_unlearned2remain_np[mode_percentage] = sess.run(
                kl_unlearned2remain[mode_percentage]
            )

    return kl_unlearned2full_np, kl_unlearned2remain_np


def run_variational_bayesian_unlearning(
    experiment="moon",
    approximate="gauss_fullcov",
    nbijector=5,
    nhidden=30,
    nsample=1000,
    ntrain=30000,
    batchsize=1000,
    folder="./out",
    gpu="0",
):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    approximate_dist, approximate_config = make_approximate(
        approximate, nbijector, nhidden
    )
    experiment_data = get_experiment(experiment)
    if not os.path.exists(folder):
        os.makedirs(folder)

    path = "{}/{}".format(folder, experiment)
    if not os.path.exists(path):
        os.makedirs(path)

    path = "{}/{}".format(path, approximate)
    if not os.path.exists(path):
        os.makedirs(path)

    dim = experiment_data["dim"]
    nparam = experiment_data["nparam"]
    model = experiment_data["model"]

    if experiment.startswith("moon"):
        eubo_mode_percentages = [0.5, 0.1, 1e-3, 1e-5, 1e-9, 0.0]
        elbo_mode_percentages = [0.5, 0.1, 1e-3, 1e-5, 1e-9, 0.0]
    elif experiment == "banknote_authentication1":
        if approximate == "gauss_fullcov":
            eubo_mode_percentages = [0.5, 0.1, 1e-3, 1e-5, 1e-9, 0.0]
            elbo_mode_percentages = [0.5, 0.1, 1e-3, 1e-5, 1e-9, 0.0]
        elif approximate == "maf":
            eubo_mode_percentages = [1e-3, 1e-5, 1e-7, 1e-9, 0.0]
            elbo_mode_percentages = [1e-3, 1e-5, 1e-7, 1e-9, 0.0]
    else:
        raise Exception("Unknown experiment {}".format(experiment))

    unlearn = Unlearnizable(
        dim,
        nparam,
        approximate_dist,
        approximate_config,
        model.log_likelihood,
        model.log_prior,
        nsample=nsample,
    )

    data = experiment_data["data"]
    remain_data = experiment_data["remain_data"]
    removed_data = experiment_data["removed_data"]

    filename = "{}/full_data_post.p".format(path)
    learned_param_full = utils.load_file_if_exist_else_create(
        filename,
        lambda: unlearn.learn(data, ntrain=ntrain, batchsize=batchsize),
        force_create_new=False,
    )

    filename = "{}/remain_data_retrain_post.p".format(path)
    learned_param_remain = utils.load_file_if_exist_else_create(
        filename,
        lambda: unlearn.learn(
            remain_data,
            init_post_param=learned_param_full,
            ntrain=ntrain,
            batchsize=batchsize,
        ),
        force_create_new=False,
    )

    approximate_dist_instance = approximate_dist(nparam, **approximate_config)
    mode_input, mode_lprob = approximate_dist_instance.find_mode(
        param=learned_param_full, n_init=10, ntrain=10000
    )

    print("Remaining data: {}".format(remain_data.shape))
    print("Removed data: {}".format(removed_data.shape))

    print("Mode probability: {}".format(np.exp(mode_lprob)))

    print("*** Unlearn ***")
    all_eubo_unlearned_param = {}

    for mode_percentage in eubo_mode_percentages:
        if mode_percentage > 0:
            log_threshold = mode_lprob + np.log(mode_percentage)
        else:
            log_threshold = -np.inf

        filename = "{}/data_remain_data_by_unlearn_eubo_{}.p".format(
            folder, mode_percentage
        )
        unlearned_param = utils.load_file_if_exist_else_create(
            filename,
            lambda: unlearn.unlearn_EUBO(
                removed_data,
                learned_param_full,
                log_threshold=log_threshold,
                ntrain=ntrain,
                batchsize=batchsize,
            ),
            force_create_new=False,
        )

        all_eubo_unlearned_param[mode_percentage] = unlearned_param

    all_elbo_unlearned_param = {}

    for mode_percentage in elbo_mode_percentages:
        if mode_percentage > 0.0:
            log_threshold = mode_lprob + np.log(mode_percentage)
        else:
            log_threshold = -np.inf

        filename = "{}/data_remain_data_by_unlearn_elbo_{}.p".format(
            folder, mode_percentage
        )
        unlearned_param = utils.load_file_if_exist_else_create(
            filename,
            lambda: unlearn.unlearn_ELBO(
                removed_data,
                learned_param_full,
                log_threshold=log_threshold,
                ntrain=ntrain,
                batchsize=batchsize,
            ),
            force_create_new=False,
        )

        all_elbo_unlearned_param[mode_percentage] = unlearned_param


# Example usage:
# run_variational_bayesian_unlearning(experiment="moon", approximate="gauss_fullcov")
