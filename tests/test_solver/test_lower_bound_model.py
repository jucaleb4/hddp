from src.solver import LowerBoundModel
import numpy as np

def test_lower_bound_model_constructor():
    """
    Test lower bound produces correct initial lower bound.
    """
    n = 3
    initial_lb = -100

    lbm = LowerBoundModel(n, initial_lb)

    x = np.zeros(n)
    lb_eval = lbm.evaluate_lb(x)

    assert lb_eval == initial_lb

def test_lower_bound_model_unbounded():
    """
    Test output and raised error when lower bound model is unbounded.
    """
    n = 3

    lbm = LowerBoundModel(n)

    x = np.zeros(n)
    lb_eval = lbm.evaluate_lb(x)

    assert lb_eval == -float('inf')

def test_lower_bound_add_cut():
    """
    Add cuts and make sure lower bound is correct.
    """
    n = 3
    initial_lb = -100

    lbm = LowerBoundModel(n, initial_lb)

    # (val, grad, x_center)
    [val, grad, x_center] = [-100, np.array([20, 40, -50]), -np.ones(3)]
    lbm.add_cut(val, grad, x_center)

    x = np.ones(n)
    lb_eval = lbm.evaluate_lb(x)
    first_eval = val + np.dot(grad, x - x_center)

    assert lb_eval == first_eval

    # new cut should not change the max value at 0
    [val, grad, x_center] = [0, np.ones(3), 100 * np.ones(3)]
    lbm.add_cut(val, grad, x_center)
    lb_eval = lbm.evaluate_lb(x)

    assert lb_eval == first_eval

    # evaluate at a point that is very negative for all cuts
    x = -100 * np.ones(n)
    lb_eval = lbm.evaluate_lb(x)

    assert lb_eval == initial_lb

def test_lower_bound_monotone():
    """
    Test that the lower bound is monotone (w.r.t number of cuts).
    """
    n_trials = 10
    n_iters = 10

    for _ in range(n_trials):
        n = 10
        initial_lb = -100
        lbm = LowerBoundModel(n, initial_lb)

        # number of points
        x = 100*(1-2*np.random.random(n))
        curr_eval_lb = initial_lb

        for _ in range(n_iters):
            val = 1-2*np.random.random()
            grad = 10*(1-2*np.random.random(n))
            x_center = 100*(1-2*np.random.random(n))
            lbm.add_cut(val, grad, x_center)

            new_eval_lb = lbm.evaluate_lb(x)

            assert new_eval_lb >= curr_eval_lb

            curr_eval_lb = new_eval_lb