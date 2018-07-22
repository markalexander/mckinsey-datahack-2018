# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize


def get_optimal_policy_incentives(policy_renewal_probs,
                                  policy_premiums,
                                  cap_renewal_prob=True) -> np.ndarray:
    """Calculates the optimal policy incentive regime for the given policy
    renewal probabilities and premiums.

    Args:
        policy_renewal_probs: The renewal probabilities of the policies.
        policy_premiums:      The premiums for the policies.
        cap_renewal_prob:     Whether to cap probabilities at 1 or not.

    Returns:
        The optimal incentive scheme.

    """

    print('Optimizing over policy incentives')

    # Start building build bounds and initial guesses
    initial_guesses = np.ones((len(policy_premiums),)) * 1000
    bounds = []
    search_region = 0

    # Calculate the max possible percentage increase in renewal prob for an
    # infinite reward
    max_pct_improvement_in_renewal_prob\
        = pct_improvement_in_renewal_prob(agent_hours_invested(np.inf))

    # Set bounds on a per-policy basis
    # Array ops would be nicer but for some reason scipy minimize() wants a
    # list of tuples anyway...
    for i in range(len(policy_premiums)):

        # Set bounds and initial guess for this policy
        # These don't need to be particularly tight, they just help reduce
        # search space

        # Incentive must be non-negative
        lb = 0

        # We only really care about how much the incentive will increase the
        # *marginal* expected value of the policy, i.e. Delta p * premium.
        #
        # Two restrictions apply to the value Delta p:
        #
        #     1. Bounded above by the maximum increase attainable by the agent
        #        (~0.1729, as calculated above in
        #        max_pct_improvement_in_renewal_prob).
        #     2. Delta p shouldn't cause the overall probability of renewal to
        #        exceed 1.0, and so it can be at most (1 - p).
        #
        # Thus:
        max_delta_p = max_pct_improvement_in_renewal_prob *\
                      policy_renewal_probs[i]
        if cap_renewal_prob:
            max_delta_p = min(
                max_pct_improvement_in_renewal_prob * policy_renewal_probs[i],
                1 - policy_renewal_probs[i]
            )

        ub = max_delta_p * policy_premiums[i]

        x0 = (ub - lb) / 2
        bounds.append((lb, ub))
        initial_guesses[i] = x0
        search_region += (ub - lb)

    print('Mean search region size: {:,}'.format(
        search_region / len(policy_premiums)
    ))

    # Run the minimization procedure
    res = minimize(
        neg_total_net_revenue,
        initial_guesses,
        (policy_renewal_probs, policy_premiums),
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': 100*15000,
            'maxfun': 100*15000,
            'disp': True,
            'gtol': np.finfo(float).eps,
            'eps': 1e-4
        }
    )
    print(res)

    policy_incentives = res.x

    # Hacky check for ideal convergence
    tol = 10
    best_revenue = 0
    best_delta = None
    for delta in range(-tol * 100, tol * 100, tol):
        this_revenue = total_net_revenue(policy_incentives + delta,
                                         policy_renewal_probs,
                                         policy_premiums)
        if this_revenue > best_revenue:
            best_revenue = this_revenue
            best_delta = delta
    print('Max revenue {:,} occurs at {}'.format(best_revenue, best_delta))
    if best_delta != 0:
        print('Try re-running BFGS with tighter tolerances or more iterations')
    policy_incentives += best_delta
    policy_incentives = np.clip(policy_incentives, 0, None)

    return policy_incentives


def agent_hours_invested(policy_incentives: np.ndarray) -> np.ndarray:
    """Get the number of hours of effort that will be invested by an agent
    towards obtaining a renewal on a policy.

    Args:
        policy_incentives: The value of the incentives attached to the policy.

    Returns:
        The number of hours the agent will invest.

    """
    return 10 * (1 - np.exp(-policy_incentives / 400))


def pct_improvement_in_renewal_prob(agent_hours: np.ndarray) -> np.ndarray:
    """Get the number % improvement in renewal probability given the number of
    hours of effort invested by the agent.

    Args:
        agent_hours: The number of hours invested by the agent on this policy.

    Returns:
        The percentage improvement.

    """
    return (20 * (1 - np.exp(-agent_hours / 5))) / 100


def total_net_revenue(policy_incentives: np.ndarray,
                      policy_renewal_probs: np.ndarray,
                      policy_premiums: np.ndarray) -> np.ndarray:
    """Calculate the total net revenue over all policies for the given set of
    variables.

    Args:
        policy_incentives:    The incentives given to agents for each policy.
        policy_renewal_probs: The probabilities of renewal for the policies.
        policy_premiums:      The premium paid for each policy.

    Returns:
        The total net revenue.
    """
    # Adjust overall renewal probs for agent response to incentives
    agent_hours = agent_hours_invested(policy_incentives)
    pct_prob_improv = pct_improvement_in_renewal_prob(agent_hours)
    total_renewal_probs = policy_renewal_probs + np.multiply(
        policy_renewal_probs, pct_prob_improv
    )
    np.clip(total_renewal_probs, 0, 1)  # 0 <= p <= 1 for any probability
    # Calc net revenues, sum over all policies, and return
    net_revenues = np.multiply(
        total_renewal_probs,
        policy_premiums
    ) - policy_incentives
    return net_revenues.sum()


def neg_total_net_revenue(*args, **kwargs) -> np.ndarray:
    """Get the negative of the total net revenue.

    Utility function for optimization procedures.

    Args:
        *args:    Set of args to pass to total_net_revenue().
        **kwargs: Set of kwargs to pass to total_net_revenue().

    Returns:
        The negative value of the total net revenue.

    """
    return -1 * total_net_revenue(*args, **kwargs)
