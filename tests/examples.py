import numpy as np
from typing import Optional, Tuple, TypeAlias

# Type alias for the function return type f_val, gradient, hessian
FunctionEvaluationResult: TypeAlias = Tuple[float, np.ndarray, Optional[np.ndarray]]

# --- Test QP -- #
def f_qp(x: np.ndarray, eval_hessian: bool = True) -> FunctionEvaluationResult:
    """
    Quadratic function for testing.
    
    Parameters
    ----------
    x : np.ndarray
        Input vector.
    eval_hessian : bool, optional
        Whether to evaluate the Hessian. Default is True.
    Returns
    -------
    FunctionEvaluationResult
        Tuple containing function value, gradient, and Hessian (if eval_hessian is True).
    """
    f_val = x[0]**2 + x[1]**2 + (x[2] + 1)**2
    grad = np.array([2 * x[0], 2 * x[1], 2 * (x[2] + 1)])
    hess = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]) if eval_hessian else None
    return f_val, grad, hess

eq_constraints_mat_qp = np.array([[1, 1, 1]])  # x[0] + x[1] + x[2]
eq_constraints_rhs_qp = np.array([1])  # = 1

ineq_qp = [
    lambda x, eval_hessian=True:
        (-x[0], np.array([-1,0,0]), np.zeros((3,3))),
    lambda x, eval_hessian=True:
        (-x[1], np.array([0,-1,0]), np.zeros((3,3))),
    lambda x, eval_hessian=True:
        (-x[2], np.array([0,0,-1]), np.zeros((3,3))),
]

# --- Test LP -- #
def f_lp(x: np.ndarray, eval_hessian: bool = True) -> FunctionEvaluationResult:
    """
    Linear function for testing.
    
    Parameters
    ----------
    x : np.ndarray
        Input vector.
    eval_hessian : bool, optional
        Whether to evaluate the Hessian. Default is True.
    Returns
    -------
    FunctionEvaluationResult
        Tuple containing function value, gradient, and Hessian (if eval_hessian is True).
    """
    f_val = -x[0] - x[1]
    grad = np.array([-1, -1])
    hess = np.zeros((2, 2)) if eval_hessian else None
    return f_val, grad, hess

ineq_lp = [
    lambda x, eval_hessian=True:
        (-x[0] - x[1] + 1, np.array([-1, -1]), np.zeros((2, 2))),
    lambda x, eval_hessian=True:
        (x[1] - 1, np.array([0, 1]), np.zeros((2, 2))),
    lambda x, eval_hessian=True:
        (x[0] - 2, np.array([1, 0]), np.zeros((2, 2))),
    lambda x, eval_hessian=True:
        (-x[1], np.array([0, -1]), np.zeros((2, 2))),
]