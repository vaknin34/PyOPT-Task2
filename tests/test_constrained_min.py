import unittest
import numpy as np
import os

from src.constrained_min import ConstrainedMin
from examples import (f_qp, ineq_qp, eq_constraints_mat_qp, eq_constraints_rhs_qp,
                      f_lp, ineq_lp)

import src.plot_utils as pu

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


class TestMinimizer(unittest.TestCase):

    def test_qp(self):
        solver = ConstrainedMin()
        x0_qp = np.array([0.1, 0.2, 0.7])
        x_star_qp = solver.interior_pt(
            f=f_qp,
            ineq_constraints=ineq_qp,
            eq_constraints_mat=eq_constraints_mat_qp,
            eq_constraints_rhs=eq_constraints_rhs_qp,
            x0=x0_qp
        )

        pu.plot_central_path_qp(solver.opt_path, f_qp, f"{PLOT_DIR}/qp_path.png")
        pu.plot_objective_history(solver.obj_val_path,
                                f"{PLOT_DIR}/qp_obj.png",
                                "QP objective vs. outer iter")
        pu.report_final(x_star_qp, f_qp, ineq_qp,
                        eq_constraints_mat_qp, eq_constraints_rhs_qp)


    def test_lp(self):
        solver_lp = ConstrainedMin()
        x0_lp = np.array([0.5, 0.75])
        x_star_lp = solver_lp.interior_pt(
            f=f_lp,
            ineq_constraints=ineq_lp,
            eq_constraints_mat=None,
            eq_constraints_rhs=None,
            x0=x0_lp
        )

        pu.plot_central_path_lp(solver_lp.opt_path, f_lp, f"{PLOT_DIR}/lp_path.png")
        pu.plot_objective_history(solver_lp.obj_val_path,
                                f"{PLOT_DIR}/lp_obj.png",
                                "LP objective vs. outer iter")
        pu.report_final(x_star_lp, f_lp, ineq_lp)

if __name__ == '__main__':
    unittest.main()