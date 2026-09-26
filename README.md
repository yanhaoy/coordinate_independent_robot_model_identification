# Coordinate-Independent Robot Model Identification

A runnable tutorial for [*Coordinate-Independent Robot Model Identification*](https://arxiv.org/abs/2603.14656) by Yanhao Yang and Ross L. Hatton.

Inverse-dynamics identification often minimizes residuals in generalized force coordinates. That error can change when coordinates, units, or scaling change. The paper instead measures each residual with the dual metric induced by the robot's mass matrix, giving a coordinate-independent objective. This tutorial shows how to formulate and solve that objective with physically consistent inertial parameters.

## Try the tutorial

[![Try it on Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yanhaoy/coordinate_independent_robot_model_identification/blob/main/src/coordinate_independent_robot_model_identification/minimal_example.ipynb)

[Open the notebook on GitHub](src/coordinate_independent_robot_model_identification/minimal_example.ipynb) and run its cells in order. The notebook installs MuJoCo and MeshCat in its first code cell. It also uses NumPy, SciPy, CVXPY, and Clarabel; install any of these if they are missing from your notebook environment. The default solver selection uses MOSEK when available and falls back to Clarabel otherwise.

The example creates noisy measurements from a synthetic three-joint arm, builds inverse-dynamics and mass-matrix regressors, and estimates ten inertial parameters. It compares least squares (LS), covariance-weighted least squares (WLS), and the dual-metric objective on full and downsampled data. The printed geodesic distance measures each estimate against the known synthetic inertia. Edit `ROBOT` in the experiment settings to change joint positions and axes, gravity, or the terminal body's mass and inertia. The notebook displays the arm in MeshCat when its cells run. On Colab, this is an interactive snapshot; rerun the view cell to display a changed pose.

## Run locally

Install [uv](https://docs.astral.sh/uv/), then run from the repository root:

```bash
uv sync
uv run python src/coordinate_independent_robot_model_identification/minimal_example.py --solver CLARABEL
```

The project targets Python 3.13. `--solver CLARABEL` runs without a MOSEK license. To use MOSEK's faster Fusion formulation, configure a MOSEK license and pass `--solver MOSEK`. The script also accepts `--seed` (default: `42`), `--solver auto` (the default), and `--solver MOSEK_PRIMAL`. Add `--show-robot` to open the MeshCat view after fitting; the script prints its URL and keeps the viewer available until Ctrl+C.

## Cite the paper

```bibtex
@INPROCEEDINGS{yang2026coordinate,
  title={Coordinate-Independent Robot Model Identification}, 
  author={Yanhao Yang and Ross L. Hatton},
  year={2026},
  booktitle={2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  url_Info={https://arxiv.org/abs/2603.14656}, 
  url_PDF={https://arxiv.org/pdf/2603.14656}
}
```

## License

This repository is released under the [MIT License](LICENSE).
