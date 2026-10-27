# ML Systems Labs

Python 3.11.8. Installation: pip install -r requirements.txt

Note: Because pyarrow==16.1.0 in the lab manual conflicts with mlflow==2.14.1 (mlflow requires pyarrow<16), pyarrow==15.0.2 was used.
## Lab 2
Run from `lab02/`: `python src/run_checks.py`, `python src/train.py`, `python src/timing.py`.
Note: "N(0, 0.01)" in the manual can mean std = 0.01 or variance = 0.01 (std = 0.1).
`train.py` runs both readings, and the report compares them.
