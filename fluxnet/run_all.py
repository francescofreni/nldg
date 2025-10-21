import argparse
import os
import subprocess
import sys

# --- Usage: python run_all.py GPP -- (where GPP is the target)

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--target",
    type=str,
    help="Target variable to predict (e.g., GPP, ET)",
)
argparser.add_argument(
    "--agg",
    type=str,
    default="daily",
    help="Data aggregation level (default: daily)",
)
argparser.add_argument(
    "--exp_name",
    type=str,
    default=None,
    help="Custom experiment name (optional)",
)
argparser.add_argument(
    "--override",
    action="store_true",
    help="Override existing results",
)
args = argparser.parse_args()
target = args.target
agg = args.agg
exp_name = args.exp_name
override = args.override

# Define models and methods
models = ["lr", "xgb", "rf"] # "gam", 
risks = ["erm", "mse"]

# First set of args
args = [(x, "erm", "mse") for x in models]

# Add second set
for x in ["rf", "gam"]:
    for y in ["mse", "regret", "reward"]:
        args.append((x, "maxrm", y))

# Run all experiments
for model, method, risk in args:
    filename = f"{agg}_insite_{target}_{model}_{method}_{risk}.csv"
    if not override:
        if os.path.exists(
            os.path.join("results", exp_name if exp_name else "", filename)
        ):
            print(f"Skipping existing result: {filename}")
            continue
    if exp_name is not None:
        filename = f"{exp_name}/" + filename
    cmd = [
        "python", "run_experiment.py",
        "--setting", "insite",
        "--agg", agg,
        "--target", target,
        "--model_name", model,
        "--method", method,
        "--risk", risk
    ]
    if exp_name is not None:
        cmd += ["--exp_name", exp_name]
    print("Running:", " ".join(cmd))
    # run and stream output in real-time; stop all on first error
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in proc.stdout:
        print(line, end="")
    proc.stdout.close()
    returncode = proc.wait()
    if returncode != 0:
        print(f"Command failed with exit code {returncode}", file=sys.stderr)
        sys.exit(returncode)
