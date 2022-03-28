import pdb
import time
import argparse
import subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", "-d", type=Path, default="./data/epic_states", help="Path to Epic-States dataset")
ap.add_argument("--gpus", "-g", type=str, required=True, default="0", help="GPUs to use (comma separated)")
ap.add_argument("--model_path", "-p", type=Path, required=True, help="Path to model")
ap.add_argument("--script", "-s", type=str, default="evaluate.py", help="Name of Python script.")
ap.add_argument("--runs", "-r", type=int, default=5, help="Number of runs")
ap.add_argument("--output_dir", "-o", type=Path, default= "out", help="Path to output")
ap.add_argument("--percent", "-pe", type=str, default="100,50,25,12.5", help="Training data percentage")
ap.add_argument("--ftune", action='store_true', help="Rune fine tuned model")
ap.add_argument("--ensemble", action='store_true', help="Skip training of ensemble model")
ap.add_argument("--skip_regular", action='store_true', help="Skip training of regular model")
ap.add_argument("--novel", action='store_true', help="Run novel classes experiment")
ap.add_argument("--and_novel", action='store_true', help="Run novel classes in conjunction with regular data.")
ap.add_argument("--seeds", type=str, default='0', help="Seeds to run in conjunction after seed 0.")
args = ap.parse_args()

assert not (args.novel and args.and_novel), "Flags are mutually exclusive."

assert args.model_path.exists() or str(args.model_path).split("_")[0] in ["resnet", "resnet50"]

base_command = f"python {args.script} --model_path {args.model_path} --percent (PERCENT) --sample_opposite --log {args.output_dir} --start_seed 0 --runs {args.runs} --data_dir {args.data_dir} --gpu (GPU)"

percent = [float(x) for x in args.percent.split(",")]
reg_commands = [base_command.replace("(PERCENT)", str(x)) for x in percent]
base_commands = []
for c in reg_commands:
    if args.novel:
        c += " --novel"
    if not args.skip_regular:
        base_commands.append(c)
    if args.ensemble:
        base_commands.append(c + " --ensemble_res")
    if args.ftune:
        base_commands.append(c + " --ftune")

if args.seeds:
    starting_seed = args.model_path.parent.name.split('_sd')[-1]
    for seed in args.seeds.split(','):
        if seed != starting_seed:
            base_commands += [c.replace(f'_sd{starting_seed}', f'_sd{seed}') for c in base_commands]

if args.and_novel:
    base_commands += [f'{c} --novel' for c in base_commands]

print("="*45)
for c in base_commands:
    print(c)
print("="*45)

gpus = [int(x) for x in args.gpus.split(",")]
procs = []

if len(gpus) > 1:
    for i in range(0, len(base_commands), len(gpus)):
        end = min(i + len(gpus), len(base_commands))
        for j in range(i, end):
            command = base_commands[j].replace("(GPU)", str(gpus[j-i]))
            print(command)
            p = subprocess.Popen(command, shell=True)
            procs.append(p)
            if j != end - 1:
                time.sleep(5)
        for p in procs[::-1]:
            p.wait()
        procs.clear()

else:
    for base in base_commands:
        command = base.replace("(GPU)", str(gpus[0]))
        print(command)
        p = subprocess.Popen(command, shell=True)
        p.wait()

