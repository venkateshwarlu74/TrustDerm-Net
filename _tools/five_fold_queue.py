"""Finish the 5-fold cross-validation: run folds 3 and 4 of every model on Kaggle.

Commands, flags and seed are those of the fold 0-2 kernels; only --fold changes. At most two kernels run at a
time (Kaggle's GPU session limit). Student-type runs wait for the teacher. State is kept in
results/fivefold/queue_state.json, so re-running this script resumes where it stopped.

Usage: python _tools/five_fold_queue.py [--dry]
"""
import json, os, re, subprocess, sys, time
from pathlib import Path

os.environ["TD_NO_COMP"] = "1"          # ISIC 2020 rules not accepted; same as the fold 0-2 runs
sys.path.insert(0, str(Path(__file__).parent))
from kaggle_launch import KAGGLE, USER, push   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fivefold"
STATE = OUT / "queue_state.json"
FOLDS = (3, 4)
MAX_RUNNING, POLL = 2, 120
TEACHER = "trustderm-teacher-f3-4"
KEEP = r"(preds/|results/|run_log|ckpt/.*_log\.csv)"


def student_cmds(tag, flags):
    return [f"python train.py --role student --fold {k} --teacher_ckpt /kaggle/working/ckpt/teacher_fold{k}.pt"
            f" --tag {tag} {flags}".strip() for k in FOLDS] + [f"python evaluate.py --tag {tag}"]


def base_cmds(tag, backbone):
    return [f"python train.py --role student --fold {k} --tag {tag} --backbone {backbone}"
            " --kd none --no_meta --no_concepts --flat --softmax" for k in FOLDS] + [f"python evaluate.py --tag {tag}"]


JOBS = [dict(tag="teacher", name=TEACHER, deps=[], sources=["trustderm-prep"],
             cmds=[f"python train.py --role teacher --fold {k}" for k in FOLDS] + ["python evaluate.py --tag teacher"])]
for tag, flags in [("student", ""), ("abl_kdnone", "--kd none"), ("abl_softmax", "--softmax"),
                   ("abl_noconcepts", "--no_concepts"), ("abl_kdplain", "--kd plain"), ("abl_nometa", "--no_meta"),
                   ("abl_concat", "--fusion concat"), ("abl_pmd0", "--p_md 0"), ("abl_flat", "--flat")]:
    JOBS.append(dict(tag=tag, name=f"trustderm-{tag.replace('_', '-')}-f3-4", deps=[TEACHER],
                     sources=["trustderm-prep", TEACHER], cmds=student_cmds(tag, flags)))
for tag, backbone in [("base_swint", "swin_tiny_patch4_window7_224"), ("base_convnextt", "convnext_tiny"),
                      ("base_effb3", "efficientnet_b3"), ("base_resnet50", "resnet50"),
                      ("base_mnv2", "mobilenetv2_100"), ("base_deits", "deit_small_patch16_224")]:
    JOBS.append(dict(tag=tag, name=f"trustderm-{tag.replace('_', '-')}-f3-4", deps=[], sources=["trustderm-prep"],
                     cmds=base_cmds(tag, backbone)))


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(OUT / "queue.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def status(name):
    r = subprocess.run([KAGGLE, "kernels", "status", f"{USER}/{name}"], capture_output=True, text=True)
    m = re.search(r'status "(?:KernelWorkerStatus\.)?(\w+)"', r.stdout + r.stderr)
    return m.group(1).upper() if m else "UNKNOWN"


def fetch(job):
    pattern = r"(preds/|results/|run_log|ckpt/.*_log\.csv|ckpt/student_fold)" if job["tag"] == "student" else KEEP
    d = OUT / job["tag"]
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run([KAGGLE, "kernels", "output", f"{USER}/{job['name']}", "-p", str(d), "-o", "-q",
                    "--file-pattern", pattern], check=True)
    n = sum(1 for k in FOLDS if (d / "preds" / f"{job['tag']}_fold{k}.csv").exists())
    if n != len(FOLDS):
        raise RuntimeError(f"{job['tag']}: only {n} of {len(FOLDS)} prediction files downloaded")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if "--dry" in sys.argv:
        for j in JOBS:
            print(j["name"], j["sources"], *j["cmds"], "", sep="\n  ")
        return
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    save = lambda: STATE.write_text(json.dumps(st, indent=1))
    while True:
        for j in JOBS:
            if st.get(j["name"]) in ("running", "fetch"):
                s = status(j["name"]) if st[j["name"]] == "running" else "COMPLETE"
                if s == "COMPLETE":
                    st[j["name"]] = "fetch"; save()
                    try:
                        fetch(j)
                        st[j["name"]] = "done"; log(f"DONE {j['tag']}")
                    except Exception as e:      # keep "fetch": retried on the next poll
                        log(f"download problem {j['tag']}: {e}")
                elif s in ("ERROR", "CANCELACKNOWLEDGED", "CANCELREQUESTED"):
                    st[j["name"]] = "failed"; log(f"FAILED {j['tag']} ({s})")
                save()
        running = sum(v == "running" for v in st.values())
        for j in JOBS:
            if running >= MAX_RUNNING:
                break
            if j["name"] in st or any(st.get(d) != "done" for d in j["deps"]):
                continue
            try:
                push(j["name"], j["cmds"], gpu=True, sources=j["sources"])
                st[j["name"]] = "running"; running += 1; save()
                log(f"STARTED {j['tag']}")
                time.sleep(30)
            except subprocess.CalledProcessError as e:
                log(f"push refused for {j['tag']}: {e}")
                break
        todo = [j for j in JOBS if st.get(j["name"]) not in ("done", "failed")]
        blocked = [j for j in todo if any(st.get(d) == "failed" for d in j["deps"])]
        if len(todo) == len(blocked):
            log(f"QUEUE FINISHED  done={sum(v == 'done' for v in st.values())}  "
                f"failed={[k for k, v in st.items() if v == 'failed']}  blocked={[j['tag'] for j in blocked]}")
            return
        time.sleep(POLL)


if __name__ == "__main__":
    main()
