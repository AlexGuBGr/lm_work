"""
Repo-hygiejne: afvis store filer og trænings-artefakter i git.

    python scripts/check_repo.py            # tjek alle sporede filer
    python scripts/check_repo.py --staged   # tjek kun filer i staging (til pre-commit hook)
"""
import fnmatch
import os
import subprocess
import sys

MAX_BYTES = 50 * 1024 * 1024
FORBIDDEN = ["*.bin", "*.pt", "*.pth", "*.safetensors", "*.ckpt", "data/*", "out/*", ".venv/*"]


def tracked_files(staged):
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=AM", "-z"] if staged \
        else ["git", "ls-files", "-z"]
    out = subprocess.run(cmd, check=True, capture_output=True).stdout.decode()
    return [f for f in out.split("\0") if f]


def main():
    problems = []
    for path in tracked_files("--staged" in sys.argv):
        if any(fnmatch.fnmatch(path, pat) for pat in FORBIDDEN):
            problems.append(f"{path}: data/checkpoints/miljø hører ikke til i git")
        if os.path.isfile(path) and os.path.getsize(path) > MAX_BYTES:
            problems.append(f"{path}: {os.path.getsize(path) / 1e6:.1f} MB > 50 MB")
    if problems:
        print("Repo-tjek fejlede:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("Repo-tjek OK")


if __name__ == "__main__":
    main()
