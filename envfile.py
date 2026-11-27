"""Load KEY=value lines from the git-ignored .env next to this file into the environment.
Values already set in the shell win; empty values are ignored."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path=os.path.join(HERE, ".env")):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if value and key not in os.environ:
                os.environ[key] = value


load()
