"""Write shell-compatible, workspace-local scratch and uv cache settings."""
import pathlib
import re
import shlex
import sys

root = pathlib.Path(sys.argv[1]).resolve()
env_file = root / ".env"
lines = env_file.read_text().splitlines() if env_file.exists() else []
lines = [line for line in lines if not re.match(r"^\s*(?:export\s+)?(?:TMPDIR|UV_CACHE_DIR)\s*=", line)]
for key, name in (("TMPDIR", "tmp"), ("UV_CACHE_DIR", "uv-cache")):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    lines.append(f"export {key}={shlex.quote(str(directory))}")
env_file.touch(mode=0o600, exist_ok=True)
env_file.chmod(0o600)
env_file.write_text("\n".join(lines) + "\n")
