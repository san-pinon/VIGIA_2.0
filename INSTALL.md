# Installation guide — VIGIA 2.0

Clone the repository and install `multicam` as an editable install.
With an editable install, `git pull` is enough to pick up code changes — no reinstall needed.

---

## Python version

Both field nodes run **Python 3.11**. Using the same version on both avoids subtle
incompatibilities when sharing data or scripts across nodes.

| Node | Hostname | IP | Python |
|------|----------|----|--------|
| RPi5 | `uvcam` | `192.168.30.101` | 3.11 (system) |
| SBC | `multicam` | `192.168.30.150` | 3.11 (system) |

> **Note:** `pyproject.toml` requires `>=3.11`. Python 3.12 is used on the dev machine
> (see dev section below) but is not required on the field nodes.

---

## RPi5 (hostname: `uvcam`, IP: `192.168.30.101`)

`picamera2` depends on system-level libcamera bindings that cannot be installed via pip.
The venv **must** be created with `--system-site-packages` to inherit them.

```bash
# 1. Clone
cd /home/user
git clone https://github.com/san-pinon/VIGIA_2.0.git multicam

# 2. Create venv — system-site-packages required for picamera2
uv venv --python=3.11 --system-site-packages /home/user/.venv

# 3. Activate
source /home/user/.venv/bin/activate

# 4. Editable install
cd /home/user/multicam
uv pip install -e .

# 5. Copy reference config and edit for this node
cp multicam_config.toml ~/.config/multicam_config.toml

# 6. Verify
multicamctl --help
```

---

## OnLogic SBC (hostname: `multicam`, IP: `192.168.30.150`)

```bash
# 1. Clone
cd /home/user
git clone https://github.com/san-pinon/VIGIA_2.0.git multicam

# 2. Create venv
uv venv --python=3.11 /home/user/.venv

# 3. Activate
source /home/user/.venv/bin/activate

# 4. Editable install
cd /home/user/multicam
uv pip install -e .

# 5. Copy reference config and edit for this node
cp multicam_config.toml ~/.config/multicam_config.toml

# 6. Verify
multicamctl --help
```

---

## Updating after a `git pull`

Because the install is editable, pulling is enough:

```bash
cd /home/user/multicam
git pull
```

Reinstall only if `pyproject.toml` changed (new dependencies were added):

```bash
git pull
uv pip install -e .
```

---

## Systemd units

After cloning, install the capture and file-monitor services on each node:

```bash
sudo cp /home/user/multicam/systemd-units/*.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
```

**RPi5** — enable:
```bash
sudo systemctl enable --now capture-uv.timer capture-spec.timer capture-env.timer file-monitor.service
```

**SBC** — enable:
```bash
sudo systemctl enable --now capture-ir.timer capture-vis.timer file-monitor.service
```

> Update `ExecStart` in `file-monitor.service` to match the actual clone path
> (`/home/user/multicam/scripts/file_monitor.py`) before enabling.

---

## Dev machine

```bash
git clone https://github.com/san-pinon/VIGIA_2.0.git
cd VIGIA_2.0
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e ".[dev]"   # includes ruff and ipython

# Lint / format
ruff check multicam/
ruff format multicam/
```
