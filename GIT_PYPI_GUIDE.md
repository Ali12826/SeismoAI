# Git + PyPI Publishing Guide — seismoai

Complete step-by-step commands for a two-person team to push to GitHub
and publish both packages to PyPI.

---

## PART 1 — GitHub Setup (Person A does this once)

### Step 1 — Create the repository on GitHub
1. Go to https://github.com/new
2. Repository name: `seismoai`
3. Visibility: Public (required for free PyPI)
4. Do NOT tick "Add a README" (we already have one)
5. Click **Create repository**

### Step 2 — Initialize git locally and push (Person A)
```bash
cd seismoai/                      # your project root

git init
git add .
git commit -m "feat: initial project structure with seismoai_io and seismoai_viz"

git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/seismoai.git
git push -u origin main
```

---

## PART 2 — Person B joins the repository

```bash
# Person B: clone and set up
git clone https://github.com/YOUR_USERNAME/seismoai.git
cd seismoai
pip install -e ".[dev]"
```

---

## PART 3 — Collaborative workflow (both persons)

### Daily workflow
```bash
# Always pull before you start working
git pull origin main

# Create a feature branch for your work
git checkout -b feat/your-feature-name

# ... write code ...

git add seismoai_io/loader.py             # stage specific files
git commit -m "feat(io): add batch loading progress bar"

# Push your branch
git push origin feat/your-feature-name
```

### Merge via Pull Request
1. Go to your repo on GitHub
2. Click **Compare & pull request** on your branch
3. Add your partner as a reviewer
4. Merge after approval

### Both students must commit — verify this:
```bash
git log --oneline --all --author="Person A Name"
git log --oneline --all --author="Person B Name"
```

---

## PART 4 — PyPI Registration

### Step 1 — Create a PyPI account
Go to https://pypi.org/account/register/ and create accounts for both students.

### Step 2 — Create an API token
1. Go to https://pypi.org/manage/account/#api-tokens
2. Click **Add API token**
3. Scope: **Entire account** (for first upload; switch to per-project later)
4. Copy the token — it starts with `pypi-`

### Step 3 — Store the token (do this on your machine)
```bash
# Option A: store in ~/.pypirc  (easiest)
cat > ~/.pypirc << 'EOF'
[distutils]
index-servers = pypi

[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE
EOF
chmod 600 ~/.pypirc

# Option B: use environment variable (CI-friendly)
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR_TOKEN_HERE
```

---

## PART 5 — Build and publish seismoai-io

```bash
# Install build tools
pip install build twine

# Navigate to project root (where pyproject.toml lives)
cd seismoai/

# Build the distribution archives
python -m build
# This creates:
#   dist/seismoai_io-0.1.0-py3-none-any.whl
#   dist/seismoai_io-0.1.0.tar.gz

# (Optional) Test on TestPyPI first — highly recommended
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ seismoai-io

# Upload to real PyPI
twine upload dist/*
```

---

## PART 6 — Build and publish seismoai-viz

```bash
# Swap the pyproject.toml so setuptools finds seismoai_viz
cp pyproject.toml pyproject_io_backup.toml
cp pyproject_viz.toml pyproject.toml

# Clean old dist artifacts
rm -rf dist/ build/ *.egg-info seismoai_io.egg-info seismoai_viz.egg-info

# Build
python -m build
# Creates:
#   dist/seismoai_viz-0.1.0-py3-none-any.whl
#   dist/seismoai_viz-0.1.0.tar.gz

# Upload
twine upload dist/*

# Restore io pyproject for development
cp pyproject_io_backup.toml pyproject.toml
```

---

## PART 7 — Verify installation works

```bash
# In a fresh virtual environment
python -m venv verify_env
source verify_env/bin/activate          # Windows: verify_env\Scripts\activate

pip install seismoai-io seismoai-viz

python - << 'EOF'
from seismoai_io import load_sgy, normalize_traces
from seismoai_viz import plot_gather, plot_trace, plot_spectrum
print("✓ Both packages installed and importable.")
EOF
```

---

## PART 8 — Bump version for future releases

Edit `pyproject.toml`:
```toml
version = "0.1.1"   # patch fix
# or
version = "0.2.0"   # new feature
```

Tag the release in git:
```bash
git tag v0.1.1
git push origin v0.1.1
```

Then rebuild and re-upload (PyPI does not allow overwriting existing versions).

---

## Quick-reference checklist

- [ ] Both students have commits: `git shortlog -sn`
- [ ] All tests pass: `pytest tests/ -v`
- [ ] README describes all 6 functions
- [ ] `pip install seismoai-io` works from PyPI
- [ ] `pip install seismoai-viz` works from PyPI
- [ ] Reflection (5 sentences) added to README or submitted separately
