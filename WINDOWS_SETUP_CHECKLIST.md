# Windows Setup Checklist

This checklist is only for the current bootstrap blockers. It does not scaffold Phase 1.

Current status note:

- This file is infrastructure-only.
- It does not imply that the project currently has an active validated strategy or that live/paper trading should start.

## 1. Install Python on Windows

- Install Python 3.12 x64.
- During install, enable `Add python.exe to PATH`.
- Open a new PowerShell window after installation.

Recommended verification:

```powershell
py --version
python --version
py -m pip --version
```

## 2. Install the Python packages in `requirements.txt`

From the project root:

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

Current `requirements.txt` covers:

- `numpy`
- `pandas`
- `requests`
- `pyarrow`

## 3. Quick dependency check

Run this from the project root:

```powershell
@'
import pyarrow
import pandas
import numpy
print("Python packages OK")
'@ | py -
```

## 5. Local config reminder

- Verify `.env` has no leading or trailing spaces after `EODHD_API_KEY=`.
