# Windows Setup Checklist

This checklist is only for the current bootstrap blockers. It does not scaffold Phase 1.

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

- `pyarrow`
- `pybind11`

## 3. Install CMake on Windows

Install CMake separately so the `cmake` CLI is available on `PATH`.

Example with `winget`:

```powershell
winget install Kitware.CMake
```

Verification:

```powershell
cmake --version
```

## 4. Quick dependency check

Run this from the project root:

```powershell
@'
import pyarrow
import pybind11
print("Python packages OK")
'@ | py -
```

## 5. Local config reminder

- Verify `.env` has no leading or trailing spaces after `EODHD_API_KEY=`.
