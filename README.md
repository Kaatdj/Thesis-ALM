# Thesis-ALM


## Setup

### 1. Create environment (first time only)

**Mac / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (cmd):**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

---

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Daily usage

### Activate environment

**Mac / Linux:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (cmd):**

```bat
.venv\Scripts\activate.bat
```

---

### (Optional) update dependencies after pull

```bash
python -m pip install -r requirements.txt
```

---

## GPU (only for GPU users)

Install PyTorch separately from https://pytorch.org before installing requirements.
