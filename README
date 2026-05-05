## ️ Setup & Data

1. **Environment**: Build or pull the Apptainer image.
   ```bash
   apptainer build my_env.sif my_env.def
   ```
2. **Dataset**: Place your raw dataset files inside the `/data/` directory.

##  Project Structure

```text
radio-map-prediction/
├── data/                   # Dataset storage
├── src/                    # Source code
│   ├── data/               # Data processing & Dataset class
│   └── models/             # Model architectures
├── train.py                # Training entry script
├── test.py                 # Evaluation script
├── env.def                 # Apptainer definition file
└── requirements.txt        # Python dependencies
```

##  Quick Start

**Local / WSL Debugging:**
```bash
pip install -r requirements.txt
python train.py
```

**Slurm Cluster Submission:**
```bash
sbatch job_script.sh
```

