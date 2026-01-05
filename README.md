# Uncertainty Calibration in Evidential Deep Learning for Fault Diagnosis

Official implementation of  **Uncertainty Calibration in Evidential Deep Learning for Fault Diagnosis on Imbalanced Data**

---

## Abstract

This repository provides the implementation of our proposed uncertainty calibration methods for evidential deep learning (EDL) applied to fault diagnosis under imbalanced data scenarios.  
The proposed approaches improve the reliability of uncertainty estimation while maintaining strong diagnostic performance.

---

## Installation

Please install the required dependencies before running the code.

```bash
pip install -r requirements.txt
```
---

## Usage

Running the Code

After installing the required environment, run the main script:

```bash
python main.py
```
---

## Method Configuration

The training method can be selected by modifying the --method argument in main.py:
```bash
parser.add_argument(
    '--method',
    type=str,
    choices=['EDL', 'A-EDL', 'AP-EDL'],
    default='AP-EDL'
)
```
Supported methods: EDL, A-EDL, AP-EDL

To switch methods, only this parameter needs to be changed. No other configuration is required.
