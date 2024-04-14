
# Predicting Postoperative Delirium in Older Patients

This repository contains the code and resources for the study titled "Predicting Postoperative Delirium in Older Patients: a multicenter retrospective cohort study" by Wu et al. (2024). The study utilizes machine learning models to predict postoperative delirium (POD) using pre-and perioperative features, with a specific focus on neuropsychological assessments.


## Citation
If you utilize the provided algorithms in your research, please cite the following paper:
```
Wu, S. C. J., Sharma, N., Bauch, A., Yang, H. C., Hect, J. L., Thomas, C., ... & PAWEL Study Group. (2024). Predicting Postoperative Delirium in Older Patients: a multicenter retrospective cohort study. medRxiv, 2024-03.
```

## Repository Structure
- `code/`: Contains the Jupyter notebooks and Python scripts used for the analysis.
  - `delirium_study.ipynb`: Main Jupyter notebook with the analysis.
  - `functions/`: Helper functions used in the analysis.
    - `ML_functions.py`: Functions for machine learning operations.
    - `Plotting.py`: Functions for plotting results.
    - `utils.py`: Utility functions for data handling and manipulation.
- `data/`: Sample data files used in the study (full dataset available upon request).
- `intermediate_files/`: Intermediate files generated during analysis, including CSVs with ROC-AUC values.
- `results/`: Visualizations and summary statistics from the analyses, including SHAP values and ROC curves.

## Dataset
Due to privacy and legal constraints, the complete dataset supporting the conclusions of this article is not publicly available. Data can be requested directly from the PAWEL study group via a formal process. Please contact the corresponding authors for more information.


## Setup and Installation
To set up the project environment:
```bash
git clone https://github.com/nsharma3150/PAWEL-Delirium-Prediction
cd PAWEL-Delirium-Prediction
pip install -r requirements.txt  # Install dependencies
```

## Usage
To run the analysis:
```bash
jupyter notebook delirium_study.ipynb  # Open the Jupyter notebook
```

