# Satellite Imagery-Based Property Valuation

## Project Overview
This project develops a **Multimodal Regression Pipeline** to predict residential property values.  
It integrates traditional **tabular features** (e.g., square footage, grade, location) with **high-dimensional visual embeddings** extracted from satellite imagery via the **Sentinel Hub API**.

The core objective is to evaluate whether **environmental context**—such as greenery, neighborhood density, and road infrastructure—provides a measurable predictive lift over standard structural attributes alone.

---

## Repository Structure & Workflow
To replicate the results of this study, please execute the files in the following logical order:

### Data Acquisition
- **`data_fetcher.py`**  
  Programmatic script to download satellite images for training and testing sets using property coordinates.  
  Requires valid **Sentinel Hub API credentials**.

---

### Modeling & Experiments
- **`preprocessing_(5_key_features)_model_training.ipynb`**  
  Ablation study using a restricted set of **5 tabular features** to isolate the marginal impact of satellite imagery under sparse information conditions.

- **`preprocessing_(9_extended_features)_model_training.ipynb`** **[PRIMARY MODEL]**  
  Best-performing pipeline using **9 extended tabular features**, combined with image embeddings.  
  Hyperparameters are optimized using **Bayesian Optimization (Optuna)**.  
  This notebook generates the final **`test_prediction.csv`** file.

---

### Analysis & Explainability
- **`neural_network_grad_cam.ipynb`**  
  Implementation of a **late-fusion Multimodal Neural Network**.  
  Includes **Grad-CAM visual explanations** to highlight which spatial regions (e.g., roads, green areas) influence property valuation.

---

### Final Deliverables
- **`test_prediction.csv`**  
  Final output file containing predicted property prices for the test dataset.

- **`Report.pdf`**  
  Complete project documentation including:
  - Overview 
  - EDA
  - Financial / Visual Insight
  - Architecture Diagram
  - Results

---

## Setup & Installation

### Clone the Repository
```bash
git clone https://github.com/Awani-Soni/Satellite-Imagery-Based-Property-Valuation-.git

### Install Required Packages
```bash
pip install pandas numpy xgboost tensorflow opencv-python matplotlib seaborn sentinelhub optuna shap
```
---

### Environment Note  
The notebooks are optimized for Google Colab.  
If running locally, please update the following variables at the top of each notebook to match your local directory structure:

- BASE_PATH  
- DRIVE_ZIP_PATH  

---

### Key Findings  

- #### Best Model  
Optimized XGBoost (9 features) achieved an R² score of 0.8884.

- #### Impact of Satellite Imagery  
Satellite imagery provides a consistent performance boost when tabular data is limited.  
However, its marginal contribution decreases when precise latitude/longitude and neighborhood density features are included.

- #### Model Explainability  
Grad-CAM visualizations confirm that the neural network prioritizes:

- Green spaces  
- Road connectivity  
- Surrounding built density  

---

## Author  

Awani Soni  
BS-MS Economics  
Indian Institute of Technology Roorkee
