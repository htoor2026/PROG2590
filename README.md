# PROG2590 – Group 5 Income Classification

This folder contains our final solution for the Adult income classification project for PROG2590.

## How to run the solution

1. **Environment**

   - Python 3.10+  

2. **Data**

   - Training data: `Data/group5-adult.csv`.  
   - For the demo, please place the instructor’s hold‑out CSV into the `Data/` folder  
     (for example: `Data/holdout_test.csv`).

3. **Main workflow**

   The main workflow is implemented in Jupyter notebooks:

   1. `notebooks/02_preprocessing_pipeline.ipynb`  
      - Builds the preprocessing pipeline (imputation, scaling, one‑hot encoding).

   2. Final model notebook:  

      - `notebooks/04_model_RandomForest.ipynb`  

      Running all cells top to bottom will train and evaluate the model on our internal test set.

4. **Hold‑out test set (demo)**

   In `04_model_RandomForest.ipynb`, there is a cell labeled  
   **“HOLD‑OUT TEST CELL”**. To evaluate on the hold‑out data:


   After updating the path, running this cell will:

   - Load the hold‑out CSV  
   - Apply the same preprocessing pipeline  
   - Evaluate the final Random Forest model  
   - Print a classification report and ROC‑AUC

5. **Tests (optional)**

   To run the automated test:

   ```bash
   pytest
   ```

   This executes `tests/test_pipeline.py` to verify that the pipeline and model run without errors
