## AFT Diagnostic Plots - Issue Fixed

### Problem
The AFT diagnostic plots were showing errors for interaction models:
```
⚠️  Error with weibull_aft_interactions: "['context_to_prompt_drift_x_model_gemini_25', 'cumulative_drift_x_model_mistral_large', ...] not in index"
```

### Root Cause
The interaction models were fitted with datasets that included interaction terms (e.g., `prompt_to_prompt_drift_x_model_mistral_large`), but the diagnostic plotting function was trying to use the original `self.combined_data` which didn't contain these interaction columns.

### Solution Implemented
1. **Added dataset storage**: Modified `AFTModeling` class to store the specific dataset used for each model in `self.model_datasets`

2. **Updated all fitting methods**: Each AFT fitting method now stores its dataset:
   - Basic models: Store the base dataset with numeric and model columns
   - Interaction models: Store the enhanced dataset with interaction terms

3. **Fixed diagnostic plotting**: Updated `plot_aft_diagnostics()` to use the correct dataset for each model:
   ```python
   # Get the correct dataset for this model
   model_data = self.model_datasets.get(model_name, self.combined_data)
   ```

4. **Improved directory structure**: All AFT figures now save to `results/figures/aft/`

### Result
- ✅ All 6 AFT models (3 basic + 3 with interactions) now generate diagnostics without errors
- ✅ Diagnostic plot file size increased from 420KB to 697KB (more content)
- ✅ All interaction models successfully visualized
- ✅ Proper dataset handling for all prediction methods

### Files Modified
- `src/modeling/aft.py`: Added dataset storage and fixed diagnostic plotting
- All AFT visualization outputs now in `results/figures/aft/` directory

The AFT diagnostic plots now work correctly for all model types without any index errors!