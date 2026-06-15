# Diet x Microbiota Metabolomics Explorer

Streamlit dashboard for exploring cecal metabolomics data from diet and microbiota experiments from: 

Desai M, Seekatz A, Koropatkin N ...
A Dietary Fiber-Deprived Gut Microbiota Degrades the Colonic Mucus Barrier and Enhances Pathogen Susceptibility
Cell, 167, 1339-1353.e21

## What The App Does

- Loads the bundled `Mahesh_metabolomics_updated.xlsx` workbook by default.
- Supports the `GF_14SM` and `SPF_diet` sheets.
- Shows PCA, metabolite boxplots, heatmaps, volcano plots, microbiota effects, and diet effects.
- Uses `nmol/g` units when detected from the workbook.
- Drops SPF sample marked as `outlier` by publication.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy On Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to Streamlit Community Cloud.
3. Create a new app from the repository.
4. Set the main file path to:

```text
streamlit_app.py
```

## Data File

The app expects `Mahesh_metabolomics_updated.xlsx` to be present in the repository root for the default bundled dataset. Users can also upload another compatible Excel workbook from the sidebar.
