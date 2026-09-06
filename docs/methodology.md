# Draft methodology

1. Acquire and validate regional boundaries and all source datasets.
2. Reproject vectors and rasters to the configured analysis CRS.
3. Build annual or seasonal cloud-masked image composites.
4. Create consistent training and validation samples across regions and years.
5. Classify LULC using the definitions in `config/lulc_classes.yaml`.
6. Apply the minimum mapping unit and perform documented post-processing.
7. Assess map accuracy using independent samples and report class-wise metrics.
8. Cross-tabulate baseline and comparison rasters to obtain transition matrices.
9. Calculate hectares and percentages by region, with special attention to:
   forest to cocoa, cocoa to other classes, and net cocoa-area change.
10. Export reproducible tables, figures, maps, and a limitations statement.

Area estimates should be adjusted for classification error when the sampling
design supports statistically valid correction.

## Cocoa versus natural-tree classification (2017)

Dynamic World class `1` is used only as a tree-domain mask. Within that mask,
Sentinel-2 spectral features are used to distinguish cocoa (`2` in the final
map) from natural tree (`1`). Pixels outside the Dynamic World tree mask are
class `0`, and nodata is `255`.

The downloaded composite contains the ten bands needed by this workflow:
`B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12`. All reflectance values are divided
by 10,000 before calculating NDVI, EVI, NDRE, NDRE2, NDMI, NBR, GNDVI, SAVI,
RECI, and IRECI. Red-edge and SWIR bands delivered at 10 m by the downloader
were resampled by Earth Engine during export; this must be reported as an
effective 10 m analysis grid, not native 10 m information for every band.

Run the pipeline from the repository root:

```bash
python scripts/cocoa_classification/01_create_dw_tree_mask.py --year 2017
python scripts/cocoa_classification/02_calculate_spectral_indices.py --year 2017 --season djf
python scripts/cocoa_classification/02b_build_annual_feature_stack.py --year 2017
python scripts/cocoa_classification/03_prepare_training_samples.py --year 2017 --season annual
python scripts/cocoa_classification/04_train_models.py --year 2017 --season annual

python scripts/cocoa_classification/05_predict_tree_classes.py --year 2017 --season annual --model random_forest
python scripts/cocoa_classification/05_predict_tree_classes.py --year 2017 --season annual --model xgboost
python scripts/cocoa_classification/05_predict_tree_classes.py --year 2017 --season annual --model mlp

python scripts/cocoa_classification/06_mosaic_tree_classification.py --year 2017 --season annual --model random_forest
```

After mosaicking the selected XGBoost prediction, create the final LULC map and
area table with:

```bash
python scripts/cocoa_classification/07_create_final_lulc_products.py --year 2017
```

The final map retains Dynamic World codes 0–8, interprets code 1 as
`other_trees`, and uses the new code 9 for `cocoa_plantation`. Cocoa is assigned
only where a pixel is both Dynamic World tree and XGBoost cocoa. The GeoTIFF is
the analysis product; the PNG is a downsampled cartographic preview, and the CSV
reports pixel counts and areas in hectares and square kilometres.

The model comparison uses a spatial-group holdout rather than a random pixel
split, reducing overly optimistic accuracy caused by nearby, spatially
autocorrelated samples. Model selection should consider cocoa precision,
recall, F1, ROC AUC, and the confusion matrix—not accuracy alone.

### Reference-data limitation

The 500 cocoa points provide positive labels only. The preparation script
therefore draws a configurable number of Dynamic World tree pixels at least
1,000 m from known cocoa points and labels them `pseudo_natural`. Some unknown
cocoa farms can occur in that set, producing label noise. These pseudo-labels
are suitable for model development but do not provide an independent accuracy
assessment. Before publishing final area or change estimates, collect or
visually interpret representative natural-tree reference samples and reserve a
separate, probability-based sample for unbiased map validation and area-error
adjustment.

The 2017 cocoa points should not be reused as independent validation samples
after being used for training or threshold selection. For predictions in other
years, check temporal transfer explicitly or create year-specific reference
data; spectral relationships and plantation age can change over time.
