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

