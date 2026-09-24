# LULC transition analysis: 2017–2025

The workflow compares the final, aligned Dynamic World + XGBoost LULC maps. It
creates pixel-level transition and change-type rasters; overall, regional,
cocoa-specific, and four-group transition tables; class gain/loss/persistence
statistics; annualized rates; Sankey diagrams; heatmaps; charts; and a change
map.

Run from the project root with the virtual environment active:

```bash
python scripts/transition/01_calculate_lulc_transitions.py \
  --start-year 2017 \
  --end-year 2025

python scripts/transition/02_visualize_lulc_transitions.py \
  --start-year 2017 \
  --end-year 2025
```

Outputs are written to `outputs/transitions/2017_2025/`. Add `--overwrite` to
the first command when deliberately replacing existing transition rasters.

Only pixels with a valid class in both maps enter the transition statistics.
`input_area_consistency.csv` compares this common footprint with each original
class-area CSV, making exclusions caused by year-specific NoData explicit.

Transition raster codes use `start_class * 10 + end_class`. The change-type
raster uses 0 for persistence, 1 for cocoa gain, 2 for cocoa loss, 3 for other
class changes, and 255 for NoData.
