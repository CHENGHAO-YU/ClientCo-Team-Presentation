<<<<<<< HEAD
# Barley Yield Projections under Climate Change in France (1982–2050)

## Overview

This project estimates future barley yields for five French departments under three climate scenarios (SSP1-2.6, SSP2-4.5, SSP5-8.5) using statistical–machine learning methods. A random forest model is trained on historical climate and yield data, then applied to CMIP6 climate projections to generate yield forecasts through 2050.

## Data

| Dataset | Source | Period | Resolution |
|---------|--------|--------|------------|
| Barley yield (t/ha) | Agreste (French agricultural statistics) | 1982–2018 | Department-level |
| Historical climate | CMIP6 multi-model ensemble | 1982–2014 | Department-level averages |
| Future climate | CMIP6 SSP1-2.6 / SSP2-4.5 / SSP5-8.5 | 2015–2050 | Department-level averages |

## Key Results

- All departments show a **positive yield trend** driven primarily by the `year` variable (technological progress proxy).
- Climate variables modulate inter-annual variability: higher `gs_temp_max` generally reduces yield, while moderate precipitation increases it.
- Under **SSP5-8.5**, projected yields plateau or slightly decline toward 2050 in southern departments (Haute-Garonne, Isère), suggesting heat stress may offset technological gains.
- Under **SSP1-2.6**, yields continue rising steadily across all departments.
=======
# ClientCo-Team-Presentation
Leveraging Client's climate and Barley's production information to drive the company's growth
>>>>>>> e5709ff0e3f84a4a347d30f0767aad2364d3d572
