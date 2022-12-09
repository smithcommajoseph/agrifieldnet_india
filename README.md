# AgrifieldNet India 

The repo contains data prep, eda and modeling code for the [AgrifieldNet India Challenge](https://zindi.africa/competitions/agrifieldnet-india-challenge). 

This project includes five models, which achieve the following scores from the AgrifieldNet competition.

| Model | Public Score | Private Score |
| ----- | ------------ | ------------- |
| rf_p  | 1.744836782  | 2.163112415   |
| xgb_p | 1.758395241  | 1.765851258   |
| svm_p | 1.465262957  | 1.497996049   |
| mlp_p | 1.806023962  | 1.767157053   |
| v_p   | 1.309915641  | 1.372958035   |

The final model, a voting classifier, *would have ranked* **22nd** place according to the [leaderboard](https://zindi.africa/competitions/agrifieldnet-india-challenge/leaderboard) had these results been submitted before the end of the cutoff date as part of the competition.

## Prerequisites

- Python
- Pipenv

## Running the project

From within this directory run the following

- `pipenv shell`
- `pipenv install`

After that, you are ready to head to the `code/notebooks` directory and explore. 

*Note 1: I ran these notebooks inside VSCode, however, one could just as easily execute `jupyter` from the commandline and use the web interface.*

*Note 2: Geospatial processing will require `gdal` which is not currently included in the `Pipfile` due to compatibility issues with other packages.*
