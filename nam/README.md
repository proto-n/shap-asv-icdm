# NAM experiments

## Setup

Please clone https://github.com/lemeln/nam to the "nam" repository under this folder and install its dependencies.

## Reproducing the NAM paper (https://neural-additive-models.github.io)

See `nam-reproduce.ipynb`. Figures show single-variable functions of the model.

![Age](out/recidivism_0.png)
![Race](out/recidivism_1.png)
![Gender](out/recidivism_2.png)
![Priors Count](out/recidivism_3.png)
![Length of Stay](out/recidivism_4.png)
![Charge Degree](out/recidivism_5.png)

There seems to be a very large randomness (see yellow lines) in the individual prediction functions. Dashed red lines signify 10 and 90 percentiles regarding the 20 trained functions. Blue is the average. (Same is visible in the original NAM paper).


## Single and double variable models

See `nam-asv.ipynb`. Figures show single-variable functions of models trained only on that single variable.

### Single variable models:

![Age](out/recidivism_solo_0.png)
![Race](out/recidivism_solo_1.png)
![Gender](out/recidivism_solo_2.png)
![Priors Count](out/recidivism_solo_3.png)
![Length of Stay](out/recidivism_solo_4.png)
![Charge Degree](out/recidivism_solo_5.png)

As we can say, these don't work very well. Looks like the probabilities are not well-calibrated, even if they give an ok AUC score. This is confirmed at the end of the notebook (`nam-asv.ipynb`) where we measure logit variances.

### Double variable models:

One line is two single-variable functions of the same model, trained on only those two variables.

![pic](out/recidivism_double_0.png)
![pic](out/recidivism_double_1.png)
![pic](out/recidivism_double_2.png)
![pic](out/recidivism_double_3.png)
![pic](out/recidivism_double_4.png)
![pic](out/recidivism_double_5.png)
![pic](out/recidivism_double_6.png)
![pic](out/recidivism_double_7.png)
![pic](out/recidivism_double_8.png)
![pic](out/recidivism_double_9.png)
![pic](out/recidivism_double_10.png)
![pic](out/recidivism_double_11.png)
![pic](out/recidivism_double_12.png)
![pic](out/recidivism_double_13.png)
![pic](out/recidivism_double_14.png)

# Interactions

The code will be uploaded by 2023-07-03 12:00 UTC at the latest.