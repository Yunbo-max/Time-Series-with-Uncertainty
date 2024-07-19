from itertools import product
from pprint import pprint

import numpy as np
from darts.dataprocessing.transformers import MinTReconciliator, BottomUpReconciliator
from darts.dataprocessing.transformers.reconciliation import _get_summation_matrix
from darts.datasets import AustralianTourismDataset
from darts.models import LinearRegressionModel
from matplotlib import pyplot as plt


tourism_series = AustralianTourismDataset().load()

sum_city_noncity = (
    tourism_series["NSW - hol - city"] + tourism_series["NSW - hol - noncity"]
)

## HIERARCHICAL LABELS
reasons = ["Hol", "VFR", "Bus", "Oth"]
regions = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT"]
city_labels = ["city", "noncity"]

## HIERARCHICAL DICT
hierarchy = dict()

## LEVEL 1: Fill in grouping by overall reason & regions
for reason in reasons:
    hierarchy.update({reason: ["Total"]})
for region in regions:
    hierarchy.update({region: ["Total"]})

## LEVEL 2: Group by reason - region
for reason, region in product(reasons, regions):
    hierarchy.update({f"{region} - {reason.lower()}": [region, reason]})

## LEVEL 3: Group by reason - region - city/noncity
for reason, region, city_label in product(reasons, regions, city_labels):
    hierarchy.update(
        {
            f"{region} - {reason.lower()} - {city_label}": [
                f"{region} - {reason.lower()}"
            ]
        }
    )

pprint(hierarchy)

## INSTALL HIERCHICAL INFO
tourism_hierarchical = tourism_series.with_hierarchy(hierarchy)

## TRAIN SPLIT
train, val = tourism_hierarchical[:-12], tourism_hierarchical[-12:]
val_df = val.pd_dataframe()

## FORECASTS (AR)
model = LinearRegressionModel(lags=7)
model.fit(train)

train_pred = model.predict(n=len(train))
pred = model.predict(n=len(val))

pred_df = pred.pd_dataframe()


## PLOTS (BEFORE RECONCILIATION)
## NOTE: THE FORECAST TOTAL DOES NOT ADD UP!
plt.figure()
plt.plot(pred_df[regions].sum(1), label="Regions")
plt.plot(pred_df[reasons].sum(1), label="Reasons")
plt.plot(pred_df["Total"], label="Total")

plt.legend()
plt.title("Forecasts (before MINT reconciliation)")


## RECONCILE
## ENSURE COHERENCE (FOR TRUSTWORTHINESS) & IMPROVE ACCURACY
err_train = train - train_pred

reconciliator = MinTReconciliator(method="wls_var")
reconciliator.fit(err_train)
reconcilied_preds = reconciliator.transform(pred)
reconcilied_preds_df = reconcilied_preds.pd_dataframe()

plt.figure()
plt.plot(reconcilied_preds_df[regions].sum(1), label="Regions")
plt.plot(reconcilied_preds_df[reasons].sum(1), label="Reasons")
plt.plot(reconcilied_preds_df["Total"], label="Total")


plt.legend()
plt.title("Forecasts (after MINT reconciliation)")

## EVALUATE MAE
total_pred_recon = reconcilied_preds_df["Total"].values
total_pred_nonrecon = pred_df["Total"].values
total_true = val["Total"].pd_dataframe().values

print(f"MAE (- recon): {np.abs(total_true - total_pred_nonrecon).mean() : .4f}")
print(f"MAE (+ recon): {np.abs(total_true - total_pred_recon).mean() : .4f}")


##===TESTBED===
S = _get_summation_matrix(train)  ## shape = (All, Base)

## Step 1: Estimate Wh, the covariance matrix of the corresponding base errors
Wh = np.diag(
    (err_train.values() ** 2).mean(axis=0)
)  ## temporal average of the variance of the forecasting residuals.

## Step 2: Inverse matrix
Wh_inv = np.linalg.inv(Wh)
G = np.linalg.inv(S.T @ Wh_inv @ S) @ S.T @ Wh_inv  ## shape = (Base, All)

## Step 3:
y_pred_recon = pred.with_values(S @ G @ pred.all_values())

print(
    f"MAE (+ recon): {np.abs(y_pred_recon.pd_dataframe()['Total'].values - total_true).mean() : .4f}"
)
