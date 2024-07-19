from itertools import product
from pprint import pprint

import numpy as np
from darts.dataprocessing.transformers import MinTReconciliator, BottomUpReconciliator
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

## LEVEL 3: Greoup by reason - region - city/noncity
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
pred = model.predict(n=len(val))
pred_df = pred.pd_dataframe()


## PLOTS (BEFORE RECONCILIATION)
## NOTE: THE FORECAST TOTAL DOES NOT ADD UP!
plt.figure()
plt.plot(pred_df[regions].sum(1), label="Regions")
plt.plot(pred_df[reasons].sum(1), label="Reasons")
plt.plot(pred_df["Total"], label="Total")
plt.plot(val_df["Total"], label="True")

plt.legend()
plt.title("Forecasts (before reconciliation)")


## RECONCILE
## ENSURE COHERENCE (FOR TRUSTWORTHINESS) & IMPROVE ACCURACY

# reconciliator = MinTReconciliator(method="ols")
reconciliator = MinTReconciliator(method="wls_var")

train_pred = model.predict(n=len(train))
err_train = train - train_pred

# reconciliator.fit(train)
reconciliator.fit(err_train)

reconcilied_preds = reconciliator.transform(pred)
reconcilied_preds_df = reconcilied_preds.pd_dataframe()

plt.figure()
plt.plot(reconcilied_preds_df[regions].sum(1), label="Regions")
plt.plot(reconcilied_preds_df[reasons].sum(1), label="Reasons")
plt.plot(reconcilied_preds_df["Total"], label="Total")
plt.plot(val_df["Total"], label="True")

plt.legend()
plt.title("Forecasts (after reconciliation)")

## EVALUATE MAE
total_pred_recon = reconcilied_preds_df["Total"].values
total_pred_nonrecon = pred_df["Total"].values
total_true = val["Total"].pd_dataframe().values

print(f"MAE (- recon): {np.abs(total_true - total_pred_nonrecon).mean() : .4f}")
print(f"MAE (+ recon): {np.abs(total_true - total_pred_recon).mean() : .4f}")
