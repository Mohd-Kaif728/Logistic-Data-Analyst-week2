
from pathlib import Path
import json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUT=Path("week4_outputs"); OUT.mkdir(exist_ok=True)
rng=np.random.default_rng(21); n=1000
distance=rng.uniform(10,900,n); volume=rng.uniform(.5,35,n)
planned=8+distance/55+volume*.16+rng.normal(0,1,n)
weather=rng.gamma(1.4,.8,n); congestion=rng.gamma(2,.7,n)
mode=rng.choice(["Road","Rail","Air"],n,p=[.62,.23,.15])
region=rng.choice(["North","South","East","West"],n)
mf=pd.Series(mode).map({"Road":1.,"Rail":1.22,"Air":.52}).to_numpy()
rf=pd.Series(region).map({"North":.3,"South":.6,"East":.8,"West":.2}).to_numpy()
actual=np.maximum(1,planned+weather*1.15+congestion*1.3+(distance/250)*mf+rf+rng.normal(0,1.15,n))
cost=45+distance*.72+volume*4.1+pd.Series(mode).map({"Road":1.,"Rail":.78,"Air":2.3}).to_numpy()*22+rng.normal(0,18,n)
df=pd.DataFrame({"distance_km":distance.round(1),"shipment_volume_m3":volume.round(2),
"planned_delivery_hours":planned.round(2),"weather_delay_hours":weather.round(2),
"congestion_delay_hours":congestion.round(2),"transport_mode":mode,"region":region,
"transport_cost":cost.round(2),"actual_delivery_hours":actual.round(2)})
df.to_csv(OUT/"hypothetical_logistics_dataset.csv",index=False)
num=["distance_km","shipment_volume_m3","planned_delivery_hours","weather_delay_hours","congestion_delay_hours"]
cat=["transport_mode","region"]; features=num+cat; X=df[features]; y=df.actual_delivery_hours
pre=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
cv=KFold(n_splits=5,shuffle=True,random_state=42)
models={"Linear Regression":Pipeline([("pre",pre),("model",LinearRegression())]),
"Random Forest":Pipeline([("pre",pre),("model",RandomForestRegressor(n_estimators=150,max_depth=12,min_samples_leaf=2,random_state=42,n_jobs=-1))])}
rows=[]
for name,m in models.items():
    scores=-cross_val_score(m,Xtr,ytr,cv=cv,scoring="neg_mean_absolute_error")
    m.fit(Xtr,ytr); p=m.predict(Xte)
    rows.append({"Model":name,"CV MAE mean (hours)":scores.mean(),"CV MAE std (hours)":scores.std(),
    "Test MAE (hours)":mean_absolute_error(yte,p),"Test RMSE (hours)":mean_squared_error(yte,p)**.5,"Test R-squared":r2_score(yte,p)})
res=pd.DataFrame(rows); res.to_csv(OUT/"model_evaluation_results.csv",index=False)
tuner=GridSearchCV(Pipeline([("pre",pre),("model",RandomForestRegressor(random_state=42,n_jobs=-1))]),
{"model__n_estimators":[100,150],"model__max_depth":[8,12,None],"model__min_samples_leaf":[1,2,4]},
scoring="neg_mean_absolute_error",cv=cv,n_jobs=-1)
tuner.fit(Xtr,ytr); p=tuner.predict(Xte)
tm={"best_params":tuner.best_params_,"cv_mae":-tuner.best_score_,"test_mae":mean_absolute_error(yte,p),
"test_rmse":mean_squared_error(yte,p)**.5,"test_r2":r2_score(yte,p)}
json.dump(tm,open(OUT/"tuned_random_forest_results.json","w"),indent=2)
plt.figure(figsize=(7,4.5)); plt.scatter(yte,p,alpha=.65); lim=[min(yte.min(),p.min()),max(yte.max(),p.max())]
plt.plot(lim,lim,"--"); plt.xlabel("Actual delivery time (hours)"); plt.ylabel("Predicted delivery time (hours)")
plt.title("Actual vs Predicted Delivery Time"); plt.tight_layout(); plt.savefig(OUT/"actual_vs_predicted.png",dpi=150); plt.close()
plt.figure(figsize=(7,4.5)); plt.hist(yte.to_numpy()-p,bins=25,edgecolor="black")
plt.xlabel("Prediction error (actual − predicted, hours)"); plt.ylabel("Shipments")
plt.title("Prediction Error Distribution"); plt.tight_layout(); plt.savefig(OUT/"prediction_error_distribution.png",dpi=150); plt.close()
print(res.round(3).to_string(index=False)); print("Tuned RF:",tm)
