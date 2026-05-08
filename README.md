# Regression-by-using-pycharm-CANARA-BANK
import pandas as pd, statsmodels.api as sm, matplotlib.pyplot as plt
df = pd.read_csv(r"D:\canaraBS.csv")
df.columns = df.columns.str.strip()
fa = [c for c in df.columns if "fixed" in c.lower() and "asset" in c.lower()][0]
y = df[fa]
X = df.select_dtypes("number").drop(columns=[fa], errors="ignore")
X = X.drop(columns=[c for c in X.columns if "other" in c.lower() and "liab" in c.lower()], errors="ignore")
model = sm.OLS(y, sm.add_constant(X), missing="drop").fit()
print(model.summary())
coef = model.params.drop("const", errors="ignore")
plt.bar(coef.index, coef.values)
plt.xticks(rotation=45)
plt.title(" VANSHAJ MISHRA Regression Coefficients")
plt.tight_layout()
plt.show()
