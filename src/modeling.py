# src/modeling.py
import statsmodels.formula.api as smf

def run_mixed_model(df):
    model = smf.mixedlm("behavior ~ day + SP_density + beta_rest + theta_power + SO_SP_coupling", df, groups=df["subject"])
    result = model.fit()
    return result
