To run the mixed-effects model in R:

```r
# R Script (run_mixed_model.R)

library(lme4)
df <- read.csv("data/synthetic_motor_data.csv")
model <- lmer(behavior_score ~ theta_power + spindle_rate + so_sp_coupling + aec_mean + session + (1|subject), data = df)
summary(model)
```

This mirrors the Python mixed model and can validate results cross-platform.
