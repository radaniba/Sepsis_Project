---
title: "Neonatal mouse scoring system statistics"
author: "Danny Harbeson"
date: "April 16, 2019"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
knitr::opts_knit$set(root.dir = "C:/Users/danny/Documents/Kollmann lab/plos_predictor_paper/Revised_data")
```

```{r libraries, warning=FALSE, results=FALSE, message=FALSE}
library(tidyverse)
library(survival)
library(survminer)
library(rmarkdown)


```


## Data import and cleaning
All statistical analyses performed here begin from the files which can be found on the Open Science Framework. The five files are imported as df, df2, df3, df4, and df5 [shown below]
```{r Data import}
df = read.csv("mouse_mortality_data.csv", stringsAsFactors = TRUE) 
df2 = read.csv("mouse_CFU_data.csv", stringsAsFactors = TRUE)
df3 = read.csv("classifier_validation_cohort.csv", stringsAsFactors = TRUE) 
df4 = read.csv("classifier_building_data.csv", stringsAsFactors = TRUE) 
df5 = read.csv("sham_challenge_control_set.csv", stringsAsFactors = TRUE)
```

Some cleaning needs to be done prior to analysis on the 'mouse_mortality_data' file, as this is where data was directly entered. That cleaning is shown below.
```{r mortality_csv cleanup}
# For mice which recorded the same score on both sides often only one score was entered (into the high column). The following lines impute the high score if the lower score is not entered.
df$v1.score.low <- ifelse(is.na(df$v1.score.low), df$v1.score.high, df$v1.score.low)
df$v2.score.low <- ifelse(is.na(df$v2.score.low), df$v2.score.high, df$v2.score.low)
df$v3.score.low <- ifelse(is.na(df$v3.score.low), df$v3.score.high, df$v3.score.low)
df$v4.score.low <- ifelse(is.na(df$v4.score.low), df$v4.score.high, df$v4.score.low)
df$v5.score.low <- ifelse(is.na(df$v5.score.low), df$v5.score.high, df$v5.score.low)
df$v6.score.low <- ifelse(is.na(df$v6.score.low), df$v6.score.high, df$v6.score.low)
# The 18 HPC monitoring time point could have been the first, second or third recorded monitoring time point. Similarly, the 24-hour monitoring time point came either at the second, third, or fourth recorded monitoring time point. The following lines identify which monitoring time point is associated with 18 and 24 HPC for each mouse by filtering to timepoints within +/- 3 hours of 18 and 24 hours.
df$v18_timepoint = ifelse(is.na(df$v1.hour.post.challenge) == FALSE & 
                            abs(df$v1.hour.post.challenge - 18) <= 3, "v1", 
                          ifelse(is.na(df$v2.hour.post.challenge) == FALSE & 
                            abs(df$v2.hour.post.challenge - 18) <= 3,"v2", 
                            ifelse(is.na(df$v3.hour.post.challenge) == FALSE & 
                            abs(df$v3.hour.post.challenge - 18) <= 3, "v3", NA)))
# Same for v24
df$v24_timepoint = ifelse(is.na(df$v2.hour.post.challenge) == FALSE & 
                            abs(df$v2.hour.post.challenge - 24) <= 3, "v2", 
                           ifelse(is.na(df$v3.hour.post.challenge) == FALSE & 
                            abs(df$v3.hour.post.challenge - 24) <= 3, "v3", 
                            ifelse(is.na(df$v4.hour.post.challenge) == FALSE & 
                            abs(df$v4.hour.post.challenge - 24) <= 3, "v4", 
                            NA)))
df$v24_timepoint = ifelse(is.na(df$v24_timepoint), "v3", df$v24_timepoint) # 4 mice were monitored at 27.33 hours at visit 3, this is to include those
# Next few lines create a column with the score at 18 hours, pulled from their respective time points
df$v18_score = NA
for(i in 1:nrow(df)) {
  df[i,]$v18_score = parse(text=paste("df[", i, ",]$", df[i,]$v18_timepoint, ".score.low", sep = ""))
  }
df$v18_score = sapply(df$v18_score, eval)
# Same for score at 24 hours
df$v24_score = NA
for(i in 1:nrow(df)) {
  df[i,]$v24_score = parse(text=paste("df[", i, ",]$", df[i,]$v24_timepoint, ".score.low", sep = ""))
  }
df$v24_score = sapply(df$v24_score, eval)
# If the score is greater than or equal to 3 the mouse righted, if less than 3 it failed to right
df$v18_righting = ifelse(df$v18_score >= 3, "Rights", "FTR")
df$v24_righting = ifelse(df$v24_score >= 3, "Rights", "FTR")
# Outcome needs to be shown as a numeric binary choice, 0 = survive, 1 = die
df$outcome_binary = ifelse(df$outcome == "live", 0, 1) 
```

## Statistical analyses - Main figures
### Figure 1 - Righting reflex and outcome
To compare survival curves shown in Figure 1B, a log-rank test is performed using the surv_pvalue() function from the package 'survminer'.
```{r Fig 1B stats}
table(df$v24_righting)
sv = surv_fit(Surv(time.alive, outcome_binary) ~ v24_righting, data = df)
surv_pvalue(sv)
```

Bacterial load data were tested for normality using the Shapiro-Wilk method after log transformation
```{r normality}
shapiro.test(log10(df2$blood.cfu.per.ml+1)) # blood
shapiro.test(log10(df2$spleen.cfu.per.g.tissue+1)) # spleen
shapiro.test(log10(df2$liver.cfu.per.g.tissue+1)) # liver
shapiro.test(log10(df2$lung.cfu.per.g.tissue+1)) # lung
```
P-values < 0.05 indicate the data does not follow a normal distribution.

Simple numeric objects containing the values of bacterial load (in CFU) are created for ease of interpretation
```{r example}
blood_rights = df2[df2$v24.righting.response.low == "rights",]$blood.cfu.per.ml # CFU in blood of mice which were able to right at 24 HPC
blood_ftr = df2[df2$v24.righting.response.low == "ftr",]$blood.cfu.per.ml # CFU in blood of mice which were unable to right at 24 HPC
spleen_rights = df2[df2$v24.righting.response.low == "rights",]$spleen.cfu.per.g.tissue # CFU in spleen of mice which were able to right at 24 HPC
spleen_ftr = df2[df2$v24.righting.response.low == "ftr",]$spleen.cfu.per.g.tissue # CFU in spleen of mice which were unable to right at 24 HPC
liver_rights = df2[df2$v24.righting.response.low == "rights",]$liver.cfu.per.g.tissue # CFU in liver of mice which were able to right at 24 HPC
liver_ftr = df2[df2$v24.righting.response.low == "ftr",]$liver.cfu.per.g.tissue # CFU in liver of mice which were unable to right at 24 HPC
lung_rights = df2[df2$v24.righting.response.low == "rights",]$lung.cfu.per.g.tissue # CFU in lung of mice which were able to right at 24 HPC
lung_ftr = df2[df2$v24.righting.response.low == "ftr",]$lung.cfu.per.g.tissue # CFU in lung of mice which were unable to right at 24 HPC
```

The base R function wilcox.test() is used to perform the Wilcoxon Rank-Sum tests comparing bacterial load between mice righted or failed to right at 24 HPC(Figure 1C).
```{r Fig 1C wilcoxons}
wilcox.test(blood_rights, blood_ftr)
wilcox.test(spleen_rights, spleen_ftr)
wilcox.test(liver_rights, liver_ftr)
wilcox.test(lung_rights, lung_ftr)
```

The p-values are extracted using a simple custom function and the bonferroni adjustment is manually applied by multiplying each p-value by 4.
```{r Fig 1C stats}
pval_extraction = function(data, column, val1, val2, adjust){
  bld = wilcox.test(
    data[data[, column] == val1,]$blood.cfu.per.ml,
    data[data[, column] == val2,]$blood.cfu.per.ml)
  spl = wilcox.test(
    data[data[, column] == val1,]$spleen.cfu.per.g.tissue,
    data[data[, column] == val2,]$spleen.cfu.per.g.tissue)
  liv = wilcox.test(
    data[data[, column] == val1,]$liver.cfu.per.g.tissue,
    data[data[, column] == val2,]$liver.cfu.per.g.tissue)
  lun = wilcox.test(
    data[data[, column] == val1,]$lung.cfu.per.g.tissue,
    data[data[, column] == val2,]$lung.cfu.per.g.tissue)
  pvals = list(bld$p.value, spl$p.value, liv$p.value, lun$p.value)
  names(pvals) = c("blood", "spleen", "liver", "lung")
  adj.pvals = lapply(pvals, function(x) x*4) # Bonferroni
  if(adjust == FALSE) {
    return(lapply(pvals, function(x) signif(x, 2))) 
  } else {
    return(lapply(adj.pvals, function(x) signif(x, 2)))
  }
}
pval_extraction(df2, "v24.righting.response.low", "rights", "ftr", TRUE)
```
### Figure 2 - Score and survival
Standard error is defined manually, then the proportion of mice which succumb to illness is calculated using the dplyr summarise() function.
```{r Fig 2a workup}
se <- function(x) sqrt(var(x)/length(x))
df_sum = df %>% group_by(v24_score) %>%
  summarise(
    proportion_death = mean(outcome_binary),
    se = se(outcome_binary),
    n = n() 
  ) %>% as.data.frame()
df_sum
```
A simple linear regression is performed using base R lm() and summary() functions
```{r lin reg}
mod = lm(1-proportion_death ~ v24_score, data = df_sum)
summary(mod)
```

### Figure 3 - Bacterial load and score
Bacterial load data were previously found to not be normally distributed. A series of Kruskal-Wallis tests are used to quantify the relationship between score and bacterial load at 18 and 24 HPC. P-values are corrected using the Bonferroni method Scores of 0 are excluded from analysis as they are not independent from outcome.
```{r fig 3}
# Function to extract and adjust the p-values from the htest object generated by kruskal test
pval_extraction_kruskal = function(data, visit){
  bld = kruskal.test(
    data[data[, visit] != "0",]$blood.cfu.per.ml ~ 
      data[data[, visit] != "0",][,visit])
  spl = kruskal.test(
    data[data[, visit] != "0",]$spleen.cfu.per.g.tissue,
    data[data[, visit] != "0",][,visit])
  liv = kruskal.test(
    data[data[, visit] != "0",]$liver.cfu.per.g.tissue,
    data[data[, visit] != "0",][,visit])
  lun = kruskal.test(
    data[data[, visit] != "0",]$lung.cfu.per.g.tissue,
    data[data[, visit] != "0",][,visit])
  pvals = list(bld$p.value, spl$p.value, liv$p.value, lun$p.value)
  names(pvals) = c("blood", "spleen", "liver", "lung")
  adj.pvals = lapply(pvals, function(x) x*4)
  
  return(lapply(adj.pvals, function(x) signif(x, 2)))
}
# 18 HPC kruskal-wallis tests
# Blood 18 HPC
kruskal.test(df2[df2$v18.score.low != 0,]$blood.cfu.per.ml, 
             df2[df2$v18.score.low != 0,]$v18.score.low)
# Spleen 18 HPC
kruskal.test(df2[df2$v18.score.low != 0,]$spleen.cfu.per.g.tissue, 
             df2[df2$v18.score.low != 0,]$v18.score.low)
# Liver 18 HPC
kruskal.test(df2[df2$v18.score.low != 0,]$liver.cfu.per.g.tissue, 
             df2[df2$v18.score.low != 0,]$v18.score.low)
# Lung 18 HPC
kruskal.test(df2[df2$v18.score.low != 0,]$lung.cfu.per.g.tissue, 
             df2[df2$v18.score.low != 0,]$v18.score.low)
pval_extraction_kruskal(df2, "v18.score.low") # adjust p-values
## Repeat at 24 HPC

# 24 HPC kruskal-wallis tests
# Blood 24 HPC
kruskal.test(df2[df2$v24.score.low != 0,]$blood.cfu.per.ml, 
             df2[df2$v24.score.low != 0,]$v24.score.low)
# Spleen 24 HPC
kruskal.test(df2[df2$v24.score.low != 0,]$spleen.cfu.per.g.tissue, 
             df2[df2$v24.score.low != 0,]$v24.score.low)
# Liver 24 HPC
kruskal.test(df2[df2$v24.score.low != 0,]$liver.cfu.per.g.tissue, 
             df2[df2$v24.score.low != 0,]$v24.score.low)
# Lung 24 HPC
kruskal.test(df2[df2$v24.score.low != 0,]$lung.cfu.per.g.tissue, 
             df2[df2$v24.score.low != 0,]$v24.score.low)
pval_extraction_kruskal(df2, "v24.score.low") # adjust p-values
```


### Figure 4 - Classifier validation
Wilcoxon-rank sum tests are performed as before, this time comparing between predicted outcome. Numerical objects containing the CFU values for each group are defined prior to testing for ease of interpretation.
```{r Fig 4B workup}
df2$predicted_outcome = ifelse(df2$Prediction == 0, "non-survivor", "survivor") # Re-label survivors and non-survivors from binary 0/1 column (0 = non-survivor, 1 = survivor)
blood_surv = na.omit(df2[df2$predicted_outcome == "survivor", "blood.cfu.per.ml"])
blood_die = na.omit(df2[df2$predicted_outcome == "non-survivor", "blood.cfu.per.ml"])
spleen_surv = na.omit(df2[df2$predicted_outcome == "survivor", "spleen.cfu.per.g.tissue"])
spleen_die = na.omit(df2[df2$predicted_outcome == "non-survivor", "spleen.cfu.per.g.tissue"])
liver_surv = na.omit(df2[df2$predicted_outcome == "survivor", "liver.cfu.per.g.tissue"])
liver_die = na.omit(df2[df2$predicted_outcome == "non-survivor", "liver.cfu.per.g.tissue"])
lung_surv = na.omit(df2[df2$predicted_outcome == "survivor", "lung.cfu.per.g.tissue"])
lung_die = na.omit(df2[df2$predicted_outcome == "non-survivor", "lung.cfu.per.g.tissue"])
```
The base wilcox.test() function is again used, then p-values are extracted from the htest objects and the bonferroni adjustment is applied using the previously defined custom pval_extraction() function.
```{r Fig 4B wilcox}
wilcox.test(blood_surv, blood_die)
wilcox.test(spleen_surv, spleen_die)
wilcox.test(liver_surv, liver_die)
wilcox.test(lung_surv, lung_die)
# Apply bonferroni adjustment
pval_extraction(df2, # Data frame
                "predicted_outcome", # Factor to compare
                "survivor", "non-survivor", # Levels of factor 
                TRUE # Apply adjustment? T / F
                )
```


## Statistical analyses - Supplementary figures
### Figure S1
To rule out changes in score resulting from the IP injection, scores between mice which received a sham dPBS or D5W injection were compared against those which received a cecal slurry injection using the base r t.test() function at both 18 HPC and 24 HPC.
```{r S1 Fig}
df5$cs_or_sham = ifelse(df5$chal.type == "cecal slurry", "cs", "sham") # group D5W and dPBS injections together
# v1.score = score at 18 HPC
t.test(
  df5[df5$cs_or_sham == "cs",]$v1.score, # Score of CS challenged mice at 18 HPC
  df5[df5$cs_or_sham == "sham",]$v1.score, # Score of sham injected mice at 18 HPC
  paired = FALSE,
  var.equal = TRUE)
# v2.score = score at 24 HPC
t.test(
  df5[df5$cs_or_sham == "cs",]$v2.score, # Score of CS challenged mice at 24 HPC
  df5[df5$cs_or_sham == "sham",]$v2.score, # Score of sham injected mice at 24 HPC
  paired = FALSE,
  var.equal = TRUE)
```
### Session info
```{r session info}
sessionInfo()
```

