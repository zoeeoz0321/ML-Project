library(ggplot2)
library(tidyverse)
library(caTools)
library(dplyr)
library(sjPlot)
library(sjlabelled)
library(mlr)
library(rpart)

# load data
df <- read.csv("./tutorials/mental health in tech.csv")
summary(df)

# transform data type
df$Gender <- plyr::revalue(df$Gender,c('Female'='F', 'M'='M', 'Male'='M', 'male'='M', 'female'='F', 'm'='M', 'Male-ish'='M', 'maile'='M', 'Trans-female'='F',
 'Cis Female'='F', 'F'='F', 'something kinda male?'='M', 'Cis Male'='M', 'Woman'='M', 'f'='F', 'Mal'='M',
 'Male (CIS)'='M', 'queer/she/they'='M', 'non-binary'='F', 'Femake'='F', 'woman'='M', 'Make'='M', 'Nah'='F',
 'All'='F', 'Enby'='F', 'fluid'='F', 'Genderqueer'='F', 'Female '='F', 'Androgyne'='M', 'Agender'='M',
 'cis-female/femme'='F', 'Guy (-ish) ^_^'='M', 'male leaning androgynous'='M', 'Male '='M',
 'Man'='M', 'Trans woman'='F', 'msle'='M', 'Neuter'='F', 'Female (trans)'='F', 'queer'='F',
 'Female (cis)'='F', 'Mail'='M', 'cis male'='M', 'A little about you'='F', 'Malr'='M', 'p'='F', 'femail'='F',
 'Cis Man'='M', 'ostensibly male, unsure what that really means'='M'))
df$Gender <- factor(df$Gender, labels=c('F','M'))
df$Country <- as.factor(df$Country)
df$state <- as.factor(df$state)
df$self_employed <- as.factor(df$self_employed)
df$family_history <- as.factor(df$family_history)
df$treatment <- as.factor(df$treatment)
df$work_interfere <- as.factor(df$work_interfere)
df$no_employees <- as.factor(df$no_employees)
df$remote_work <- as.factor(df$remote_work)
df$tech_company <- as.factor(df$tech_company)
df$benefits <- as.factor(df$benefits)
df$care_options <- as.factor(df$care_options)
df$wellness_program <- as.factor(df$wellness_program)
df$seek_help <- as.factor(df$seek_help)
df$anonymity <- as.factor(df$anonymity)
df$leave <- as.factor(df$leave)
df$mental_health_consequence <- as.factor(df$mental_health_consequence)
df$phys_health_consequence <- as.factor(df$phys_health_consequence)
df$coworkers <- as.factor(df$coworkers)
df$supervisor <- as.factor(df$supervisor)
df$mental_health_interview <- as.factor(df$mental_health_interview)
df$phys_health_interview <- as.factor(df$phys_health_interview)
df$mental_vs_physical <- as.factor(df$mental_vs_physical)
df$obs_consequence <- as.factor(df$obs_consequence)

summary(df)

qqnorm(df$Age, main="Age - Normal Q-Q plot")
qqline(df$Age)

q <- quantile(df$Age, probs=c(.25, .75), na.rm = T)
iqr <- IQR(df$Age, na.rm = T)

df1 <- df %>% filter(Age > (q[1] - 1.5 * iqr) & Age < (q[2] + 1.5 * iqr))
par(mfrow=c(2,1))
options(repr.plot.width=10, repr.plot.height=5)
boxplot(df$Age, col='grey40', horizontal=T, main='Age - Before Removing Outliers')
boxplot(df1$Age, col='seagreen3', horizontal=T, main='Age - After Removing Outliers')

par(mfrow=c(1,1))
df[df[,"Age"]<0,"Age"] <- 0
df[df[,"Age"]>100, "Age"] <- 100
p <- ggplot(df,aes(Age)) 
p + geom_histogram(color="black", fill="lightblue", binwidth=2) + geom_vline(aes(xintercept=mean(Age)), color="red", linetype="dashed", size=1) + theme_bw()

cols <- colnames(df)

# Make plots. 
bar_list <- list() 
rate_list <- list()
for (i in 1:27) {
  bar <- ggplot(df, aes_string(x=cols[i])) + geom_bar() + theme(axis.text.x = element_text(angle = 45, hjust = 0.5, vjust = 0.5))
  bar_list[[i]] <- bar 
  rate <- ggplot(df, aes_string(x=cols[i],fill='treatment')) + geom_bar(position='fill') + theme(axis.text.x = element_text(angle = 45, hjust = 0.5, vjust = 0.5))
  rate_list[[i]] <- rate
} 

# Save plots to jpg. Makes a separate file for each plot. 
for (i in 1:27) { 
  jpeg(filename = paste0("./images/",cols[i],"_barplot.jpg"), width = 600, height = 800,res = 300)
  print(bar_list[i]) 
  dev.off() 
} 
for (i in 1:27) { 
  jpeg(filename = paste0("./images/",cols[i],"_rateplot.jpg"), width = 1000, height = 600,res = 300)
  print(rate_list[i]) 
  dev.off() 
} 
dev.off() 

print(sapply(df, function(x) sum(is.na(x))))

round((sum(is.na(df))/nrow(df)))

missmap(df, main="Missing Map")

# reformat
cols <- colnames(df)
recols <- cols[-which(cols=='Timestamp'|cols=='state'|cols=='comments')]
df <- df[recols]

# delete the rows with NAs
df1 <- drop_na(df)
nrow(df1)

# create task and split dataset
library(mlbench)
mental <- createDummyFeatures(df1, target='treatment')
task <- makeClassifTask(data=mental, target='treatment')
holdout <- makeResampleInstance("Holdout", task)
task.train <- subsetTask(task, holdout$train.inds[[1]])
task.test <- subsetTask(task, holdout$test.inds[[1]])

library(FSelector)
var_imp <- generateFilterValuesData(task, method = "information.gain")
var_imp <- generateFilterValuesData(task, method = c("information.gain"))
plotFilterValues(var_imp,n.show=10, feat.type.cols = TRUE)

# create classifier and evaluate performance
clf1 <- makeLearner("classif.rpart",predict.type='prob')
cv1 <- makeResampleDesc("CV",iters=5)
result1 <- resample(clf1, task.train, cv1, acc)

model1 <- train(clf1, task.train)
pred1 <- predict(model1, task.test)
calculateConfusionMatrix(pred1)

res1 <- performance(pred1, measures = list(fpr, fnr, auc))
res1

otp1 <- generateThreshVsPerfData(pred1, measures = list(fpr, fnr, mmce))
plotThreshVsPerf(otp1)

library(rpart.plot)
rpart.plot(model1$learner.model, roundint=FALSE, varlen=100, type = 3, clip.right.labs = FALSE, yesno = 2)



# create classifier and evaluate performance
clf2 <- makeLearner("classif.randomForest",predict.type = "prob")
cv2 <- makeResampleDesc("CV",iters=5)
result2 <- resample(clf2, task.train, cv2, acc)

model2 <- train(clf2, task.train)
pred2 <- predict(model2, task.test)
calculateConfusionMatrix(pred2)

res2 <- performance(pred2, measures = list(fpr, fnr, auc))
res2

otp2 = generateThreshVsPerfData(pred2, measures = list(fpr, fnr, mmce))
plotThreshVsPerf(otp2)

cur = generateThreshVsPerfData(list(decisionTree=pred1, randomForest=pred2), measures = list(fpr, tpr))
plotROCCurves(cur)

