##Mapping SAV for BTT May report
##initial date: 5/7/2025
##Author: Jon Rodemann
library(tidyverse)
library(terra)
library(sf)
library(randomForest)
library(mlr3)
library(mlr)

savdat <mlr3savdat <- read.csv('SAV.csv')
savpoints <- read.csv('SAV_points.csv')

sav <- merge(savdat, savpoints, by = 'SQ')

write.csv(sav, file = 'SAV_surveys.csv')

##Sat images
sak <- terra::rast('Layers/SAK_clip.tif')
plot(sak)
shk <- terra::rast('Layers/SHK_clip.tif')
plot(shk)
saw <- terra::rast('Layers/SAW_clip.tif')
plot(saw)
lhb <- terra::rast('Layers/LHBT_clip.tif')
plot(lhb)
jfb <- terra::rast('Layers/JFB_clip.tif')
plot(jfb)
mrs <- terra::rast('Layers/MRS_clip.tif')
plot(mrs)

##load in shaefiles of training
sakb <- terra::vect('Layers/SAK_bare.shp')
plot(sakb)
sakd <- terra::vect('Layers/SAK_dense.shp')
plot(sakd)
shkb <- terra::vect('Layers/SHK_bare.shp')
shkd <- terra::vect('Layers/SHK_dense.shp')
sawb <- terra::vect('Layers/SAW_bare.shp')
sawd <- terra::vect('Layers/SAW_dense.shp')
lhbb <- terra::vect('Layers/LHBT_bare.shp')
lhbd <- terra::vect('Layers/LHBT_dense.shp')
jfbb <- terra::vect('Layers/JFB_bare.shp')
jfbd <- terra::vect('Layers/JFB_dense.shp')
mrsb <- terra::vect('Layers/MRS_bare.shp')
mrsd <- terra::vect('Layers/MRS_dense.shp')

##Extract and create dataframe

sakbd <- terra::extract(sak, sakb) %>% mutate(train = 'bare')
sakdd <- terra::extract(sak, sakd) %>% mutate(train = 'dense')
shkbd <- terra::extract(shk, shkb)%>% mutate(train = 'bare')
shkdd <- terra::extract(shk, shkd) %>% mutate(train = 'dense')
sawbd <- terra::extract(saw, sawb)%>% mutate(train = 'bare')
sawdd <- terra::extract(saw, sawd) %>% mutate(train = 'dense')
lhbbd <- terra::extract(lhb, lhbb)%>% mutate(train = 'bare')
lhbdd <- terra::extract(lhb, lhbd) %>% mutate(train = 'dense')
jfbbd <- terra::extract(jfb, jfbb)%>% mutate(train = 'bare')
jfbdd <- terra::extract(jfb, jfbd) %>% mutate(train = 'dense')
mrsbd <- terra::extract(mrs, mrsb)%>% mutate(train = 'bare')
mrsdd <- terra::extract(mrs, mrsd) %>% mutate(train = 'dense')

##combine data
sak_d <- rbind(sakbd, sakdd) %>% select(-ID)
shk_d <- rbind(shkbd, shkdd) %>% select(-ID)
saw_d <- rbind(sawbd, sawdd) %>% select(-ID)
lhb_d <- rbind(lhbbd, lhbdd) %>% select(-ID)
jfb_d <- rbind(jfbbd, jfbdd) %>% select(-ID)
mrs_d <- rbind(mrsbd, mrsdd) %>% select(-ID)

##Random Forest to create models
#SAK
rforestLearner <- makeLearner('classif.randomForest')
SAVTask <- makeClassifTask(data = sak_d, target = 'train')
SAVtrained <- train(rforestLearner, SAVTask)
SAVtrained

p <- predict(SAVtrained, newdata = sak_d) #overfitting easily can happen, evaluating on training set
performance(p, measures = list(acc, mmce))

SAVtrained
str(SAVtrained)
getFeatureImportance(SAVtrained)
SAVmodel <- getLearnerModel(SAVtrained)

SAVmodel$predicted

SAVmap <- terra::predict(sak, SAVmodel)
plot(SAVmap)

writeRaster(SAVmap, filename = 'outputs/SAK1.tif')

#SHK
rforestLearner <- makeLearner('classif.randomForest')
SAVTask <- makeClassifTask(data = shk_d, target = 'train')
SAVtrained <- train(rforestLearner, SAVTask)
SAVtrained

p <- predict(SAVtrained, newdata = shk_d) #overfitting easily can happen, evaluating on training set
performance(p, measures = list(acc, mmce))

SAVtrained
str(SAVtrained)
getFeatureImportance(SAVtrained)
SAVmodel <- getLearnerModel(SAVtrained)

SAVmodel$predicted

SAVmap <- terra::predict(shk, SAVmodel)
plot(SAVmap)

writeRaster(SAVmap, filename = 'outputs/shk1.tif')

#saw
rforestLearner <- makeLearner('classif.randomForest')
SAVTask <- makeClassifTask(data = saw_d, target = 'train')
SAVtrained <- train(rforestLearner, SAVTask)
SAVtrained

p <- predict(SAVtrained, newdata = saw_d) #overfitting easily can happen, evaluating on training set
performance(p, measures = list(acc, mmce))

SAVtrained
str(SAVtrained)
getFeatureImportance(SAVtrained)
SAVmodel <- getLearnerModel(SAVtrained)

SAVmodel$predicted

SAVmap <- terra::predict(saw, SAVmodel)
plot(SAVmap)

writeRaster(SAVmap, filename = 'outputs/saw2.tif')

#lhb
rforestLearner <- makeLearner('classif.randomForest')
SAVTask <- makeClassifTask(data = lhb_d, target = 'train')
SAVtrained <- train(rforestLearner, SAVTask)
SAVtrained

p <- predict(SAVtrained, newdata = lhb_d) #overfitting easily can happen, evaluating on training set
performance(p, measures = list(acc, mmce))

SAVtrained
str(SAVtrained)
getFeatureImportance(SAVtrained)
SAVmodel <- getLearnerModel(SAVtrained)

SAVmodel$predicted

SAVmap <- terra::predict(lhb, SAVmodel)
plot(SAVmap)

writeRaster(SAVmap, filename = 'outputs/lhb2.tif')

#jfb
rforestLearner <- makeLearner('classif.randomForest')
SAVTask <- makeClassifTask(data = jfb_d, target = 'train')
SAVtrained <- train(rforestLearner, SAVTask)
SAVtrained

p <- predict(SAVtrained, newdata = jfb_d) #overfitting easily can happen, evaluating on training set
performance(p, measures = list(acc, mmce))

SAVtrained
str(SAVtrained)
getFeatureImportance(SAVtrained)
SAVmodel <- getLearnerModel(SAVtrained)

SAVmodel$predicted

SAVmap <- terra::predict(jfb, SAVmodel)
plot(SAVmap)

writeRaster(SAVmap, filename = 'outputs/jfb1.tif')

#mrs## - not included
rforestLearner <- makeLearner('classif.randomForest')
SAVTask <- makeClassifTask(data = mrs_d, target = 'train')
SAVtrained <- train(rforestLearner, SAVTask)
SAVtrained

p <- predict(SAVtrained, newdata = mrs_d) #overfitting easily can happen, evaluating on training set
performance(p, measures = list(acc, mmce))

SAVtrained
str(SAVtrained)
getFeatureImportance(SAVtrained)
SAVmodel <- getLearnerModel(SAVtrained)

SAVmodel$predicted

SAVmap <- terra::predict(mrs, SAVmodel)
plot(SAVmap)

writeRaster(SAVmap, filename = 'outputs/mrs1.tif')

##accuracy assessment
sav_points <- terra::vect('Layers/sav_surveys.shp')

sav_lhb <- sav_points[grepl("LHB", sav_points$SQ), ]
sav_saw <- sav_points[grepl("SAW", sav_points$SQ), ]
sav_jfb <- sav_points[grepl("JFB", sav_points$SQ), ]
sav_sak <- sav_points[grepl("SAK", sav_points$SQ), ]

slhb <- terra::extract(SAVmap, sav_lhb)
slhb1 <- as.data.frame(cbind(sav_lhb, slhb))
slhb1 <- slhb1 %>% mutate(survey = if_else(Tot >= 25, 'dense', 'bare')) %>% mutate(check = if_else(survey == class, 1, 0))
sum(slhb1$check)/30

ssaw <- terra::extract(SAVmap, sav_saw)
ssaw1 <- as.data.frame(cbind(sav_saw, ssaw))
ssaw1 <- ssaw1 %>% mutate(survey = if_else(Tot >= 25, 'dense', 'bare')) %>% mutate(check = if_else(survey == class, 1, 0))
sum(ssaw1$check)/30

sjfb <- terra::extract(SAVmap, sav_jfb)
sjfb1 <- as.data.frame(cbind(sav_jfb, sjfb))
sjfb1 <- sjfb1 %>% mutate(survey = if_else(Tot >= 25, 'dense', 'bare')) %>% mutate(check = if_else(survey == class, 1, 0))
sum(sjfb1$check)/30

ssak <- terra::extract(SAVmap, sav_sak)
ssak1 <- as.data.frame(cbind(sav_sak, ssak))
ssak1 <- ssak1 %>% mutate(survey = if_else(Tot >= 25, 'dense', 'bare')) %>% mutate(check = if_else(survey == class, 1, 0))
sum(ssak1$check)/30
