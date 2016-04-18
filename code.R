rm(list=ls())
options(scipen=999)

#set workingdirectory
setwd("~/Personal/Assign")

###############################################################
#             Invoke required library                         #
###############################################################
require("tseries")||install.packages("tseries");library("tseries")
require("timeSeries")||install.packages("timeSeries");library("timeSeries")
require("forecast")||install.packages("forecast");library("forecast")
require("xgboost")||install.packages("xgboost");library("xgboost")
require("ggplot2")||install.packages("ggplot2");library("ggplot2")
require("lubridate")||install.packages("lubridate");library("lubridate")
require("Matrix")||install.packages("Matrix");library("Matrix")

#read file
data=read.delim("data_scientist_assignment.tsv",header=TRUE)
attach(data)

#join data and hour column to plot
data$newdate <- with(data, as.POSIXct(paste(date, hr_of_day), format="%Y-%m-%d %H"))
head(data)

#plot data
plot(vals ~ newdate, data=data, type="b",  col="blue")

# ggplot(data, aes(newdate, vals)) + 
#   geom_point() + geom_line() 

#convert data to time series
tsdata=ts(data$vals,frequency=24)
plot(tsdata)
abline(reg=lm(tsdata~time(tsdata)),col="blue")  #add regression line

#summarize
summary(tsdata)

boxplot(tsdata~cycle(tsdata))

plot(decompose(tsdata))

#applying transformations
logdata=log(tsdata)

plot(logdata)
abline(reg=lm(logdata~time(logdata)),col="blue")  #add regression line

#summarize
summary(logdata)

#check and replace outliers
(tsout=tsoutliers(logdata,iterate=24))
logdatarep=logdata
logdatarep[tsout$index]=tsout$replacements
plot(logdatarep,type="l")
summary(logdatarep)

#differencing
logdatas=logdatarep[25:3264]-logdatarep[25:3264-24]

#outliers replace - 2
(tsout2=tsoutliers(logdatas,iterate=24))
logdatarep2=logdatas
logdatarep2[tsout2$index]=tsout2$replacements
plot(logdatarep2,type="l")
summary(logdatarep2)

#final data
datanew = logdatas
rm(logdatas)
rm(logdatarep2)
rm(logdatarep)
rm(logdata)

#Augmented Dickey-Fuller Test
adf.test(datanew, alternative="stationary", k=0)

#Plot ACF/PACF and find optimal parameters
tsdisplay(datanew)

#decomposing
# plot(decompose(datanew))
# fitstl=stl(datanew, t.window = NULL, s.window=24, robust=TRUE)
# plot(fitstl)

#Build Arima model
(findbest <- auto.arima(datanew)) #ARIMA(2,0,2)

#after multiple iterations
(fitSARIMA <- arima(datanew, order = c(3,0,4), seasonal= list(order=c(1,0,0), period=24)))
(fitSARIMA2 <- arima(datanew, order = c(5,0,2), seasonal= list(order=c(1,0,0), period=24)))

#residuals
tsdisplay(residuals(fitSARIMA2))
Box.test(residuals(fitSARIMA2), lag=24, fitdf=4, type="Ljung-Box")

#Make predictions
tspred=tsdata[25:3264-24]*exp(fitted.values(fitSARIMA2))
tspredtable=cbind(tsdata[25:3264],tspred)
head(tspredtable)
plot(tspredtable)
plot(tspred, tsdata[25:3264])

plot(forecast(fitSARIMA2,h=10*24))
forecastSARIMA <- forecast(fitSARIMA2, level=c(80,95), h=10*24)

#######################################################################################
#supervised learning parameters
#day of the year
data$yearday=format(data$newdate,"%j")
data$yearday<-as.integer(data$yearday)

#Day of the month
data$monthday =day(data$newdate)

#Day of the week
data$weekday=format(data$newdate,"%w")
data$weekday=as.integer(data$weekday)

#Ordinal date 
data$datefrom1=difftime(data$newdate,"01-01-1900", units="days")
data$datefrom1=as.integer(data$datefrom1)

features=c("hr_of_day","yearday","monthday","weekday","vals")
data1=data[,features]
head(data1)
str(data1)

summary(data1)

#Train Test
set.seed(2345)
sub = sample(nrow(data1), floor(nrow(data1) * 0.7))
train = data1[sub,]
yTrain = train$vals
test = data1[-sub,]
yTest = test$vals
feature.formula = formula(vals~hr_of_day+yearday+monthday+weekday)
#Lasso regression

# Matrix
indexes <- sample(seq_len(nrow(train)), floor(nrow(train)*0.85))
datax <- sparse.model.matrix(feature.formula, data = train[indexes, ])
sparseMatrixColNamesTrain <- colnames(datax)
dtrain <- xgb.DMatrix(datax, label = train[indexes, 'vals'])
rm(datax)
dvalid <- xgb.DMatrix(sparse.model.matrix(feature.formula, data = train[-indexes, ]),
                      label = train[-indexes, 'vals'])
dtest <- sparse.model.matrix(feature.formula, data = test)

watchlist <- list(valid = dvalid, train = dtrain)

# XGBOOST
params <- list(booster = "gbtree", objective = "reg:linear",
               max_depth = 8, eta = 0.02,
               colsample_bytree = 0.8, subsample = 0.9)
model <- xgb.train(params = params, data = dtrain,
                   nrounds = 500, early.stop.round = 50,
                     maximize = F,
                   watchlist = watchlist, print.every.n = 10)

pred <- predict(model, dtest)

plot(pred,yTest)
