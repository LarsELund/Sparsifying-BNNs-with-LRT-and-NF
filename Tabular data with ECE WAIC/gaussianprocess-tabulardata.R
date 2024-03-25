library(mgpr)




RMSE <- numeric(10)

for (i in 0:9) {
  print(i)
  path = paste("~/Documents/Tabular dataset WAIC ECE/Tabular data/Wine Quality/wine/wine_",sep ='')
  train_path = paste(path,"train",i,".csv",sep = '')
  test_path = paste(path,"test",i,".csv",sep = '')
  train_data = ca_test0 <- read.csv(train_path, header=FALSE)
  test_data = ca_test0 <- read.csv(test_path, header=FALSE)

  x<- train_data[,1:11]
  y<- train_data[,12]
  gp1 <- mgpr(datay = y, datax = x, 
              kernel = "matern32", meanf = "avg", kernpar = list(sigma = 1, corlen = 5, errorvar = 0.1))
  
  test_x <- as.matrix(test_data[,1:11])
  
  y_hat <- predict(gp1,test_x)
  y_test <- test_data[,12]
  
  RMSE[i +1] <- sqrt(mean((y_hat$V1 - y_test)^2))
  
  write.table(y_test, paste("wine_y_true",i,".csv",sep = ''),row.names = FALSE,col.names = FALSE)
  write.table(y_hat, paste("wine_y_pred",i,".csv",sep = ''),row.names = FALSE,col.names = FALSE)
  
} 








