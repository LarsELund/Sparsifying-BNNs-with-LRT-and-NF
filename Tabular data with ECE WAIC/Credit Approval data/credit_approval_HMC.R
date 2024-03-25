ca_test0 <- read.csv("~/Documents/Tabular dataset WAIC ECE/Tabular data/Credit Approval data/credit_approval_test0.csv", header=FALSE)
ca_train0 <- read.csv("~/Documents/Tabular dataset WAIC ECE/Tabular data/Credit Approval data/credit_approval_train0.csv", header=FALSE)

library("devtools")
library(BLNN)
library(fairml)
library(rstan)

path = paste("~/Documents/Tabular dataset WAIC ECE/Tabular data/Credit Approval data/credit_approval_","test","0",".csv",sep ='')

for (i in 0:9) {
  path = paste("~/Documents/Tabular dataset WAIC ECE/Tabular data/Credit Approval data/credit_approval_",sep ='')
  train_path = paste(path,"train",i,".csv",sep = '')
  test_path = paste(path,"test",i,".csv",sep = '')
  train_data = ca_test0 <- read.csv(train_path, header=FALSE)
  test_data = ca_test0 <- read.csv(test_path, header=FALSE)

} 



ClassNet<-BLNN_Build(ncov=15, nout=1, hlayer_size = 3,
                     actF = "tanh",
                     costF = "crossEntropy",
                     outF = "sigmoid",
                     hp.Err = 20, hp.W1 = 20, hp.W2=20,
                     hp.B1 = 20, hp.B2 = 20)

x <- as.matrix(ca_train0[,1:15])
y <- ca_train0[,16]

num_chains = 2
initials = lapply(1:num_chains, 
                  function(i) rnorm(length(BLNN_GetWts(ClassNet)),0,0.1))
n.par = length(BLNN_GetWts(ClassNet))
m1 <- rep(1/2, n.par)

ClassHMC <- BLNN_Train(NET = ClassNet,
                       x = x,
                       y = y,
                       iter = 100000,
                       
                       init = initials,
                       warmup = 10,
                       chains = num_chains,
                       algorithm = "HMC",
                       display = 1,  control = list(adapt_delta = 0.80, momentum_mass = m1, stepsize =5 , gamma=1,
                                                    Lambda = 0.001)
                       )



ClassHMC<-BLNN_Update(ClassNet, ClassHMC)
x_test <- ca_test0[,1:15]
out <- BLNN_Predict(ClassNet,x_test)
out_pred = (out > 0.5) * 1
y_test = ca_test0[,16]
print(mean(out_pred == y_test))


