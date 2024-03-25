library("devtools")
library(BLNN)
library(fairml)
library(rstan)
library(parallel)

results = mclapply(mc.cores = 10,X = 0:9,FUN = function(fold)
{
  ca_test0 <- read.csv(paste0("abalone_test",fold,".csv"), header=FALSE)
  ca_train0 <- read.csv(paste0("abalone_train",fold,".csv"), header=FALSE)
  
  ClassNet<-BLNN_Build(ncov=10, nout=1, hlayer_size = 10,
                       actF = "tanh",
                       costF = "MSE",
                       outF = "linear",
                       hp.Err = 1, hp.W1 = 1, hp.W2=1,
                       hp.B1 = 1, hp.B2 = 1)
  
  x <- as.matrix(ca_train0[,1:10])
  y <- ca_train0[,11]
  
  num_chains = 1
  initials = lapply(1:num_chains, 
                    function(i) rnorm(length(BLNN_GetWts(ClassNet)),0,0.001))
  n.par = length(BLNN_GetWts(ClassNet))
  m1 <- rep(1, n.par)
  
  ClassHMC <- BLNN_Train(NET = ClassNet,
                         x = x,
                         y = y,
                         iter = 100000,
                         init = initials,
                         warmup = 10,
                         chains = num_chains,
                         algorithm = "HMC",
                        control = list(adapt_delta = 0.65, momentum_mass = m1, stepsize = 1 , gamma=2,
                                                      Lambda = 0.005)
        
  )
  
  print(ClassHMC$Rhat)
  write.csv(x = ClassHMC$Rhat,file = paste0("res/rhat_abalone_resutls_",fold,".csv"))
  
  ClassHMC<-BLNN_Update(ClassNet, ClassHMC)
  x_test <- ca_test0[,1:10]
  out <- BLNN_Predict(ClassNet,x_test)
  y_test = ca_test0[,11]
  
  
  write.csv(x = out,file = paste0("res/abalone_resutls_",fold,".csv"))
  
  return(sqrt(mean((out - ca_test0[,11])^2)))
})

