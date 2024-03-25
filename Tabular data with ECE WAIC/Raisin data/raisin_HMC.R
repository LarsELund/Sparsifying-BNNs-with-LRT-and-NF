raisins_test0 <- read.csv("~/Documents/Tabular dataset WAIC ECE/Tabular data/Raisin data/raisins_test0.csv", header=FALSE)
raisins_train0 <- read.csv("~/Documents/Tabular dataset WAIC ECE/Tabular data/Raisin data/raisins_train0.csv", header=FALSE)

library("devtools")
library(BLNN)
library(fairml)
library(rstan)


             
ClassNet<-BLNN_Build(ncov=7, nout=1, hlayer_size = 500,
                     actF = "tanh",
                     costF = "crossEntropy",
                     outF = "sigmoid",
                     hp.Err = 10, hp.W1 = 10, hp.W2=10,
                     hp.B1 = 10, hp.B2 = 10)

x <- as.matrix(raisins_train0[,1:7])
y <- raisins_train0[,8]

num_chains = 1
initials = lapply(1:num_chains, 
                 function(i) rnorm(length(BLNN_GetWts(ClassNet)),0,0.3))
n.par = length(BLNN_GetWts(ClassNet))
m1 <- rep(1/500 ^2, n.par)

ClassHMC <- BLNN_Train(NET = ClassNet,
                       x = x,
                       y = y,
                       iter = 100000,
        
                       init = initials,
                       warmup = 5000,
                       chains = num_chains,
                       algorithm = "HMC",
                       display = 1, control = list(adapt_delta = 0.5,
                                                   Lambda = 0.0001,stepsize= 5,
                                                   momentum_mass = m1
                                        )
)


ClassHMC<-BLNN_Update(ClassNet, ClassHMC)

out <- BLNN_Predict(ClassNet,x_test)
out_pred = (out > 0.5) * 1
y_test = raisins_test0[,8]
print(mean(out_pred == y_test))


