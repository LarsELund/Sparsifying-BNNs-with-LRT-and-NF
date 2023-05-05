library(varbvs)
X <- as.matrix(read.csv("~/Documents/Variable selection clean up code/sim3-X.csv", header=FALSE))
Y <- as.matrix(read.csv("~/Documents/Variable selection clean up code/sim3-Y.csv", header=FALSE))
X <- scale(X)
q <- 0.5
logodds = log10(q / ( 1 - q))

num_models <- 100
predicted_alphas <- matrix(0,nrow = num_models,ncol = 20) 
set.seed(1)
start_time <- Sys.time()

for (i in 1:num_models) {
  out = varbvs(X,Z = NULL,Y,family = 'binomial',logodds = logodds,verbose = FALSE,tol = 1e-6,nr = 1000)
  alpha <- (as.numeric(out$alpha) >  0.5 ) * 1 #median probability model
  predicted_alphas[i,] <- alpha
} 

print(Sys.time() - start_time)
write.table(predicted_alphas, file="alphas_Carb_05.txt", row.names=FALSE, col.names=FALSE)
