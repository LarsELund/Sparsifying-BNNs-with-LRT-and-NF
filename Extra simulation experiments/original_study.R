library(simrel)
library(varbvs)
library(parallel)
num_models = 20

library(varbvs)
X <- as.matrix(read.csv("https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/Mode%20Jumping%20MCMC/supplementary/examples/Simulated%20Logistic%20Data%20With%20Multiple%20Modes%20(Example%203)/sim3-X.txt", header=FALSE))
Y <- as.matrix(read.csv("https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/Mode%20Jumping%20MCMC/supplementary/examples/Simulated%20Logistic%20Data%20With%20Multiple%20Modes%20(Example%203)/sim3-Y.txt", header=FALSE))
X <- scale(X)
q <- 0.25
logodds = log10(q / ( 1 - q))

tpr.quants = matrix(NA,30,3)
fpr.quants = matrix(NA,30,3)

corrs = matrix(NA,30,3)
cor = numeric(30)
cor_max = numeric(30)

i = 0
beta = c(-4,0,1,0,0,0,1,0,0,0,1.2,0,37.1,0,0,50,-0.00005,10,3,0)
tps = beta!=0
for(sigma in c(1,0.1,0.01))
{
  for(j in c(c(1:10)/10))
  {
    i = i + 1
    set.seed(i)
   
    X = as.matrix(read.csv("https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/Mode%20Jumping%20MCMC/supplementary/examples/Simulated%20Logistic%20Data%20With%20Multiple%20Modes%20(Example%203)/sim3-X.txt", header=FALSE))
    for(k in 1:4)
      X[,(k*4+1):((k+1)*4)] = j*X[,(k*4+1):((k+1)*4)] + (1-j)*X[,((k-1)*4+1):(k*4)] 
    X = scale(X)
    cor_mat = abs(cor(X))
    cor[i] = mean(cor_mat[lower.tri(cor_mat)])
    cor_max[i] = max(cor_mat[lower.tri(cor_mat)])
    
    nu = rnorm(n = nrow(X),beta%*%t(X),sd = sigma)
    
    Y = rbinom(2000,size = 1,prob = 1/(1+exp(-nu)))
    write.table(X,paste(path,'//X_train_original',toString(i-1),'.csv',sep = ''), row.names=FALSE,sep = ',',quote = FALSE)
    write.table(Y,paste(path,'//Y_train_original',toString(i-1),'.csv',sep = ''), row.names=FALSE,sep = ',',quote = FALSE)
    
    print(unique(sort(abs(cor(X)[beta==0,beta!=0]),decreasing = T)[1:10]))
    
    corrs[i,1] = mean(abs(cor(X)[beta==0,beta!=0]))
    corrs[i,2] = min(abs(cor(X)[beta==0,beta!=0]))
    corrs[i,3] = max(abs(cor(X)[beta==0,beta!=0]))
    
    #res = do.call(rbind,mclapply(1:num_models,FUN = function(x){
    #  set.seed(x)
    #  out = varbvs(X,Z = NULL,Y,family = "binomial",logodds = logodds,verbose = FALSE,tol = 1e-6,nr = 1000)
    #  alpha = (as.numeric(out$alpha) >  0.5 ) * 1 #median probability model
    #  tpr = sum(alpha[tps])/sum(tps)
    #  fpr = sum(alpha[!tps])/max(sum(!tps),1)
    #  return(c(tpr,fpr))
  #  }))
    
  #  tpr.quants[i,] = quantile(res[,1],probs = c(0.1,0.5,0.9))
   # fpr.quants[i,] = quantile(res[,2],probs = c(0.1,0.5,0.9))
  }
}

plot(corrs[,1],type = "l",ylim=c(0,1))
lines(corrs[,2],col = 2)
lines(corrs[,3],col = 3)

plot(tpr.quants[,2],col = 1,type = "l",main = "TPR",ylim = c(0,1))
lines(tpr.quants[,1],col = 2, lty = "dotted")
lines(tpr.quants[,3],col = 2, lty = "dotted")
abline(v = 10,col = 4)
abline(v = 20,col = 4)

plot(fpr.quants[,2],col = 1,type = "l",main = "FPR",ylim = c(0,1))
lines(fpr.quants[,1],col = 2, lty = "dotted")
lines(fpr.quants[,3],col = 2, lty = "dotted")
abline(v = 10,col = 4)
abline(v = 20,col = 4)


#write.table(tpr.quants,'/Users/larsskaaret-lund/Documents/Extra results paper 1/sim/tpr_original.csv',sep = ',',row.names = FALSE,col.names = FALSE)
#write.table(fpr.quants,'/Users/larsskaaret-lund/Documents/Extra results paper 1/sim/fpr_original.csv',sep = ',',row.names = FALSE,col.names = FALSE)

write.table(cor,'/Users/larsskaaret-lund/Documents/Extra results paper 1/sim/cov_original.csv',sep = '',row.names = FALSE,col.names = FALSE)
write.table(beta,'/Users/larsskaaret-lund/Documents/Extra results paper 1/sim/beta_original.csv',sep = '',row.names = FALSE,col.names = FALSE)

write.table(cor_max,'/Users/larsskaaret-lund/Documents/Extra results paper 1/sim/cov_max_original.csv',sep = '',row.names = FALSE,col.names = FALSE)