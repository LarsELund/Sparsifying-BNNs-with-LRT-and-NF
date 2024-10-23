library(simrel)
library(varbvs)
library(parallel)
num_models = 20

q = 0.25
logodds = log10(q / ( 1 - q))

tpr.quants = matrix(NA,30,3)
fpr.quants = matrix(NA,30,3)
i = 0

cor = numeric(30)
max_cor = numeric(30)

set.seed(1)
tps = runif(50,0,1) > 0.25
beta = c(tps*rnorm(10,ifelse(runif(10>0.5),1,-1),1))
for(sigma in c(50,10,1))
{
  for(j in c(c(10:1)/10))
  {
    i = i + 1
    set.seed(i)
    
    X = matrix(nrow = 1000, ncol = 50, rnorm(1000*50))
    for(k in 1:4)
      X[,(k*10+1):((k+1)*10)] = j*X[,(k*10+1):((k+1)*10)] + (1-j)*X[,((k-1)*10+1):(k*10)] 
    X = scale(X)
    X = sapply(1:dim(X)[2], function(i){
      if(i%%2 == 0)
        X[,i]
      else
        as.integer(X[,i]>median(X[,i]))
    })
    Y = rnorm(n = nrow(X),beta%*%t(X),sd = sigma)
    Y = as.integer(Y>median(Y))
    
    cor_mat = abs(cor(X))
    cor[i] = mean(cor_mat[lower.tri(cor_mat)])
    max_cor[i] = max(cor_mat[lower.tri(cor_mat)])
    path <-  '/Users/larsskaaret-lund/Documents/Extra results paper 1/sim'
    write.table(X,paste(path,'//X_train_comp',toString(i-1),'.csv',sep = ''), row.names=FALSE,sep = ',',quote = FALSE)
    write.table(Y,paste(path,'//Y_train_comp',toString(i-1),'.csv',sep = ''), row.names=FALSE,sep = ',',quote = FALSE)
    
    
    
    print(unique(sort(abs(cor(X)[beta==0,beta!=0]),decreasing = T)[1:10]))
    
    res = do.call(rbind,mclapply(1:num_models,FUN = function(x){
      set.seed(x)
      out = varbvs(X,Z = NULL,Y,family = "binomial",logodds = logodds,verbose = FALSE,tol = 1e-6,nr = 1000)
      alpha = (as.numeric(out$alpha) >  0.5 ) * 1 #median probability model
      tpr = sum(alpha[tps])/sum(tps)
      fpr = sum(alpha[!tps])/max(sum(!tps),1)
      return(c(tpr,fpr))
    }))
    
    tpr.quants[i,] = quantile(res[,1],probs = c(0.1,0.5,0.9))
    fpr.quants[i,] = quantile(res[,2],probs = c(0.1,0.5,0.9))
  }
}

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

