library(simrel)
library(varbvs)
library(parallel)
num_models = 20

q = 0.25
logodds = log10(q / ( 1 - q))

tpr.quants = matrix(NA,30,3)
fpr.quants = matrix(NA,30,3)
i = 0

tps = c(rep(T,10),rep(F,40))
beta = c(tps*1)

cor = numeric(30)
cor_max = numeric(30)

for(sigma in c(10,1,0.1))
{
  for(j in 10:1)
  {
    i = i + 1
    set.seed(i)
    simres = simrel::unisimrel(n = 1000, p = 50, q = 50, relpos = 1:25, R2 = 0.8, gamma = 0.1*(j-1))
    X = simres$X
    X = scale(X)
    cor_mat = abs(cor(X))
    cor[i] = mean(cor_mat[lower.tri(cor_mat)])
    cor_max[i] = max(cor_mat[lower.tri(cor_mat)])
    
    Y = rnorm(n = nrow(X),beta%*%t(X),sd = sigma)
    Y = (Y > median(Y)) * 1 
    path <-  '/Users/larsskaaret-lund/Documents/Extra results paper 1/sim'
    write.table(X,paste(path,'//X_train',toString(i-1),'.csv',sep = ''), row.names=FALSE,sep = ',',quote = FALSE)
    write.table(Y,paste(path,'//Y_train',toString(i-1),'.csv',sep = ''), row.names=FALSE,sep = ',',quote = FALSE)
    
    print(unique(sort(abs(cor(X)[beta==0,beta!=0]),decreasing = T))[1:10])
    
    res = do.call(rbind,mclapply(1:num_models,FUN = function(x){
      set.seed(x)
      out = varbvs(X,Z = NULL,Y,family = "binomial",logodds = logodds,verbose = FALSE,tol = 1e-6,nr = 10)
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


