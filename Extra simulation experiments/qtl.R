

library(qtl)

set.seed(1)

map = sim.map(c(100,90,80,60,40), 50/5, include.x=FALSE, eq.spacing=TRUE)
#plotMap(map)

# The longer the chromosomal length the less correlated are markers
# (at least theoretically)

# Now simulate data from a backcross design using the sim.cross function
n.ind = 1000   #sample size

simbc = sim.cross(map, type="bc", n.ind=n.ind,
                  model=rbind(c(1,45,1), c(5,20,1), c(5,50,1)))

# The relevant genotype data is in the structure geno 
str(simbc$geno)   #it is a bit complicated

# Get an X matrix for each chromosome
X.list = vector("list", 5)
for (chr in 1:5){
  X.list[[chr]] = pull.geno(simbc, chr)
}

#Check the correlations between markers within the same chromosome
lapply(X.list, cor)


#Create one large X matrix which you can the use to make your own 
# simulations of Y data with a logic regression model
X = cbind(X.list[[1]],X.list[[2]],X.list[[3]],X.list[[4]],X.list[[5]])-1
#permute elements of X
X = X[,sample.int(n = 50,size = 50,replace = F)]
X2 = as.data.frame(X)
library(varbvs)
num_models = 20

q = 0.25
logodds = log10(q / ( 1 - q))

tpr.quants = matrix(NA,30,3)
fpr.quants = matrix(NA,30,3)
i = 0
cor = numeric(30)
cor_max = numeric(30)
set.seed(1)
tps = runif(50,0,1) > 0.75
beta = c(tps*rnorm(10,ifelse(runif(10>0.5),1,-1),1))
for(sigma in c(1,0.1,0.01))
{
  for(j in c(1:10)/10)
  {
    i = i + 1
    set.seed(i)
    X = X2
    for(k in 1:4)
      X[,(k*10+1):((k+1)*10)] = j*X[,(k*10+1):((k+1)*10)] + (1-j)*X[,((k-1)*10+1):(k*10)] 
    X = scale(X)
    cor_mat = abs(cor(X))
    cor[i] = mean(cor_mat[lower.tri(cor_mat)])
    cor_max[i] = max(cor_mat[lower.tri(cor_mat)])
    
    nu = rnorm(n = nrow(X),beta%*%t(X),sd = sigma)
    
    Y = rbinom(1000,size = 1,prob = 1/(1+exp(-nu)))
    path <-  '/Users/larsskaaret-lund/Documents/Extra results paper 1/sim'
    write.table(X,paste(path,'//X_train_logit2',toString(i-1),'.csv',sep = ''), row.names=FALSE,sep = ',',quote = FALSE)
    write.table(Y,paste(path,'//Y_train_logit2',toString(i-1),'.csv',sep = ''), row.names=FALSE,sep = ',',quote = FALSE)
    
    print(unique(sort(abs(cor(X)[beta==0,beta!=0]),decreasing = T))[1:10])
    
  #  res = do.call(rbind,mclapply(1:num_models,FUN = function(x){
  #    set.seed(x)
  #    out = varbvs(X,Z = NULL,Y,family = "binomial",logodds = logodds,verbose = FALSE,tol = 1e-6,nr = 1000)
  #    alpha = (as.numeric(out$alpha) >  0.5 ) * 1 #median probability model
  #    tpr = sum(alpha[tps])/sum(tps)
   #   fpr = sum(alpha[!tps])/max(sum(!tps),1)
    #  return(c(tpr,fpr))
    #}))
    
   # tpr.quants[i,] = quantile(res[,1],probs = c(0.1,0.5,0.9))
  #  fpr.quants[i,] = quantile(res[,2],probs = c(0.1,0.5,0.9))
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


#write.table(tpr.quants,'/Users/larsskaaret-lund/Documents/Extra results paper 1/sim/tpr_logits2.csv',sep = ',',row.names = FALSE,col.names = FALSE)
#write.table(fpr.quants,'/Users/larsskaaret-lund/Documents/Extra results paper 1/sim/fpr_logits2.csv',sep = ',',row.names = FALSE,col.names = FALSE)
write.table(cor,'/Users/larsskaaret-lund/Documents/Extra results paper 1/sim/cov_logits2.csv',sep = '',row.names = FALSE,col.names = FALSE)
write.table(beta,'/Users/larsskaaret-lund/Documents/Extra results paper 1/sim/beta_logits2.csv',sep = '',row.names = FALSE,col.names = FALSE)
write.table(cor_max,'/Users/larsskaaret-lund/Documents/Extra results paper 1/sim/cov_max_logits2.csv',sep = '',row.names = FALSE,col.names = FALSE)
