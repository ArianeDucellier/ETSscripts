# This script write event time series for the following process:
# - Homogeneous Poisson process
# - ETAS model
# - Fractional ARIMA

library(bayesianETAS)
library(fracdiff)

N = 3645 # Number of time windows

# Homogeneous Poisson process
lambda = 0.2

set.seed(1)
ts <- cumsum(-log(runif(N)) / lambda)
nb = rep(0, N)
for (i in 0:N){
    for (j in 1:length(ts)){
        if (ts[j] >= i & ts[j] < i + 1){
            nb[i] = nb[i] + 1
        }
    }
}
write.table(nb, file="poisson.txt", row.names=FALSE, col.names=FALSE)
print(c("Poisson", mean(nb), sqrt(var(nb))))

# ETAS model
mu = 0.2
K = 0.2
alpha = 1.5
c = 0.5
p = 2
beta = 2.4
M0 = 3

set.seed(1)
data <- simulateETAS(mu, K, alpha, c, p, beta, M0, T=N, displayOutput=FALSE)
nb = rep(0, N)
for (i in 0:N){
    for (j in 1:length(data$ts)){
        if (data$ts[j] >= i & data$ts[j] < i + 1){
            nb[i] = nb[i] + 1
        }
    }
}
write.table(nb, file="ETAS.txt", row.names=FALSE, col.names=FALSE)
print(c("ETAS", mean(nb), sqrt(var(nb))))

# Fractional ARIMA
d = 0.3
rand.gen = rnorm
mu = 0.2

set.seed(1)
data = fracdiff.sim(n=N, d=d, rand.gen=rand.gen, mu=mu)
write.table(data$series, file="FARIMA.txt", row.names=FALSE, col.names=FALSE)
print(c("FARIMA", mean(data$series), sqrt(var(data$series))))
