# Read LFE catalog and compute ETAS model

library(bayesianETAS)

families = list.files("catalogs")
families = gsub(".txt", "", families)

mu = vector(length=length(families))
K = vector(length=length(families))
alpha = vector(length=length(families))
c = vector(length=length(families))
p = vector(length=length(families))
beta = vector(length=length(families))

for (i in 1:length(families)){
	
    catalog = families[i]

    # Create output directory
    output_dir <- paste("models/", catalog, sep="")
    if (!dir.exists(output_dir)){
        dir.create(output_dir)
    }

    # Read data
    filename = paste("catalogs/", catalog, ".txt", sep="")
    data = read.table(filename, header=TRUE)
    ts = data$time
    magnitudes = data$mw

    # Compute ETAS model
    model = maxLikelihoodETAS(ts, magnitudes, min(magnitudes), max(ts))
    mu[i] = model$params[1]
    K[i] = model$params[2]
    alpha[i] = model$params[3]
    c[i] = model$params[4]
    p[i] = model$params[5]
    beta[i] = model$params[6]

    # Compute posterior distribution
    posterior = sampleETASposterior(ts, magnitudes, min(magnitudes), max(ts), sims=5000)
    X_mu = posterior[, 1]
    X_K = posterior[, 2]
    X_alpha = posterior[, 3]
    X_c = posterior[, 4]
    X_p = posterior[, 5]

    # Estimate density function
    bandwidth <- function(X){
        n = length(X)
        sigma = sqrt(var(X))
        iqr = IQR(X)
        hROT = 0.9 * min(sigma, iqr / 1.34) * n ^ (- 1 / 5)
        if (hROT == 0.0){
        	hROT = 1.0
        }
        return(hROT)
    }
    kernel <- function(X){
        mask = abs(X) <= 1
        f = 1 - abs(X)
        return(f * mask)
    }
    phat <- function(x, X, K, h){
        n = length(X)
        p = (1 / ( n * h)) * sum(K((x - X) / h))
        return(p)
    }
    plot_pdf <- function(X, K, h, phat, label, filename){
	    x = seq(min(X), max(X), length.out=200)
	    y = vector(length=200)
	    for (i in 1:200){
            y[i] = phat(x[i], X, K, h)
        }
        png(filename=filename)
        par(cex.axis=1.5, cex.lab=1.5, cex.main=1.5)
        plot(x, y, xlab=label, ylab="PDF", pch=19, main="Estimated density function")
        dev.off()
    }
    # mu
    h <- bandwidth(X_mu)
    plot_pdf(X_mu, kernel, h, phat, "mu", paste(output_dir, "/mu.png", sep=""))
    # K
    h <- bandwidth(X_K)
    plot_pdf(X_K, kernel, h, phat, "K", paste(output_dir, "/K.png", sep=""))
    # alpha
    h <- bandwidth(X_alpha)
    plot_pdf(X_alpha, kernel, h, phat, "alpha", paste(output_dir, "/alpha.png", sep=""))
    # c
    h <- bandwidth(X_c)
    plot_pdf(X_c, kernel, h, phat, "c", paste(output_dir, "/c.png", sep=""))
    # p
    h <- bandwidth(X_p)
    plot_pdf(X_p, kernel, h, phat, "p", paste(output_dir, "/p.png", sep=""))
}
results = data.frame(families, mu, K, alpha, c, p, beta)
write.table(results, "ETAS.txt", sep="\t", row.names=FALSE, col.names=TRUE) 
