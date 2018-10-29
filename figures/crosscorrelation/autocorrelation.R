dt = 0.01
T1 = 14.0
n1 = floor(T1 / dt)
ampn = 1

set.seed(1)
t1 = dt * c(0 : n1)
fnoise = ampn * rnorm(1 + n1)
f1 = 0.5 * (1 + cos(0.5 * pi * (t1 - 1.0)))
f1[t1 <= 3 | t1 >= 7] = 0.0
f2 = 0.5 * (1 + cos(0.5 * pi * (t1 - 4.5)))
f2[t1 <= 6.5 | t1 >= 10.5] = 0.0
finit = f1 + f2
f = finit + fnoise

t = dt * c(- n1 : n1)
ff = rep(0, 1 + n1 + n1)
for (i in 1 : (1 + n1 + n1)) {
    fsub = c(rep(0, n1), f, rep(0, n1))
    if (i == 1) {
        gsub = c(rep(0, 1 - i + n1 + n1), f)
    } else if (i == 1 + n1 + n1) {
        gsub = c(f, rep(0, i - 1))
    } else {
        gsub = c(rep(0, 1 - i + n1 + n1), f, rep(0, i - 1))
    }
    ff[i] = sum(fsub * gsub) * dt
}
filename <- "/Users/ariane/Documents/ResearchProject/ETSscripts/figures/crosscorrelation/seismicwaves_ac.eps"
postscript(filename, width=21, height=5, horizontal=FALSE)
oldpar <- par(mfrow=c(1,2))

par(cex.axis=2, cex.lab=3, cex.main=1.5, mfg=c(1, 1, 1, 2), lwd=2, mar=c(2, 3, 5, 1) + 0.1)
plot(t1 - 3, finit, xlim=c(-1.5, 9), ylim=c(-1, 2), xlab="", ylab="", type="l", lwd=2)
par(new=TRUE)
plot(c(2, 2), c(-2, 3), xlim=c(-1.5, 9), ylim=c(-1, 2), xlab="", ylab="", type="l", lwd=2, col="red")
text(0.5, 1.7, labels="t = 2", cex = 2.2, col="red")
par(new=TRUE)
plot(c(5.5, 5.5), c(-2, 3), xlim=c(-1.5, 9), ylim=c(-1, 2), xlab="", ylab="", type="l", lwd=2, col="red")
text(7.5, 1.7, labels="t = 5.5", cex = 2.2, col="red")
title(main="Direct wave + Reflected wave")

par(cex.axis=2, cex.lab=3, cex.main=1.5, mfg=c(1, 2, 1, 2), lwd=2, mar=c(2, 3, 5, 1) + 0.1)
plot(t1 - 3, f, xlim=c(-1.5, 9), ylim=c(-1, 2), xlab="", ylab="", type="l", lwd=2)
title(main="Signal + noise")

dev.off()

filename <- "/Users/ariane/Documents/ResearchProject/ETSscripts/figures/crosscorrelation/autocorrelation.eps"
postscript(filename, width=21, height=5, horizontal=FALSE)

par(cex.axis=2, cex.lab=3, cex.main=2, lwd=2, mar=c(2, 3, 5, 1) + 0.1)
plot(t, ff, xlim=c(-1, 8), ylim=c(-1, 8), xlab="", ylab="", type="l", lwd=2)
par(new=TRUE)
plot(c(3.5, 3.5), c(-2, 9), xlim=c(-1, 8), ylim=c(-1, 8), xlab="", ylab="", type="l", lwd=2, col="red")
text(4.5, 5, labels="t = 3.5", cex = 2.2, col="red")
par(new=TRUE)
plot(c(0, 0), c(-2, 9), xlim=c(-1, 8), ylim=c(-1, 8), xlab="", ylab="", type="l", lwd=2, col="red")
text(1, 5, labels="t = 0", cex = 2.2, col="red")
title(main="Autocorrelation")

dev.off()
