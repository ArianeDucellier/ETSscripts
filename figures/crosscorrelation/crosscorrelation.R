dt = 0.01
T1 = 10.0
T2 = 18.0
n1 = floor(T1 / dt)
n2 = floor(T2 / dt)
ampn = 1.0

set.seed(1)
t1 = dt * c(0 : n1)
fnoise = ampn * rnorm(1 + n1)
finit = 0.5 * (1 + cos(0.5 * pi * (t1 - 1.0)))
finit[t1 <= 3 | t1 >= 7] = 0.0
f = finit + fnoise

t2 = dt * c(0 : n2)
gnoise = ampn * rnorm(1 + n2)
ginit = 0.5 * (1 + cos(0.5 * pi * (t2 - 2.5)))
ginit[t2 <= 4.5 | t2 >= 8.5] = 0.0
g = ginit + gnoise

t = dt * c(- n1 : n2)
fg = rep(0, 1 + n1 + n2)
for (i in 1 : (1 + n1 + n2)) {
    fsub = c(rep(0, n2), f, rep(0, n2))
    if (i == 1) {
        gsub = c(rep(0, 1 - i + n1 + n2), g)
    } else if (i == 1 + n1 + n2) {
        gsub = c(g, rep(0, i - 1))
    } else {
        gsub = c(rep(0, 1 - i + n1 + n2), g, rep(0, i - 1))
    }
    fg[i] = sum(fsub * gsub) * dt
}
filename <- "/Users/ariane/Documents/ResearchProject/ETSscripts/figures/crosscorrelation/seismicwaves_cc.eps"
postscript(filename, width=21, height=5, horizontal=FALSE)
oldpar <- par(mfrow=c(2,2))

par(cex.axis=2, cex.lab=3, cex.main=1.5, mfg=c(1, 1, 2, 2), lwd=2, mar=c(2, 3, 5, 1) + 0.1)
plot(t1 - 3, finit, xlim=c(-1.5, 7), ylim=c(-1, 2), xlab="", ylab="", type="l", lwd=2)
par(new=TRUE)
plot(c(0, 0), c(-2, 3), xlim=c(-1.5, 7), ylim=c(-1, 2), xlab="", ylab="", type="l", lwd=2, col="red")
text(1.5, 1.7, labels="t = 0", cex = 2.2, col="red")
title(main="Vertical component (P-wave)")

par(cex.axis=2, cex.lab=3, cex.main=1.5, mfg=c(1, 2, 2, 2), lwd=2, mar=c(2, 3, 5, 1) + 0.1)
plot(t1 - 3, f, xlim=c(-1.5, 7), ylim=c(-1, 2), xlab="", ylab="", type="l", lwd=2)
title(main="Vertical + noise")

par(cex.axis=2, cex.lab=3, cex.main=1.5, mfg=c(2, 1, 2, 2), lwd=2, mar=c(2, 3, 5, 1) + 0.1)
plot(t2 - 3, ginit, xlim=c(-1.5, 7), ylim=c(-1, 2), xlab="", ylab="", type="l", lwd=2)
par(new=TRUE)
plot(c(1.5, 1.5), c(-2, 3), xlim=c(-1.5, 7), ylim=c(-1, 2), xlab="", ylab="", type="l", lwd=2, col="red")
text(3, 1.7, labels="t = 1.5", cex = 2.2, col="red")
title(main="Horizontal component (S-wave)")

par(cex.axis=2, cex.lab=3, cex.main=1.5, mfg=c(2, 2, 2, 2), lwd=2, mar=c(2, 3, 5, 1) + 0.1)
plot(t2 - 3, g, xlim=c(-1.5, 7), ylim=c(-1, 2), xlab="", ylab="", type="l", lwd=2)
title(main="Horizontal + noise")

dev.off()

filename <- "/Users/ariane/Documents/ResearchProject/ETSscripts/figures/crosscorrelation/crosscorrelation.eps"
postscript(filename, width=21, height=5, horizontal=FALSE)

par(cex.axis=2, cex.lab=3, cex.main=2, lwd=2, mar=c(2, 3, 5, 1) + 0.1)
plot(t, fg, xlim=c(-3.5, 6.5), ylim=c(-1.5, 2.5), xlab="", ylab="", type="l", lwd=2)
par(new=TRUE)
plot(c(1.5, 1.5), c(-2, 3), xlim=c(-3.5, 6.5), ylim=c(-1.5, 2.5), xlab="", ylab="", type="l", lwd=2, col="red")
text(2.5, -1, labels="t = 1.5", cex = 2.2, col="red")
title(main="Vertical * Horizontal")

dev.off()
