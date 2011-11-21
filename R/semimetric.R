Semimetric <- function(Data1, Data2, semimetric, semimetric.params) {
  
  if (semimetric == "SemimetricDeriv") {
    Dist <- SemimetricDeriv (Data1, 
                             Data2, 
                             q = semimetric.params$q, 
                             nknot = semimetric.params$nknot, 
                             range.grid = semimetric.params$range.grid,
                             Hhalf = semimetric.params$HhalfDeriv)
  } else if (semimetric == "SemimetricPCA") {
    Dist <- SemimetricPCA (Data1, 
                           Data2, 
                           semimetric.params$q,
                           semimetric.params$EigenVec)
  } else {
    
  }
  return(Dist)
}

################################################################################
#
# Semimetric based on the L_2-norm of the qth derivatives
#
################################################################################
SemimetricDeriv <- function (Data1, Data2, 
                             q = 2, nknot = 20, range.grid = c(0,1), 
                             Hhalf = NULL) {
    
    library (splines)
    if (is.vector(Data1)) Data1 <- as.matrix(t(Data1))
    if (is.vector(Data2)) Data2 <- as.matrix(t(Data2))
    testfordim <- sum (dim (Data1) == dim (Data2)) == 2
    twodatasets <- 1
    if (testfordim) twodatasets <- sum (Data1 == Data2) != prod (dim (Data1))
    #####################################################################
    # B-spline approximation of the curves containing in DATASET
    #####################################################################
    p <- ncol (Data1)
    a <- range.grid[1]
    b <- range.grid[2]
    x <- seq (a, b, length = p)
    order.Bspline <- q + 3
    nknotmax <- (p - order.Bspline - 1) %/% 2
    if (nknot > nknotmax) {
      stop (paste ("give a number nknot smaller than ", 
                   nknotmax, " for avoiding ill-conditioned matrix"))
    }
    Knot <- seq (a, b, length = nknot + 2)[ - c(1, nknot + 2)]
    delta <- sort (c(rep(c(a, b), order.Bspline), Knot))
    Bspline <- splineDesign (delta, x, order.Bspline)
    #######################################################################
    # Numerical integration by the Gauss method 
    #######################################################################   
    point.gauss <- c(-0.9324695142, -0.6612093865, -0.2386191861, 
                     0.2386191861, 0.6612093865, 0.9324695142)
    weight.gauss <- c(0.1713244924, 0.360761573, 0.4679139346, 
                      0.4679139346, 0.360761573, 0.1713244924)
    x.gauss <- 0.5 * ((b + a) + (b - a) * point.gauss)
    lx.gauss <- length(x.gauss)
    
    Bspline.deriv <- splineDesign (delta, 
                                   x.gauss, 
                                   order.Bspline, 
                                   rep(q, lx.gauss))
    if (is.atomic(Hhalf)) {
    Hhalf <- .Call ("SemimetricDerivDesign", 
                    Bspline.deriv, 
                    weight.gauss, 
                    b - a, 
                    PACKAGE = "nfda")
    }
    if (twodatasets) {      
      semimetric <- .Call ("SemimetricDeriv", 
                           Hhalf, Bspline, 
                           Data1, Data2,
                           twodatasets,
                           PACKAGE = "nfda")
    } else {    
      semimetric <- .Call ("SemimetricDeriv", 
                           Hhalf, Bspline, 
                           Data1, Data1,
                           twodatasets,
                           PACKAGE = "nfda")
    }
    return(list(semimetric = semimetric, Hhalf = Hhalf))
}
################################################################################
#
# Semimetric based on Fourier coefficients
#
################################################################################



################################################################################
#
# Semimetric based on PCA
#
################################################################################
SemimetricPCA <- function (Data1, Data2, q, EigenVec = NULL) {
  
  if (is.vector(Data1)) Data1 <- as.matrix(t(Data1))
  if (is.vector(Data2)) Data2 <- as.matrix(t(Data2))
	testfordim <- sum(dim(Data1)==dim(Data2))==2
	twodatasets <- 1
	if (testfordim) twodatasets <- sum(Data1==Data2)!=prod(dim(Data1))
	qmax <- ncol(Data1)
	if (q > qmax) stop(paste("give a integer q smaller than ", qmax))
  if (is.atomic(EigenVec)) {
      EigenVec <- .Call ("SemimetricPCAEV", 
                          Data1,
                          q, 
                          PACKAGE = "nfda")
  }
  if (twodatasets) {
    semimetric <- .Call ("SemimetricPCA",  
                         EigenVec,
                         Data1, Data2,
                         twodatasets,
                         PACKAGE = "nfda")
  } else {
    semimetric <- .Call ("SemimetricPCA",  
                         EigenVec,
                         Data1, Data1,
                         twodatasets,
                         PACKAGE = "nfda")
  }
  return(list(semimetric = semimetric, EigenVec = EigenVec))
}
