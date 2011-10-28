#ifndef _nfda_RCPP_NFDA_H
#define _nfda_RCPP_NFDA_H

#include <RcppArmadillo.h>

RcppExport SEXP KernelPrediction(SEXP DistanceMatrix, SEXP response, SEXP bandwidth);
RcppExport SEXP KernelPredictionCV(SEXP DistanceMatrix, SEXP response, SEXP bandwidth);

RcppExport SEXP KernelPredictionBoot(SEXP DistanceMatrixPred, SEXP YLearn, SEXP Yhat, SEXP predresponse, SEXP BootRep, SEXP neighbours);

RcppExport SEXP KernelPredictionkNN(SEXP DistanceMatrix, SEXP response, SEXP neighbour, SEXP local);
RcppExport SEXP KernelPredictionkNNgCV(SEXP DistanceMatrix, SEXP response, SEXP neighbour);
RcppExport SEXP KernelPredictionkNNlCV(SEXP DistanceMatrix, SEXP response, SEXP knnlen);
#endif