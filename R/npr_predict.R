predict.FuNopaRe <- function(object, newdata, method.params, Bootstrapping = FALSE, ...) {
  
  Dist <- Semimetric (object$X.learn, newdata, object$Semimetric, object$semimetric.params)
  DistMat <- Dist$semimetric
  if (object$Method == "KernelPredictionCV") {   
    Y <- .Call ("KernelPrediction", 
                DistMat, 
                object$Y.learn, 
                object$h.opt, 
                PACKAGE = "nfda")  
    object$Prediction <- Y
  } else if (object$Method == "KernelPredictionkNNgCV") {   
      Y <- .Call ("KernelPredictionkNN", 
                    DistMat, 
                    object$Y.learn, 
                    object$k.opt,
                    FALSE,
                    PACKAGE = "nfda")
      object$Prediction <- Y
  } else if (object$Method == "KernelPredictionkNNlCV") {   
      Y <- .Call ("KernelPredictionkNN", 
                    DistMat, 
                    object$Y.learn, 
                    object$k.opt,
                    TRUE,
                    PACKAGE = "nfda")
      object$Prediction <- Y
  }
  if (Bootstrapping == TRUE) {
    R <- .Call ("KernelPredictionBoot", 
                DistMat,
                object$Y.learn, 
                object$Y.hat,
                Y, 
                method.params$NB, 
                method.params$neighbours,
                PACKAGE = "nfda")
    object$Prediction <- R$pred
  }
  object
}