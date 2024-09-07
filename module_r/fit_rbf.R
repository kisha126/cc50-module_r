box::use(torch[...])
box::use(./torch_rbf[...])

cc50_dataset <- dataset(
     name = "cc50_dataset",
     initialize = function(df) {
          self$x <- as.matrix(df[, 1:9]) %>% torch_tensor()
          self$y <- torch_tensor(
               as.numeric(df$y)
          )$to(torch_long())
     },
     .getitem = function(i) {
          list(x = self$x[i, ], y = self$y[i])
     },
     .length = function() {
          dim(self$x)[1]
     }
)

#' @export
fit_rbf_nn <- function(train_data, label_name, in_features = 9, num_classes = 2,
                       num_rbf = 30, loss_fn = nn_cross_entropy_loss(),
                       basis_func = "gaussian", epochs = 1000, 
                       lr = 0.001, verbose = TRUE) {
     
     cc50_train_torch <- cc50_dataset(train_data)
     
     # Model definition
     rbf_gauss_model <- RBFNetwork(
          in_features = in_features, 
          num_classes = num_classes, 
          num_rbf = num_rbf, 
          basis_func = basis_func
     )
     
     # Loss function and optimizer
     loss_fn <- loss_fn
     optimizer <- optim_adam(rbf_gauss_model$parameters, lr = lr)
     
     # Training loop
     for (epoch in 1:epochs) {
        optimizer$zero_grad()
        cc50_train_out <- rbf_gauss_model(cc50_train_torch$x)
        loss1_mod <- loss_fn(cc50_train_out, cc50_train_torch$y)
        loss1_mod$backward()
        optimizer$step()
          
        if (verbose) {
              if (epoch %% 100 == 0) {
                   cat("Epoch:", epoch, "Loss:", as.numeric(loss1_mod$item()), "\n")
              }
        }
     }
     
     # Predictions on Train Data
     
     cc50_train_out <- rbf_gauss_model(cc50_train_torch$x)
     cc50_train_pred <- apply(as.array(cc50_train_out), 1, which.max)
     predicted_cc50_train <- as.factor(label_name[cc50_train_pred])
     
     return(
          list(
               model = rbf_gauss_model,
               preds = predicted_cc50_train
          )
     )
}

#' @export
test_predict <- function(object, test_data, label_name, ...) {
     model <- object$model
     
     cc50_test_torch <- cc50_dataset(test_data)
     
     # Predictions on Test Data
     cc50_test_out <- model(cc50_test_torch$x)
     cc50_test_pred <- apply(as.array(cc50_test_out), 1, which.max)
     predicted_cc50_test <- as.factor(label_name[cc50_test_pred])
     
     return(
          list(
               object = object,
               model = model,
               preds = predicted_cc50_test
          )
     )
}

