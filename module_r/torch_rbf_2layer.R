box::use(torch[...])
box::use(./torch_rbf[...])

#' @export
RBFNetwork2 <- nn_module(
     "RBFNetwork2",
     initialize = function(
          in_features, 
          num_classes, 
          num_rbf1, 
          hidden_size, 
          basis_func1 = 'gaussian'
     ) {
          self$rbf_layer1 <- RBFLayer(in_features, num_rbf1, basis_func1)
          self$hidden_layer <- nn_linear(num_rbf1, hidden_size)  # Replaced RBF with linear layer
          self$output_layer <- nn_linear(hidden_size, num_classes)
     },
     
     forward = function(x) {
          out <- self$rbf_layer1(x)
          out <- nnf_relu(self$hidden_layer(out))  # Activation function added for the linear layer
          self$output_layer(out)
     }
)


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
                       num_rbf1 = 30, hidden_size = 20, loss_fn = nn_cross_entropy_loss(),
                       basis_func1 = "gaussian", epochs = 1000, lr = 0.001, verbose = TRUE) {
     
     # Dataset Preparation
     cc50_train_torch <- cc50_dataset(train_data)
     
     # Model Definition
     rbf_gauss_model <- RBFNetwork2(
          in_features = in_features, 
          num_classes = num_classes, 
          num_rbf1 = num_rbf1, 
          hidden_size = hidden_size, 
          basis_func1 = basis_func1
     )
     
     # Loss function and optimizer
     optimizer <- optim_adam(rbf_gauss_model$parameters, lr = lr)
     
     # Training Loop
     for (epoch in 1:epochs) {
          optimizer$zero_grad()
          output <- rbf_gauss_model(cc50_train_torch$x)
          loss <- loss_fn(output, cc50_train_torch$y)
          loss$backward()
          optimizer$step()
          
          if (verbose && epoch %% 100 == 0) {
               cat("Epoch:", epoch, "Loss:", as.numeric(loss$item()), "\n")
          }
     }
     
     # Predictions on Train Data
     output <- rbf_gauss_model(cc50_train_torch$x)
     predicted_labels <- apply(as.array(output), 1, which.max)
     predicted_classes <- as.factor(label_name[predicted_labels])
     
     return(
          list(
               model = rbf_gauss_model,
               preds = predicted_classes
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


