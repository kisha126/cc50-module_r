box::use(torch[...])

basis_func_dict <- function() {
     list(
          gaussian = gaussian,
          linear = linear,
          quadratic = quadratic,
          inverse_quadratic = inverse_quadratic,
          multiquadric = multiquadric,
          inverse_multiquadric = inverse_multiquadric,
          spline = spline,
          poisson_one = poisson_one,
          poisson_two = poisson_two,
          matern32 = matern32,
          matern52 = matern52
     )
}

#' @export
RBFLayer <- nn_module(
     "RBFLayer",
     initialize = function(in_features, out_features, basis_func = 'gaussian') {
          self$in_features <- in_features
          self$out_features <- out_features
          self$centres <- nn_parameter(torch_randn(out_features, in_features))
          self$log_sigmas <- nn_parameter(torch_zeros(out_features))
          self$basis_func <- basis_func_dict()[[basis_func]]
     },
     
     forward = function(input) {
          size <- c(input$size(1), self$out_features, self$in_features)
          x <- input$unsqueeze(2)$expand(size)
          c <- self$centres$unsqueeze(1)$expand(size)
          distances <- (x - c)$pow(2)$sum(3)$sqrt() / torch_exp(self$log_sigmas)$unsqueeze(1)
          self$basis_func(distances)
     }
)

#' @export
RBFNetwork <- nn_module(
     "RBFNetwork",
     initialize = function(in_features, num_classes, num_rbf, basis_func = 'gaussian') {
          self$rbf_layer <- RBFLayer(in_features, num_rbf, basis_func)
          self$linear <- nn_linear(num_rbf, num_classes)
     },

     forward = function(x) {
          out <- self$rbf_layer(x)
          self$linear(out)
     }
)
# RBFNetwork <- nn_module(
#      "RBFNetwork",
#      initialize = function(in_features, num_rbf, basis_func = 'gaussian') {
#           self$rbf_layer <- RBFLayer(in_features, num_rbf, basis_func)
#           self$linear <- nn_linear(num_rbf, 1)
#      },
#      
#      forward = function(x) {
#           out <- self$rbf_layer(x)
#           self$linear(out)$squeeze(2)  
#      }
# )


gaussian <- function(alpha) {
     torch_exp(-1 * alpha$pow(2))
}

linear <- function(alpha) {
     alpha
}

quadratic <- function(alpha) {
     alpha$pow(2)
}

inverse_quadratic <- function(alpha) {
     torch_ones_like(alpha) / (torch_ones_like(alpha) + alpha$pow(2))
}

multiquadric <- function(alpha) {
     (torch_ones_like(alpha) + alpha$pow(2))$pow(0.5)
}

inverse_multiquadric <- function(alpha) {
     torch_ones_like(alpha) / (torch_ones_like(alpha) + alpha$pow(2))$pow(0.5)
}

spline <- function(alpha) {
     alpha$pow(2) * torch_log(alpha + torch_ones_like(alpha))
}

poisson_one <- function(alpha) {
     (alpha - torch_ones_like(alpha)) * torch_exp(-alpha)
}

poisson_two <- function(alpha) {
     ((alpha - 2 * torch_ones_like(alpha)) / (2 * torch_ones_like(alpha))) * alpha * torch_exp(-alpha)
}

matern32 <- function(alpha) {
     (torch_ones_like(alpha) + 3^0.5 * alpha) * torch_exp(-3^0.5 * alpha)
}

matern52 <- function(alpha) {
     (torch_ones_like(alpha) + 5^0.5 * alpha + (5/3) * alpha$pow(2)) * torch_exp(-5^0.5 * alpha)
}
