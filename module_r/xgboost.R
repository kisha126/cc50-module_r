box::use(
     recipes[recipe, update],
     parsnip[set_engine, set_mode, tune, fit, boost_tree],
     workflows[add_recipe, add_model, workflow],
     rsample[vfold_cv],
     dials[grid_random, min_n, tree_depth, learn_rate, sample_size],
     tidyr[crossing],
     tune[tune_grid, select_best, finalize_model, finalize_workflow, parameters],
     yardstick[metric_set, roc_auc],
     xgboost[xgboost, xgb.DMatrix],
     stats[predict, as.formula, model.frame]
)

# With tidymodels

train_TM_xgb <- function(
          formula, data, new_data = NULL,
          trees = 500,
          mode = "classification",
          loss_reduction = tune(),
          learn_rate_range = c(1e-3, 0.1), learn_rate_lvl = 3,
          tree_depth_range = c(3, 10), tree_depth_lvl = 3,
          min_n_range = c(2L, 40L), min_n_lvl = 3,
          loss_reduction_range = c(-10, 1), loss_reduction_lvl = 3,
          sample_size_range = c(0.5, 1), sample_size_lvl = 3, 
          v = 3,
          n_iter = 10
) {
     xgb_recipe <- recipe(formula, data = data)
     
     xgb_spec <- boost_tree(
          trees = trees,
          learn_rate = tune(),
          tree_depth = tune(),
          min_n = tune()
     ) |>
          set_engine("xgboost") |>
          set_mode(mode)
     
     xgb_workflow <- workflow() |>
          add_recipe(xgb_recipe) |>
          add_model(xgb_spec)
     
     xgb_grid <- grid_random(
          tree_depth(range = tree_depth_range),
          learn_rate(range = learn_rate_range),
          min_n(range = min_n_range),
          size = n_iter
     )
     
     xgb_resamples <- rsample::vfold_cv(data, v = v, strata = y)
     
     num_cores <- parallel::detectCores() - 1
     doParallel::registerDoParallel(cores = num_cores)
     
     xgb_tune <- tune_grid(
          xgb_workflow,
          resamples = xgb_resamples,
          grid = xgb_grid,
          metrics = metric_set(roc_auc)
     )
     
     best_xgb <- select_best(xgb_tune, metric = "roc_auc")
     final_xgb <- finalize_model(xgb_spec, best_xgb)
     
     xgb_fit <- final_xgb |> 
          fit(formula, data = data)
     
     xgb_prediction <- xgb_fit |> 
          predict(new_data = data)
     
     doParallel::stopImplicitCluster()
     
     out <- list(
          model = xgb_fit,  
          prediction = xgb_prediction$.pred_class
     )
     
     if (!is.null(new_data)) {
          xgb_prediction_new <- xgb_fit |> 
               predict(new_data = new_data)
          out$predict_new <- xgb_prediction_new$.pred_class
     }
     
     out
}

# Without tidymodels

traintest_xgb <- function(formula, data, new_data = NULL, nr = 100, verbose = F, ...) {
     f <- as.formula(formula)
     
     dat <- model.frame(f, data)
     x <- as.matrix(dat[-1])  
     y <- dat[[1]] 
     
     xgb_dat <- xgb.DMatrix(data = x, label = as.numeric(y) - 1)
     
     params <- list(...)
     
     xgb_model <- xgboost(
          params = params,
          data = xgb_dat,
          nrounds = nr,
          objective = "multi:softmax",  
          num_class = length(unique(y)),
          verbose = verbose
     )
     
     pred_train <- predict(xgb_model, xgb_dat)
     predictions <- factor(pred_train, levels = 0:(length(unique(y)) - 1), labels = levels(y))
     
     predictions_new <- NULL
     
     if (!is.null(new_data)) {
          new_dat <- model.frame(f, new_data)
          x_new <- as.matrix(new_dat[-1])
          
          xgb_new <- xgb.DMatrix(data = x_new)
          
          pred_new <- predict(xgb_model, xgb_new)
          predictions_new <- factor(pred_new, levels = 0:(length(unique(y)) - 1), labels = levels(y))
     }
     
     list(
          model = xgb_model,
          predictions = predictions,
          predictions_new = predictions_new
     )
}
