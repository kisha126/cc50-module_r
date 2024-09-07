box::use(
     recipes[recipe],
     parsnip[set_engine, set_mode, tune, fit, rand_forest],
     workflows[add_recipe, add_model, workflow],
     rsample[vfold_cv],
     dials[grid_regular, min_n, mtry],
     tidyr[crossing],
     tune[tune_grid, select_best, finalize_model, finalize_workflow],
     yardstick[metric_set, roc_auc],
     ranger[ranger],
     stats[predict]
)

# Tidymodels' Workflow

train_TM_rf <- function(
          formula, data, new_data = NULL, trees = 500,
          engine = "ranger", importance = "permutation", mode = "classification",
          mtry_range = c(10, 35), mtry_lvl = 5, min_n_range = c(2L, 40L), min_n_lvl = 2
) {
     rf_recipe <- recipe(formula, data = data)
     
     rf_spec <- rand_forest(
          mtry = tune(), 
          trees = trees,
          min_n = tune() 
     ) |>
          set_engine(engine, importance = importance) |>
          set_mode(mode)
     
     rf_workflow <- workflow() |>
          add_recipe(rf_recipe) |>
          add_model(rf_spec)
     
     mtry_grid <- grid_regular(
          mtry(range = mtry_range),
          levels = mtry_lvl
     )
     
     min_n_grid <- grid_regular(
          min_n(range = min_n_range),
          levels = min_n_lvl  
     )
     rf_grid <- crossing(mtry_grid, min_n_grid)
     
     rf_resamples <- rsample::vfold_cv(data)
     
     doParallel::registerDoParallel()
     rf_tune <- tune_grid(
          rf_workflow,
          resamples = rf_resamples,
          grid = rf_grid,
          metrics = metric_set(roc_auc)
     )
     
     best_rf <- select_best(rf_tune, metric = "roc_auc")
     
     final_workflow <- finalize_workflow(rf_workflow, best_rf)
     
     rf_fit <- final_workflow |> 
          fit(data = data)
     
     predict_data <- rf_fit |> 
          predict(new_data = data)
     
     out <- list(
          model = rf_fit,  
          prediction = predict_data$.pred_class
     )
     
     if (!is.null(new_data)) {
          predict_newdata <- rf_fit |> 
               predict(new_data = new_data)
          out$predict_new <- predict_newdata$.pred_class
     }
     out
}



# Just `ranger` workflow

traintest_ranger <- function(formula, data, new_data = NULL, ...) {
     
     model <- ranger(
          formula = formula,
          data = data,
          ...                  
     )
     
     out <- list(
          model = model,
          train_preds = model$predictions
     )
     
     if (!is.null(new_data)) {
          predictions <- predict(model, data = new_data, ...)
          predicted_classes <- predictions$predictions
          out$test_preds <- predicted_classes
     } 
     
     out
}






