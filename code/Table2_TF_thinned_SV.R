suppressPackageStartupMessages({
  library(data.table)
  library(GpGp)
})

ROOT <- normalizePath(getwd())
if (!dir.exists(file.path(ROOT, "data"))) {
  ROOT <- normalizePath(file.path(ROOT, ".."))
}

DATA_DIR    <- file.path(ROOT, "data")
PROCESSED_DIR <- file.path(DATA_DIR, "processed_data")
RESULTS_DIR <- file.path(ROOT, "results", "intermediate")
CODE_DIR    <- file.path(ROOT, "code")

dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

source(file.path(CODE_DIR, "thinnedsv_source.R"))

DONOR_FILE     <- file.path(PROCESSED_DIR, "matching_weighted_ks.csv")
OUTPUT_DETAIL  <- file.path(RESULTS_DIR,   "Table2_TF_thinnedSV_detail.csv")
OUTPUT_SUMMARY <- file.path(RESULTS_DIR,   "Table2_TF_thinnedSV_summary.csv")

TARGET             <- "power"
BASE_FEATURES      <- c("wind_speed", "temperature", "turbulence_intensity", "std_wind_direction")
ANGLE_FEATURE      <- "wind_direction"
K                  <- 7L
SEED               <- 2026L
MAX_THINNING_NUMBER <- 20L
YEARS_TEST         <- c(2017L, 2018L)

mae_vec  <- function(y, yhat) mean(abs(y - yhat))
rmse_vec <- function(y, yhat) sqrt(mean((y - yhat)^2))

feature_names <- function() c(BASE_FEATURES, "wind_direction_sin", "wind_direction_cos")

load_turbine_year <- function(turbine_id, year) {
  path <- file.path(DATA_DIR, sprintf("Turbine%d_%d.csv", as.integer(turbine_id), as.integer(year)))
  if (!file.exists(path)) stop("Missing file: ", path)
  dt   <- fread(path, showProgress = FALSE)
  need <- c(BASE_FEATURES, ANGLE_FEATURE, TARGET)
  miss <- setdiff(need, names(dt))
  if (length(miss)) stop("Missing columns: ", paste(miss, collapse = ", "))
  dt <- dt[, ..need]
  dt[, (need) := lapply(.SD, as.numeric), .SDcols = need]
  dt <- na.omit(dt)
  if (!nrow(dt)) stop("No usable rows in ", path)
  rad <- dt[[ANGLE_FEATURE]] * pi / 180
  dt[, wind_direction_sin := sin(rad)]
  dt[, wind_direction_cos := cos(rad)]
  dt[, (ANGLE_FEATURE) := NULL]
  dt
}

read_donor_table <- function(path) {
  dt <- fread(path, showProgress = FALSE)
  if (names(dt)[1] != "target") setnames(dt, 1L, "target")
  donor_cols <- grep("^donor", names(dt), value = TRUE)
  if (!length(donor_cols)) stop("No donor columns found in ", path)
  ord <- order(as.integer(gsub("^donor", "", donor_cols)))
  donor_cols <- donor_cols[ord]
  dt[, target := as.integer(target)]
  dt[, (donor_cols) := lapply(.SD, as.integer), .SDcols = donor_cols]
  list(dt = dt, donor_cols = donor_cols)
}

compute_thinning_number <- function(trainX, max_thinning_number) {
  n <- nrow(trainX)
  if (n < 5L) return(1L)
  thinning_vec <- rep(max_thinning_number, ncol(trainX))
  for (j in seq_len(ncol(trainX))) {
    pacf_vals <- tryCatch(
      stats::pacf(trainX[, j], plot = FALSE, lag.max = max_thinning_number)$acf[, 1, 1],
      error = function(e) rep(0, max_thinning_number)
    )
    thresh <- 2 / sqrt(n)
    idx <- which(c(1, abs(pacf_vals)) <= thresh)
    if (length(idx)) thinning_vec[j] <- min(idx)
  }
  max(1L, max(thinning_vec))
}

create_thinned_bins <- function(dataX, dataY, thinning_number) {
  n    <- nrow(dataX)
  bins <- vector("list", thinning_number)
  if (thinning_number < 2L) {
    bins[[1]] <- list(X = dataX, y = dataY)
    return(bins)
  }
  for (i in seq_len(thinning_number)) {
    n_points <- floor((n - i) / thinning_number)
    last_idx <- i + n_points * thinning_number
    idx      <- seq(i, last_idx, length.out = n_points + 1L)
    bins[[i]] <- list(X = dataX[idx, , drop = FALSE], y = dataY[idx])
  }
  bins
}

# --- fit model on donor 2017 data, return fit_obj + metadata (NO prediction yet) ---
fit_donor_model <- function(donor_id) {
  feats     <- feature_names()
  train_dt  <- load_turbine_year(donor_id, 2017L)
  x_train   <- as.matrix(train_dt[, ..feats])
  y_train   <- train_dt[[TARGET]]
  T_use     <- compute_thinning_number(x_train, MAX_THINNING_NUMBER)
  bins      <- create_thinned_bins(x_train, y_train, T_use)
  
  set.seed(SEED)
  t0      <- proc.time()[["elapsed"]]
  fit_obj <- fit_scaled_thinned(
    y          = y_train,
    inputs     = x_train,
    thinnedBins = bins,
    T          = T_use,
    ms         = 30
  )
  fit_time <- proc.time()[["elapsed"]] - t0
  
  list(fit_obj = fit_obj, T_use = T_use, fit_time = fit_time)
}

# --- predict for a given test set using an already-fitted model ---
predict_donor_model <- function(fitted, target_id, test_year) {
  feats   <- feature_names()
  test_dt <- load_turbine_year(target_id, test_year)
  x_test  <- as.matrix(test_dt[, ..feats])
  y_test  <- test_dt[[TARGET]]
  
  t0   <- proc.time()[["elapsed"]]
  pred <- predictions_scaled_thinned(
    fitted$fit_obj,
    locs_pred = x_test,
    m         = 200,
    joint     = TRUE,
    nsims     = 0,
    predvar   = FALSE,
    scale     = "parms"
  )
  pred_time <- proc.time()[["elapsed"]] - t0
  
  list(
    pred      = pred,
    actual    = y_test,
    rmse      = rmse_vec(y_test, pred),
    mae       = mae_vec(y_test, pred),
    pred_time = pred_time
  )
}

run_metric <- function() {
  donor_obj  <- read_donor_table(DONOR_FILE)
  donor_dt   <- donor_obj$dt
  donor_cols <- donor_obj$donor_cols
  targets    <- sort(unique(donor_dt$target))
  n_targets  <- length(targets)
  
  # --- resume: skip already-done (target, year) pairs ---
  done_keys <- character(0)
  if (file.exists(OUTPUT_DETAIL) && file.size(OUTPUT_DETAIL) > 0) {
    prev <- tryCatch(fread(OUTPUT_DETAIL, showProgress = FALSE), error = function(e) data.table())
    if (nrow(prev) && all(c("target", "year") %in% names(prev))) {
      done_keys <- unique(sprintf("%d|%d", as.integer(prev$target), as.integer(prev$year)))
      cat(sprintf("[INFO] Resuming — %d pairs already done.\n", length(done_keys)))
    }
  }
  
  cat(sprintf("[INFO] Starting TF_thinnedSV — %d targets x %d years\n",
              n_targets, length(YEARS_TEST)))
  
  for (t_idx in seq_along(targets)) {
    target_id <- targets[t_idx]
    
    donor_row <- donor_dt[target == target_id]
    donors    <- as.integer(na.omit(unlist(donor_row[, ..donor_cols])))
    donors    <- unique(donors[donors != target_id])
    donors    <- donors[seq_len(min(K, length(donors)))]
    if (!length(donors)) next
    
    # check if ALL years already done for this target — skip fitting entirely
    all_done <- all(
      sapply(YEARS_TEST, function(y) sprintf("%d|%d", target_id, y) %in% done_keys)
    )
    if (all_done) {
      cat(sprintf("  [SKIP] Target %d — all years done.\n", target_id))
      next
    }
    
    cat(sprintf("[%d/%d] Target %d | Donors: %s\n",
                t_idx, n_targets, target_id, paste(donors, collapse = ",")))
    flush.console()
    
    # --- accumulators per year ---
    pred_lists   <- setNames(vector("list", length(YEARS_TEST)), as.character(YEARS_TEST))
    actual_list  <- setNames(vector("list", length(YEARS_TEST)), as.character(YEARS_TEST))
    runtime_list <- setNames(vector("list", length(YEARS_TEST)), as.character(YEARS_TEST))
    rmse_lists   <- setNames(vector("list", length(YEARS_TEST)), as.character(YEARS_TEST))
    mae_lists    <- setNames(vector("list", length(YEARS_TEST)), as.character(YEARS_TEST))
    T_lists      <- setNames(vector("list", length(YEARS_TEST)), as.character(YEARS_TEST))
    donors_used  <- setNames(vector("list", length(YEARS_TEST)), as.character(YEARS_TEST))
    for (yr in as.character(YEARS_TEST)) {
      pred_lists[[yr]]  <- list()
      runtime_list[[yr]] <- numeric(0)
      rmse_lists[[yr]]  <- numeric(0)
      mae_lists[[yr]]   <- numeric(0)
      T_lists[[yr]]     <- integer(0)
      donors_used[[yr]] <- integer(0)
    }
    
    for (donor_id in donors) {
      
      # --- FIT ONCE ---
      cat(sprintf("    Fitting donor %d (2017) ...\n", donor_id))
      flush.console()
      
      fitted <- tryCatch(
        fit_donor_model(donor_id),
        error = function(e) {
          cat(sprintf("    [ERROR] fit donor %d: %s\n", donor_id, conditionMessage(e)))
          NULL
        }
      )
      if (is.null(fitted)) next
      
      cat(sprintf("    Fit done — T: %d | %.1f sec\n", fitted$T_use, fitted$fit_time))
      flush.console()
      
      # --- PREDICT FOR EACH YEAR using the same fitted model ---
      for (test_year in YEARS_TEST) {
        yr <- as.character(test_year)
        
        if (sprintf("%d|%d", target_id, test_year) %in% done_keys) next
        
        cat(sprintf("      Predicting year %d ...\n", test_year))
        flush.console()
        
        res <- tryCatch(
          predict_donor_model(fitted, target_id, test_year),
          error = function(e) {
            cat(sprintf("      [ERROR] predict year %d: %s\n", test_year, conditionMessage(e)))
            NULL
          }
        )
        if (is.null(res)) next
        
        pred_lists[[yr]][[length(pred_lists[[yr]]) + 1L]] <- res$pred
        actual_list[[yr]]  <- res$actual
        runtime_list[[yr]] <- c(runtime_list[[yr]], fitted$fit_time + res$pred_time)
        rmse_lists[[yr]]   <- c(rmse_lists[[yr]],   res$rmse)
        mae_lists[[yr]]    <- c(mae_lists[[yr]],    res$mae)
        T_lists[[yr]]      <- c(T_lists[[yr]],      fitted$T_use)
        donors_used[[yr]]  <- c(donors_used[[yr]],  donor_id)
        
        cat(sprintf("      Year %d done — RMSE: %.4f | %.1f sec\n",
                    test_year, res$rmse, res$pred_time))
        flush.console()
      }
      
      rm(fitted); gc()
    }
    
    # --- save results per year ---
    for (test_year in YEARS_TEST) {
      yr        <- as.character(test_year)
      pair_key  <- sprintf("%d|%d", target_id, test_year)
      if (pair_key %in% done_keys) next
      if (!length(pred_lists[[yr]])) next
      
      ensemble_pred <- Reduce(`+`, pred_lists[[yr]]) / length(pred_lists[[yr]])
      
      row_dt <- data.table(
        method           = "TF_thinnedSV",
        target           = target_id,
        year             = test_year,
        donors_used      = paste(donors_used[[yr]], collapse = ","),
        n_models         = length(pred_lists[[yr]]),
        rmse             = rmse_vec(actual_list[[yr]], ensemble_pred),
        mae              = mae_vec(actual_list[[yr]], ensemble_pred),
        runtime_sec      = sum(runtime_list[[yr]]),
        mean_single_rmse = mean(rmse_lists[[yr]]),
        mean_single_mae  = mean(mae_lists[[yr]]),
        mean_T           = mean(T_lists[[yr]])
      )
      
      fwrite(row_dt, OUTPUT_DETAIL, append = file.exists(OUTPUT_DETAIL))
      done_keys <- c(done_keys, pair_key)
      
      cat(sprintf("  -> Saved target %d year %d. Ensemble RMSE: %.4f\n",
                  target_id, test_year, row_dt$rmse))
      flush.console()
    }
  }
  
  # --- final summary ---
  if (file.exists(OUTPUT_DETAIL) && file.size(OUTPUT_DETAIL) > 0) {
    detail_dt  <- fread(OUTPUT_DETAIL, showProgress = FALSE)
    summary_dt <- detail_dt[, .(
      avg_rmse          = mean(rmse),
      avg_mae           = mean(mae),
      total_runtime_sec = sum(runtime_sec)
    ), by = .(method, year)]
    fwrite(summary_dt, OUTPUT_SUMMARY)
    cat("[DONE] Summary saved.\n")
  }
}

run_metric()