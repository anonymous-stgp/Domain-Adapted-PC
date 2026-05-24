suppressPackageStartupMessages({
  library(data.table)
  library(twingp)
})
# --- ADD THIS BLOCK AT TOP ---
TEST_IDS <- c(38:44)
ALL_IDS <- 1:66
TRAIN_IDS <- setdiff(ALL_IDS, TEST_IDS)
SETUP_SUFFIX <- "_dfp"
# --------------------------------
ROOT <- normalizePath(getwd())
if (!dir.exists(file.path(ROOT, "data"))) {
  ROOT <- normalizePath(file.path(ROOT, ".."))
}

DATA_DIR <- file.path(ROOT, "data")
PROCESSED_DIR <- file.path(DATA_DIR, "processed_data")
RESULTS_DIR <- file.path(ROOT, "results", "intermediate")

dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

TARGET <- "power"
BASE_FEATURES <- c("wind_speed", "temperature", "turbulence_intensity", "std_wind_direction")
ANGLE_FEATURE <- "wind_direction"

K <- 7L
SEED <- 2026L
MAX_THINNING_NUMBER <- 20L
YEARS_TEST <- c(2017L, 2018L)

DONOR_FILES <- list(
  weighted_ks = file.path(PROCESSED_DIR, "matching_weighted_ks_dfp.csv"),
  mean_ks = file.path(PROCESSED_DIR, "matching_mean_ks_dfp.csv"),
  marginal_energy = file.path(PROCESSED_DIR, "matching_marginal_energy_dfp.csv"),
  sinkhorn_wasserstein = file.path(PROCESSED_DIR, "matching_sinkhorn_wasserstein_dfp.csv")
)

args <- commandArgs(trailingOnly = TRUE)
METRICS_TO_RUN <- if (length(args) >= 1L) args else names(DONOR_FILES)

mae_vec <- function(y, yhat) mean(abs(y - yhat))
rmse_vec <- function(y, yhat) sqrt(mean((y - yhat)^2))

feature_names <- function() {
  c(BASE_FEATURES, "wind_direction_sin", "wind_direction_cos")
}

ensure_header <- function(path, cols) {
  if (!file.exists(path) || is.na(file.size(path)) || file.size(path) == 0L) {
    fwrite(as.data.table(setNames(replicate(length(cols), logical(0), simplify = FALSE), cols)), path)
  }
}

append_row <- function(path, row_dt) {
  fwrite(row_dt, file = path, append = TRUE)
}

done_pairs <- function(detail_path) {
  if (!file.exists(detail_path) || is.na(file.size(detail_path)) || file.size(detail_path) == 0L) {
    return(character())
  }
  dt <- tryCatch(fread(detail_path, showProgress = FALSE), error = function(e) data.table())
  req <- c("target", "year")
  if (!nrow(dt) || !all(req %in% names(dt))) return(character())
  unique(sprintf("%d|%d", as.integer(dt$target), as.integer(dt$year)))
}

load_turbine_year <- function(turbine_id, year) {
  path <- file.path(DATA_DIR, sprintf("Turbine%d_%d.csv", as.integer(turbine_id), as.integer(year)))
  if (!file.exists(path)) stop("Missing file: ", path)
  
  dt <- fread(path, showProgress = FALSE)
  need <- c(BASE_FEATURES, ANGLE_FEATURE, TARGET)
  miss <- setdiff(need, names(dt))
  if (length(miss)) stop("Missing columns in ", path, ": ", paste(miss, collapse = ", "))
  
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

thinned_twingp_full <- function(x, y, x_test, T) {
  n <- nrow(x)
  d <- ncol(x)
  
  bins <- vector("list", T)
  for (b in seq_len(T)) {
    idx <- seq(from = b, to = n, by = T)
    bins[[b]] <- list(x = x[idx, , drop = FALSE], y = y[idx])
  }
  
  pred_list <- vector("list", T)
  
  for (b in seq_len(T)) {
    train_x <- bins[[b]]$x
    train_y <- bins[[b]]$y
    
    l_num <- max(25, 3 * d)
    g_num <- min(50 * d, max(sqrt(nrow(train_x)), 10 * d))
    v_num <- 2 * min(50 * d, max(sqrt(nrow(train_x)), 10 * d))
    
    pred_list[[b]] <- twingp::twingp(
      as.matrix(train_x),
      as.matrix(train_y),
      as.matrix(x_test),
      l_num = l_num,
      g_num = g_num,
      v_num = v_num
    )
    
    gc()
  }
  
  mu <- numeric(nrow(x_test))
  for (i in seq_len(nrow(x_test))) {
    mu_vals <- vapply(pred_list, function(p) p$mu[i], numeric(1))
    mu[i] <- mean(mu_vals)
  }
  
  rm(pred_list, bins)
  gc()
  
  mu
}

fit_one_donor <- function(donor_id, target_id, test_year) {
  feats <- feature_names()
  
  train_dt <- load_turbine_year(donor_id, 2017L)
  test_dt <- load_turbine_year(target_id, test_year)
  
  x_train <- as.matrix(train_dt[, ..feats])
  y_train <- train_dt[[TARGET]]
  x_test <- as.matrix(test_dt[, ..feats])
  y_test <- test_dt[[TARGET]]
  
  rm(train_dt, test_dt)
  gc()
  
  set.seed(SEED)
  T_use <- compute_thinning_number(x_train, MAX_THINNING_NUMBER)
  
  t0 <- proc.time()[["elapsed"]]
  pred <- thinned_twingp_full(x_train, y_train, x_test, T = T_use)
  runtime <- proc.time()[["elapsed"]] - t0
  
  rm(x_train, y_train, x_test)
  gc()
  
  list(
    pred = pred,
    actual = y_test,
    rmse = rmse_vec(y_test, pred),
    mae = mae_vec(y_test, pred),
    runtime = runtime,
    T = T_use
  )
}

read_donor_table <- function(path) {
  dt <- fread(path, showProgress = FALSE)
  if (names(dt)[1] != "target") setnames(dt, 1L, "target")
  
  donor_cols <- grep("^donor", names(dt), value = TRUE)
  if (!length(donor_cols)) donor_cols <- setdiff(names(dt), "target")
  if (!length(donor_cols)) stop("No donor columns in ", path)
  
  dt[, (c("target", donor_cols)) := lapply(.SD, as.integer), .SDcols = c("target", donor_cols)]
  list(dt = dt, donor_cols = donor_cols)
}

run_metric <- function(metric_name, donor_path) {
  donor_obj <- read_donor_table(donor_path)
  donor_dt <- donor_obj$dt
  donor_cols <- donor_obj$donor_cols
  
  detail_path <- file.path(RESULTS_DIR, sprintf("Table1_%s_detail%s.csv", metric_name, SETUP_SUFFIX))
  summary_path <- file.path(RESULTS_DIR, sprintf("Table1_%s_summary%s.csv", metric_name, SETUP_SUFFIX))
  
  detail_cols <- c(
    "metric", "target", "year", "donors_used", "n_models",
    "rmse", "mae", "runtime_sec",
    "mean_single_rmse", "mean_single_mae", "mean_T"
  )
  ensure_header(detail_path, detail_cols)
  
  done_keys <- done_pairs(detail_path)
  turbine_ids <- intersect(sort(donor_dt$target), TEST_IDS)
  
  for (target_id in turbine_ids) {
    donor_row <- donor_dt[target == target_id]
    donors <- as.integer(na.omit(unlist(donor_row[, ..donor_cols])))
    donors <- unique(donors[donors %in% TRAIN_IDS])
    donors <- donors[seq_len(min(K, length(donors)))]
    if (!length(donors)) next
    
    for (test_year in YEARS_TEST) {
      pair_key <- sprintf("%d|%d", target_id, test_year)
      if (pair_key %in% done_keys) {
        cat("Skipping completed:", metric_name, "| Target:", target_id, "| Year:", test_year, "\n")
        next
      }
      
      cat("Metric:", metric_name, "| Target:", target_id, "| Year:", test_year, "\n")
      
      donor_preds <- list()
      donor_runtimes <- numeric(0)
      donor_rmses <- numeric(0)
      donor_maes <- numeric(0)
      donor_T <- integer(0)
      donors_ok <- integer(0)
      actual <- NULL
      
      for (donor_id in donors) {
        res <- tryCatch(
          fit_one_donor(donor_id, target_id, test_year),
          error = function(e) NULL
        )
        if (is.null(res)) {
          gc()
          next
        }
        
        donor_preds[[length(donor_preds) + 1L]] <- res$pred
        actual <- res$actual
        donor_runtimes <- c(donor_runtimes, res$runtime)
        donor_rmses <- c(donor_rmses, res$rmse)
        donor_maes <- c(donor_maes, res$mae)
        donor_T <- c(donor_T, res$T)
        donors_ok <- c(donors_ok, donor_id)
        
        rm(res)
        gc()
      }
      
      if (!length(donor_preds)) {
        gc()
        next
      }
      
      ensemble_pred <- Reduce(`+`, donor_preds) / length(donor_preds)
      
      row_dt <- data.table(
        metric = metric_name,
        target = target_id,
        year = test_year,
        donors_used = paste(donors_ok, collapse = ","),
        n_models = length(donor_preds),
        rmse = rmse_vec(actual, ensemble_pred),
        mae = mae_vec(actual, ensemble_pred),
        runtime_sec = sum(donor_runtimes),
        mean_single_rmse = mean(donor_rmses),
        mean_single_mae = mean(donor_maes),
        mean_T = mean(donor_T)
      )
      
      append_row(detail_path, row_dt)
      done_keys <- c(done_keys, pair_key)
      
      rm(donor_preds, donor_runtimes, donor_rmses, donor_maes, donor_T, donors_ok, actual, ensemble_pred, row_dt)
      gc()
    }
  }
  
  detail_dt <- fread(detail_path, showProgress = FALSE)
  if (nrow(detail_dt)) {
    summary_dt <- detail_dt[, .(
      avg_rmse = mean(rmse, na.rm = TRUE),
      avg_mae = mean(mae, na.rm = TRUE),
      total_runtime_sec = sum(runtime_sec, na.rm = TRUE)
    ), by = .(metric, year)]
  } else {
    summary_dt <- data.table(
      metric = metric_name,
      year = integer(),
      avg_rmse = numeric(),
      avg_mae = numeric(),
      total_runtime_sec = numeric()
    )
  }
  
  fwrite(summary_dt, summary_path)
  summary_dt
}

main <- function() {
  run_metrics <- intersect(METRICS_TO_RUN, names(DONOR_FILES))
  if (!length(run_metrics)) stop("No valid metrics requested.")
  
  all_summary <- rbindlist(lapply(run_metrics, function(metric_name) {
    donor_path <- DONOR_FILES[[metric_name]]
    if (!file.exists(donor_path)) stop("Missing donor file: ", donor_path)
    run_metric(metric_name, donor_path)
  }), fill = TRUE)
  
  fwrite(all_summary, file.path(RESULTS_DIR, paste0("Table1_summary_all_metrics", SETUP_SUFFIX, ".csv")))
}

main()