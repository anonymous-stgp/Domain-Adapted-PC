suppressPackageStartupMessages({
  library(data.table)
  library(e1071)
})

ROOT <- normalizePath(getwd())
if (!dir.exists(file.path(ROOT, "data"))) {
  ROOT <- normalizePath(file.path(ROOT, ".."))
}

DATA_DIR      <- file.path(ROOT, "data")
PROCESSED_DIR <- file.path(DATA_DIR, "processed_data")
RESULTS_DIR   <- file.path(ROOT, "results", "intermediate")
dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

# --- year argument: Rscript --vanilla code/Table2_G_SVR_.R 2017
args       <- commandArgs(trailingOnly = TRUE)
TEST_YEARS <- if (length(args) >= 1L) as.integer(args[1]) else c(2017L, 2018L)

year_suffix    <- if (length(args) >= 1L) paste0("_", args[1]) else ""
DONOR_FILE     <- file.path(PROCESSED_DIR, "matching_geographic_distance.csv")
OUTPUT_DETAIL  <- file.path(RESULTS_DIR,   paste0("Table2_G_SVR_detail",  year_suffix, ".csv"))
OUTPUT_SUMMARY <- file.path(RESULTS_DIR,   paste0("Table2_G_SVR_summary", year_suffix, ".csv"))

TARGET        <- "power"
BASE_FEATURES <- c("wind_speed", "temperature", "turbulence_intensity", "std_wind_direction")
ANGLE_FEATURE <- "wind_direction"

K          <- 7L
TRAIN_YEAR <- 2017L
SVM_KERNEL <- "radial"
SVM_COST   <- 1
SVM_GAMMA  <- 1 / 6

rmse_vec <- function(y, yhat) sqrt(mean((y - yhat)^2, na.rm = TRUE))

feature_names <- function() c(BASE_FEATURES, "wind_direction_sin", "wind_direction_cos")

load_turbine_year <- function(turbine_id, year) {
  path <- file.path(DATA_DIR, sprintf("Turbine%d_%d.csv", as.integer(turbine_id), as.integer(year)))
  if (!file.exists(path)) return(NULL)
  dt   <- fread(path, showProgress = FALSE)
  need <- c(BASE_FEATURES, ANGLE_FEATURE, TARGET)
  miss <- setdiff(need, names(dt))
  if (length(miss)) stop("Missing columns: ", paste(miss, collapse = ", "))
  dt <- dt[, ..need]
  dt[, (need) := lapply(.SD, as.numeric), .SDcols = need]
  dt <- na.omit(dt)
  if (!nrow(dt)) return(NULL)
  rad <- dt[[ANGLE_FEATURE]] * pi / 180
  dt[, wind_direction_sin := sin(rad)]
  dt[, wind_direction_cos := cos(rad)]
  dt[, (ANGLE_FEATURE) := NULL]
  dt
}

read_geo_donor_table <- function(path) {
  dt <- fread(path, showProgress = FALSE)

  if (all(c("donor", "geo_distance") %in% names(dt))) {
    dt[, target := as.integer(target)]
    dt[, donor  := as.integer(donor)]
    dt[, geo_distance := as.numeric(geo_distance)]
    dt <- dt[!is.na(target) & !is.na(donor) & !is.na(geo_distance)]
    setorder(dt, target, geo_distance, donor)
    return(dt)
  }

  df         <- as.data.frame(dt)
  donor_cols <- grep("^donor[0-9_]", names(df), value = TRUE)
score_cols <- grep("^score[0-9_]", names(df), value = TRUE)

  rows <- list()
  for (i in seq_along(donor_cols)) {
    rows[[i]] <- data.frame(
      target       = as.integer(df[["target"]]),
      donor        = as.integer(df[[donor_cols[i]]]),
      geo_distance = as.numeric(df[[score_cols[i]]])
    )
  }

  long <- as.data.table(do.call(rbind, rows))
  long <- long[!is.na(target) & !is.na(donor) & !is.na(geo_distance)]
  setorder(long, target, geo_distance, donor)
  long
}

get_top_k_donors <- function(dt, target_id, k) {
  sub <- dt[target == target_id & donor != target_id]
  head(sub$donor, k)
}

fit_one_neighbor_svm <- function(donor_id) {
  feats    <- feature_names()
  train_dt <- load_turbine_year(donor_id, TRAIN_YEAR)
  if (is.null(train_dt)) return(NULL)
  x_train <- train_dt[, ..feats]
  y_train <- train_dt[[TARGET]]
  t0      <- proc.time()[["elapsed"]]
  model   <- svm(
    x = x_train, y = y_train,
    type = "eps-regression", kernel = SVM_KERNEL,
    cost = SVM_COST, gamma = SVM_GAMMA, scale = TRUE
  )
  list(model = model, fit_time = proc.time()[["elapsed"]] - t0)
}

run_target_year <- function(target_id, year, donor_ids) {
  feats   <- feature_names()
  test_dt <- load_turbine_year(target_id, year)
  if (is.null(test_dt)) return(NULL)
  x_test  <- test_dt[, ..feats]
  y_test  <- test_dt[[TARGET]]

  models      <- list()
  fit_times   <- numeric(0)
  donors_used <- integer(0)

  for (d in donor_ids) {
    res <- tryCatch(fit_one_neighbor_svm(d), error = function(e) NULL)
    if (is.null(res)) next
    models[[length(models) + 1L]] <- res$model
    fit_times   <- c(fit_times,   res$fit_time)
    donors_used <- c(donors_used, d)
  }
  if (!length(models)) return(NULL)

  t0       <- proc.time()[["elapsed"]]
  pred_mat <- matrix(NA_real_, nrow = nrow(x_test), ncol = length(models))
  for (j in seq_along(models))
    pred_mat[, j] <- predict(models[[j]], newdata = x_test)
  pred      <- rowMeans(pred_mat, na.rm = TRUE)
  pred_time <- proc.time()[["elapsed"]] - t0

  data.table(
    method        = "G_SVR",
    target        = target_id,
    year          = year,
    donors_used   = paste(donors_used, collapse = ","),
    n_models      = length(models),
    rmse          = rmse_vec(y_test, pred),
    runtime_sec   = sum(fit_times) + pred_time,
    fit_time_sec  = sum(fit_times),
    pred_time_sec = pred_time
  )
}

main <- function() {
  geo_dt    <- read_geo_donor_table(DONOR_FILE)
  targets   <- sort(unique(geo_dt$target))
  n_targets <- length(targets)

  # --- resume ---
  done_keys <- character(0)
  if (file.exists(OUTPUT_DETAIL) && file.size(OUTPUT_DETAIL) > 0) {
    prev <- tryCatch(fread(OUTPUT_DETAIL, showProgress = FALSE), error = function(e) data.table())
    if (nrow(prev) && all(c("target", "year") %in% names(prev))) {
      done_keys <- unique(sprintf("%d|%d", as.integer(prev$target), as.integer(prev$year)))
      cat(sprintf("[INFO] Resuming — %d pairs already done.\n", length(done_keys)))
    }
  }

  cat(sprintf("[INFO] Starting G_SVR — %d targets x %d years\n",
              n_targets, length(TEST_YEARS)))

  for (t_idx in seq_along(targets)) {
    target_id <- targets[t_idx]
    donor_ids <- get_top_k_donors(geo_dt, target_id, K)
    if (!length(donor_ids)) next

    for (year in TEST_YEARS) {
      pair_key <- sprintf("%d|%d", target_id, year)
      if (pair_key %in% done_keys) {
        cat(sprintf("  [SKIP] Target %d Year %d\n", target_id, year))
        next
      }

      cat(sprintf("[%d/%d] G_SVR - Turbine %d Year %d\n",
                  t_idx, n_targets, target_id, year))
      flush.console()

      out <- tryCatch(
        run_target_year(target_id, year, donor_ids),
        error = function(e) {
          cat(sprintf("  [ERROR] %s\n", conditionMessage(e)))
          NULL
        }
      )
      if (is.null(out)) next

      fwrite(out, OUTPUT_DETAIL, append = file.exists(OUTPUT_DETAIL))
      done_keys <- c(done_keys, pair_key)

      cat(sprintf("  -> Saved. RMSE: %.4f\n", out$rmse))
      flush.console()
    }
  }

  # final summary
  if (file.exists(OUTPUT_DETAIL) && file.size(OUTPUT_DETAIL) > 0) {
    detail_dt  <- fread(OUTPUT_DETAIL, showProgress = FALSE)
    summary_dt <- detail_dt[, .(
      avg_rmse          = mean(rmse, na.rm = TRUE),
      total_runtime_sec = sum(runtime_sec, na.rm = TRUE)
    ), by = .(method, year)]
    fwrite(summary_dt, OUTPUT_SUMMARY)
    cat("[DONE] Summary saved.\n")
  }
}

main()