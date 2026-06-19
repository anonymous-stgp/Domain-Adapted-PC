suppressPackageStartupMessages({
  library(data.table)
  library(twingp)
})

ROOT <- normalizePath(getwd())
if (!dir.exists(file.path(ROOT, "data"))) {
  ROOT <- normalizePath(file.path(ROOT, ".."))
}

DATA_DIR <- file.path(ROOT, "data")
RESULTS_DIR <- file.path(ROOT, "results", "intermediate")
dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

TARGET <- "power"
BASE_FEATURES <- c("wind_speed", "temperature", "turbulence_intensity", "std_wind_direction")
ANGLE_FEATURE <- "wind_direction"

SEED <- 2026L
MAX_THINNING_NUMBER <- 20L
YEARS_TEST <- c(2017L, 2018L)

ALL_IDS <- 1:66

# ─────────────────────────────────────────
# CLI ARGS
# arg1: turbine range "start:end"  (default = all)
# ─────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)

TURBINE_RANGE <- if (length(args) >= 1L) {
  parts <- as.integer(strsplit(args[1], ":")[[1]])
  if (length(parts) == 2L) seq(parts[1], parts[2]) else NULL
} else NULL

RANGE_TAG <- if (!is.null(TURBINE_RANGE)) {
  sprintf("_t%dto%d", min(TURBINE_RANGE), max(TURBINE_RANGE))
} else ""

DETAIL_PATH  <- file.path(RESULTS_DIR, sprintf("Table2_P_thinned_twinGP%s_detail.csv",  RANGE_TAG))
SUMMARY_PATH <- file.path(RESULTS_DIR, sprintf("Table2_P_thinned_twinGP%s_summary.csv", RANGE_TAG))

DETAIL_COLS <- c(
  "method", "target", "year", "n_train_turbines",
  "rmse", "mae", "runtime_sec", "T"
)

# ─────────────────────────────────────────
# HELPERS  (identical to Table1.R)
# ─────────────────────────────────────────

mae_vec  <- function(y, yhat) mean(abs(y - yhat))
rmse_vec <- function(y, yhat) sqrt(mean((y - yhat)^2))

feature_names <- function() c(BASE_FEATURES, "wind_direction_sin", "wind_direction_cos")

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

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

main <- function() {
  feats <- feature_names()

  ensure_header(DETAIL_PATH, DETAIL_COLS)
  done_keys <- done_pairs(DETAIL_PATH)

  turbine_ids <- ALL_IDS
  if (!is.null(TURBINE_RANGE)) {
    turbine_ids <- turbine_ids[turbine_ids %in% TURBINE_RANGE]
  }

  cat("Model      : P_thinned_twinGP (pooled LOTO baseline)\n")
  cat("Turbines   :", length(turbine_ids), "\n")
  cat("Years      :", paste(YEARS_TEST, collapse = ", "), "\n\n")

  for (target_id in turbine_ids) {
    years_needed <- YEARS_TEST[!sapply(YEARS_TEST, function(y) sprintf("%d|%d", target_id, y) %in% done_keys)]
    if (!length(years_needed)) {
      cat("  [SKIP] Target", target_id, "— all years done.\n")
      next
    }

    train_ids <- setdiff(ALL_IDS, target_id)

    cat("[INFO] Pooling 2017 data from", length(train_ids), "turbines for target", target_id, "...\n")

    x_list <- list()
    y_list <- list()
    for (tid in train_ids) {
      res <- tryCatch(load_turbine_year(tid, 2017L), error = function(e) NULL)
      if (is.null(res)) next
      x_list[[length(x_list) + 1L]] <- as.matrix(res[, ..feats])
      y_list[[length(y_list) + 1L]] <- res[[TARGET]]
    }
    if (!length(x_list)) {
      cat("  [ERROR] No training data available for target", target_id, "\n")
      next
    }

    x_train <- do.call(rbind, x_list)
    y_train <- unlist(y_list)
    rm(x_list, y_list); gc()

    set.seed(SEED)
    T_use <- compute_thinning_number(x_train, MAX_THINNING_NUMBER)

    for (test_year in years_needed) {
      pair_key <- sprintf("%d|%d", target_id, test_year)

      cat("  Target:", target_id, "| Year:", test_year, "\n")

      test_dt <- tryCatch(load_turbine_year(target_id, test_year), error = function(e) NULL)
      if (is.null(test_dt)) {
        cat("    [ERROR] missing test data\n")
        next
      }
      x_test <- as.matrix(test_dt[, ..feats])
      y_test <- test_dt[[TARGET]]

      t0 <- proc.time()[["elapsed"]]
      pred <- tryCatch(
        thinned_twingp_full(x_train, y_train, x_test, T = T_use),
        error = function(e) {
          cat("    [ERROR]", conditionMessage(e), "\n")
          NULL
        }
      )
      runtime <- proc.time()[["elapsed"]] - t0
      if (is.null(pred)) { gc(); next }

      row_dt <- data.table(
        method = "P_thinned_twinGP",
        target = target_id,
        year = test_year,
        n_train_turbines = length(train_ids),
        rmse = rmse_vec(y_test, pred),
        mae = mae_vec(y_test, pred),
        runtime_sec = runtime,
        T = T_use
      )
      append_row(DETAIL_PATH, row_dt)
      done_keys <- c(done_keys, pair_key)

      cat(sprintf("    -> RMSE: %.4f  MAE: %.4f  T: %d  time: %.1fs\n",
                  row_dt$rmse, row_dt$mae, T_use, runtime))
      rm(row_dt, pred, test_dt, x_test, y_test); gc()
    }

    rm(x_train, y_train); gc()
  }

  if (file.exists(DETAIL_PATH) && file.size(DETAIL_PATH) > 0L) {
    detail_dt <- fread(DETAIL_PATH, showProgress = FALSE)
    summary_dt <- detail_dt[, .(
      avg_rmse = mean(rmse, na.rm = TRUE),
      avg_mae = mean(mae, na.rm = TRUE),
      total_runtime_sec = sum(runtime_sec, na.rm = TRUE),
      n_targets = .N
    ), by = .(method, year)]
    fwrite(summary_dt, SUMMARY_PATH)
    cat("\nSummary saved to:", SUMMARY_PATH, "\n")
    print(summary_dt[order(year)])
  }
}

main()
