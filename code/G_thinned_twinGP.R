suppressPackageStartupMessages({
  library(data.table)
  library(twingp)
})

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

DONOR_FILE <- file.path(PROCESSED_DIR, "matching_geographic_distance.csv")

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

DETAIL_PATH  <- file.path(RESULTS_DIR, sprintf("Table2_G_thinned_twinGP%s_detail.csv",  RANGE_TAG))
SUMMARY_PATH <- file.path(RESULTS_DIR, sprintf("Table2_G_thinned_twinGP%s_summary.csv", RANGE_TAG))

DETAIL_COLS <- c(
  "method", "target", "year", "donors_used", "n_models",
  "rmse", "mae", "runtime_sec",
  "mean_single_rmse", "mean_single_mae", "mean_T"
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

read_geo_donor_table <- function(path) {
  dt <- fread(path, showProgress = FALSE)

  if (all(c("donor", "geo_distance") %in% names(dt))) {
    dt[, target := as.integer(target)]
    dt[, donor := as.integer(donor)]
    dt[, geo_distance := as.numeric(geo_distance)]
    dt <- dt[!is.na(target) & !is.na(donor) & !is.na(geo_distance)]
    setorder(dt, target, geo_distance, donor)
    return(dt)
  }

  donor_cols <- grep("^donor_", names(dt), value = TRUE)
  score_cols <- grep("^score_", names(dt), value = TRUE)

  rows <- list()
  for (i in seq_along(donor_cols)) {
    tmp <- dt[, .(target = as.integer(target),
                  donor = as.integer(get(donor_cols[i])),
                  geo_distance = as.numeric(get(score_cols[i])))]
    rows[[i]] <- tmp
  }

  long <- rbindlist(rows)
  long <- long[!is.na(target) & !is.na(donor) & !is.na(geo_distance)]
  setorder(long, target, geo_distance, donor)
  long
}

get_top_k_donors <- function(dt, target_id, k) {
  sub <- dt[target == target_id & donor != target_id]
  head(sub$donor, k)
}

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

main <- function() {
  geo_dt <- read_geo_donor_table(DONOR_FILE)

  ensure_header(DETAIL_PATH, DETAIL_COLS)
  done_keys <- done_pairs(DETAIL_PATH)

  turbine_ids <- sort(unique(geo_dt$target))
  if (!is.null(TURBINE_RANGE)) {
    turbine_ids <- turbine_ids[turbine_ids %in% TURBINE_RANGE]
  }

  cat("Model      : G_thinned_twinGP (geographic donors, K =", K, ", ensemble)\n")
  cat("Turbines   :", length(turbine_ids), "\n")
  cat("Years      :", paste(YEARS_TEST, collapse = ", "), "\n\n")

  for (target_id in turbine_ids) {
    donors <- get_top_k_donors(geo_dt, target_id, K)
    if (!length(donors)) next

    for (test_year in YEARS_TEST) {
      pair_key <- sprintf("%d|%d", target_id, test_year)
      if (pair_key %in% done_keys) {
        cat("Skipping completed: Target:", target_id, "| Year:", test_year, "\n")
        next
      }

      cat("Target:", target_id, "| Year:", test_year, "\n")

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
        if (is.null(res)) { gc(); next }

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

      if (!length(donor_preds)) { gc(); next }

      ensemble_pred <- Reduce(`+`, donor_preds) / length(donor_preds)

      row_dt <- data.table(
        method = "G_thinned_twinGP",
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

      append_row(DETAIL_PATH, row_dt)
      done_keys <- c(done_keys, pair_key)

      rm(donor_preds, donor_runtimes, donor_rmses, donor_maes, donor_T, donors_ok, actual, ensemble_pred, row_dt)
      gc()
    }
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
