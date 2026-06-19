suppressPackageStartupMessages({
  library(data.table)
  library(twingp)
})

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────

ROOT <- normalizePath(getwd())
if (!dir.exists(file.path(ROOT, "data"))) {
  ROOT <- normalizePath(file.path(ROOT, ".."))
}

DATA_DIR      <- file.path(ROOT, "data")
PROCESSED_DIR <- file.path(DATA_DIR, "processed_data")
RESULTS_DIR   <- file.path(ROOT, "results", "intermediate")
dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

TARGET        <- "power"
BASE_FEATURES <- c("wind_speed", "temperature", "turbulence_intensity", "std_wind_direction")
ANGLE_FEATURE <- "wind_direction"

SEED               <- 42L
MAX_THINNING_NUMBER <- 20L
YEARS_TEST         <- c(2017L, 2018L)
K_VALUES           <- 2:10

DONOR_FILE <- file.path(PROCESSED_DIR, "matching_weighted_ks.csv")

DETAIL_FILE  <- file.path(RESULTS_DIR, "Figure5_twingp_detail.csv")
SUMMARY_FILE <- file.path(RESULTS_DIR, "Figure5_twingp_summary.csv")

DETAIL_COLS <- c(
  "model", "aggregation", "K", "target", "year",
  "donors_used", "n_models",
  "rmse", "mae", "runtime_sec", "mean_T"
)

# ─────────────────────────────────────────
# CLI ARGS
# arg1: turbine range "start:end"  (default = all)
# arg2: K range       "kmin:kmax"  (default = "2:10")
# ─────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)

TURBINE_RANGE <- if (length(args) >= 1L) {
  parts <- as.integer(strsplit(args[1], ":")[[1]])
  if (length(parts) == 2L) seq(parts[1], parts[2]) else NULL
} else NULL

if (length(args) >= 2L) {
  kparts   <- as.integer(strsplit(args[2], ":")[[1]])
  K_VALUES <- if (length(kparts) == 2L) seq(kparts[1], kparts[2]) else K_VALUES
}

RANGE_TAG <- if (!is.null(TURBINE_RANGE)) {
  sprintf("_t%dto%d", min(TURBINE_RANGE), max(TURBINE_RANGE))
} else ""

DETAIL_FILE  <- file.path(RESULTS_DIR, sprintf("Figure5_twingp%s_detail.csv",  RANGE_TAG))
SUMMARY_FILE <- file.path(RESULTS_DIR, sprintf("Figure5_twingp%s_summary.csv", RANGE_TAG))

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

rmse_vec <- function(y, yhat) sqrt(mean((y - yhat)^2))
mae_vec  <- function(y, yhat) mean(abs(y - yhat))

feature_names <- function() c(BASE_FEATURES, "wind_direction_sin", "wind_direction_cos")

ensure_header <- function(path, cols) {
  if (!file.exists(path) || is.na(file.size(path)) || file.size(path) == 0L) {
    fwrite(
      as.data.table(setNames(replicate(length(cols), logical(0), simplify = FALSE), cols)),
      path
    )
  }
}

done_keys_load <- function(path) {
  if (!file.exists(path) || is.na(file.size(path)) || file.size(path) == 0L) return(character())
  dt <- tryCatch(fread(path, showProgress = FALSE), error = function(e) data.table())
  req <- c("model", "aggregation", "K", "target", "year")
  if (!nrow(dt) || !all(req %in% names(dt))) return(character())
  unique(sprintf("%s|%s|%d|%d|%d",
    dt$model, dt$aggregation,
    as.integer(dt$K), as.integer(dt$target), as.integer(dt$year)
  ))
}

make_key <- function(model, agg, k, target, year) {
  sprintf("%s|%s|%d|%d|%d", model, agg, as.integer(k), as.integer(target), as.integer(year))
}

load_turbine_year <- function(turbine_id, year) {
  path <- file.path(DATA_DIR, sprintf("Turbine%d_%d.csv", as.integer(turbine_id), as.integer(year)))
  if (!file.exists(path)) stop("Missing file: ", path)
  dt   <- fread(path, showProgress = FALSE)
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

read_donor_table <- function(path) {
  dt <- fread(path, showProgress = FALSE)
  if (names(dt)[1] != "target") setnames(dt, 1L, "target")
  donor_cols <- grep("^donor", names(dt), value = TRUE)
  if (!length(donor_cols)) stop("No donor columns in ", path)
  dt[, (c("target", donor_cols)) := lapply(.SD, as.integer), .SDcols = c("target", donor_cols)]
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
    idx    <- which(c(1, abs(pacf_vals)) <= thresh)
    if (length(idx)) thinning_vec[j] <- min(idx)
  }
  max(1L, max(thinning_vec))
}

thinned_twingp_full <- function(x, y, x_test, T) {
  n <- nrow(x)
  d <- ncol(x)
  bins <- vector("list", T)
  for (b in seq_len(T)) {
    idx      <- seq(from = b, to = n, by = T)
    bins[[b]] <- list(x = x[idx, , drop = FALSE], y = y[idx])
  }
  pred_list <- vector("list", T)
  for (b in seq_len(T)) {
    train_x <- bins[[b]]$x
    train_y <- bins[[b]]$y
    l_num   <- max(25, 3 * d)
    g_num   <- min(50 * d, max(sqrt(nrow(train_x)), 10 * d))
    v_num   <- 2 * min(50 * d, max(sqrt(nrow(train_x)), 10 * d))
    pred_list[[b]] <- twingp::twingp(
      as.matrix(train_x), as.matrix(train_y), as.matrix(x_test),
      l_num = l_num, g_num = g_num, v_num = v_num
    )
    gc()
  }
  mu <- numeric(nrow(x_test))
  for (i in seq_len(nrow(x_test))) {
    mu_vals <- vapply(pred_list, function(p) p$mu[i], numeric(1))
    mu[i]   <- mean(mu_vals)
  }
  rm(pred_list, bins); gc()
  mu
}

# ─────────────────────────────────────────
# ENSEMBLE: train one model per donor, average predictions
# ─────────────────────────────────────────

run_ensemble <- function(donors, target_id, test_year, feats) {
  test_dt <- load_turbine_year(target_id, test_year)
  x_test  <- as.matrix(test_dt[, ..feats])
  y_test  <- test_dt[[TARGET]]
  rm(test_dt); gc()

  preds       <- list()
  T_vals      <- integer(0)
  runtimes    <- numeric(0)

  for (donor_id in donors) {
    res <- tryCatch({
      train_dt <- load_turbine_year(donor_id, 2017L)
      x_train  <- as.matrix(train_dt[, ..feats])
      y_train  <- train_dt[[TARGET]]
      rm(train_dt); gc()

      set.seed(SEED)
      T_use <- compute_thinning_number(x_train, MAX_THINNING_NUMBER)
      t0    <- proc.time()[["elapsed"]]
      pred  <- thinned_twingp_full(x_train, y_train, x_test, T = T_use)
      rt    <- proc.time()[["elapsed"]] - t0
      rm(x_train, y_train); gc()

      list(pred = pred, T = T_use, runtime = rt)
    }, error = function(e) {
      cat("    [ERROR] ensemble donor", donor_id, ":", conditionMessage(e), "\n")
      NULL
    })
    if (is.null(res)) next
    preds[[length(preds) + 1L]] <- res$pred
    T_vals   <- c(T_vals,   res$T)
    runtimes <- c(runtimes, res$runtime)
  }

  if (!length(preds)) return(NULL)

  ensemble_pred <- Reduce(`+`, preds) / length(preds)
  list(
    pred        = ensemble_pred,
    actual      = y_test,
    rmse        = rmse_vec(y_test, ensemble_pred),
    mae         = mae_vec(y_test, ensemble_pred),
    runtime_sec = sum(runtimes),
    mean_T      = mean(T_vals),
    n_models    = length(preds)
  )
}

# ─────────────────────────────────────────
# CONCAT: pool all donor data, fit one model
# ─────────────────────────────────────────

run_concat <- function(donors, target_id, test_year, feats) {
  test_dt <- load_turbine_year(target_id, test_year)
  x_test  <- as.matrix(test_dt[, ..feats])
  y_test  <- test_dt[[TARGET]]
  rm(test_dt); gc()

  # pool donor data
  all_x <- vector("list", length(donors))
  all_y <- vector("list", length(donors))
  for (i in seq_along(donors)) {
    res <- tryCatch({
      dt <- load_turbine_year(donors[i], 2017L)
      list(x = as.matrix(dt[, ..feats]), y = dt[[TARGET]])
    }, error = function(e) {
      cat("    [ERROR] concat load donor", donors[i], ":", conditionMessage(e), "\n")
      NULL
    })
    if (is.null(res)) next
    all_x[[i]] <- res$x
    all_y[[i]] <- res$y
  }

  valid   <- !vapply(all_x, is.null, logical(1))
  x_train <- do.call(rbind, all_x[valid])
  y_train <- unlist(all_y[valid])
  rm(all_x, all_y); gc()

  if (is.null(x_train) || nrow(x_train) == 0) return(NULL)

  set.seed(SEED)
  T_use <- compute_thinning_number(x_train, MAX_THINNING_NUMBER)
  t0    <- proc.time()[["elapsed"]]
  pred  <- thinned_twingp_full(x_train, y_train, x_test, T = T_use)
  rt    <- proc.time()[["elapsed"]] - t0
  rm(x_train, y_train); gc()

  list(
    pred        = pred,
    actual      = y_test,
    rmse        = rmse_vec(y_test, pred),
    mae         = mae_vec(y_test, pred),
    runtime_sec = rt,
    mean_T      = T_use,
    n_models    = 1L
  )
}

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

main <- function() {
  donor_obj  <- read_donor_table(DONOR_FILE)
  donor_dt   <- donor_obj$dt
  donor_cols <- donor_obj$donor_cols
  feats      <- feature_names()

  ensure_header(DETAIL_FILE, DETAIL_COLS)
  done_keys <- done_keys_load(DETAIL_FILE)

  turbine_ids <- sort(donor_dt$target)
  if (!is.null(TURBINE_RANGE)) {
    turbine_ids <- turbine_ids[turbine_ids %in% TURBINE_RANGE]
  }

  cat("Model         : thinned twinGP\n")
  cat("Turbines      :", length(turbine_ids), "\n")
  cat("K values      :", paste(K_VALUES, collapse = ", "), "\n")
  cat("Years         :", paste(YEARS_TEST, collapse = ", "), "\n")
  cat("Seed          :", SEED, "\n\n")

  for (target_id in turbine_ids) {
    donor_row  <- donor_dt[target == target_id]
    all_donors <- as.integer(na.omit(unlist(donor_row[, ..donor_cols])))
    all_donors <- unique(all_donors[all_donors != target_id])
    max_k      <- min(max(K_VALUES), length(all_donors))
    if (max_k < min(K_VALUES)) next

    for (k in K_VALUES) {
      donors <- all_donors[seq_len(min(k, length(all_donors)))]
      if (length(donors) < k) {
        cat("  Skipping K =", k, "target", target_id, "— not enough donors\n")
        next
      }

      for (agg in c("ensemble", "concat")) {
        for (test_year in YEARS_TEST) {
          key <- make_key("twingp", agg, k, target_id, test_year)
          if (key %in% done_keys) {
            cat("  Skip:", agg, "K =", k, "target", target_id, "year", test_year, "\n")
            next
          }

          cat(sprintf("  [twingp|%s] K=%d  target=%d  year=%d\n", agg, k, target_id, test_year))

          res <- tryCatch(
            if (agg == "ensemble") {
              run_ensemble(donors, target_id, test_year, feats)
            } else {
              run_concat(donors, target_id, test_year, feats)
            },
            error = function(e) {
              cat("  [ERROR]", conditionMessage(e), "\n"); NULL
            }
          )
          if (is.null(res)) { gc(); next }

          row_dt <- data.table(
            model       = "twingp",
            aggregation = agg,
            K           = k,
            target      = target_id,
            year        = test_year,
            donors_used = paste(donors, collapse = ","),
            n_models    = res$n_models,
            rmse        = res$rmse,
            mae         = res$mae,
            runtime_sec = res$runtime_sec,
            mean_T      = res$mean_T
          )
          fwrite(row_dt, file = DETAIL_FILE, append = TRUE)
          done_keys <- c(done_keys, key)

          cat(sprintf("    -> RMSE: %.4f  MAE: %.4f  T: %.1f  time: %.1fs\n",
                      res$rmse, res$mae, res$mean_T, res$runtime_sec))
          rm(res, row_dt); gc()
        }
      }
    }
  }

  # summary: avg RMSE/MAE per (model, aggregation, K, year)
  if (file.exists(DETAIL_FILE) && file.size(DETAIL_FILE) > 0L) {
    detail_dt  <- fread(DETAIL_FILE, showProgress = FALSE)
    summary_dt <- detail_dt[, .(
      avg_rmse          = mean(rmse, na.rm = TRUE),
      avg_mae           = mean(mae,  na.rm = TRUE),
      total_runtime_sec = sum(runtime_sec, na.rm = TRUE),
      n_targets         = .N
    ), by = .(model, aggregation, K, year)]
    fwrite(summary_dt, SUMMARY_FILE)
    cat("\nSummary saved to:", SUMMARY_FILE, "\n")
    print(summary_dt[order(aggregation, K, year)])
  }
}

main()
