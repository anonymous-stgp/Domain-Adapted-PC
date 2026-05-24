suppressPackageStartupMessages({
  library(xgboost)
  library(data.table)
})

# ---------- paths ----------
ROOT <- normalizePath(".")
if (!dir.exists(file.path(ROOT, "data"))) ROOT <- dirname(ROOT)

DATA_DIR      <- file.path(ROOT, "data")
PROCESSED_DIR <- file.path(ROOT, "data", "processed data")
RESULTS_DIR   <- file.path(ROOT, "results", "intermediate")
dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

DONOR_FILE   <- file.path(PROCESSED_DIR, "matching_geographic_distance.csv")
DETAIL_FILE  <- file.path(RESULTS_DIR, "Table2_G_XGBoost_detail_dfp.csv")
SUMMARY_FILE <- file.path(RESULTS_DIR, "Table2_G_XGBoost_summary_dfp.csv")

# ---------- config ----------
TARGET        <- "power"
BASE_FEATURES <- c("wind_speed", "temperature", "turbulence_intensity", "std_wind_direction")
FEATURES      <- c(BASE_FEATURES, "wind_direction_sin", "wind_direction_cos")
K             <- 7
TRAIN_YEAR    <- 2017
TEST_YEARS    <- c(2017, 2018)
SEED          <- 2026

ALL_IDS   <- 1:66
TEST_IDS  <- 38:44
TRAIN_IDS <- setdiff(ALL_IDS, TEST_IDS)

XGB_PARAMS <- list(
  objective        = "reg:squarederror",
  eta              = 0.3,
  max_depth        = 4,
  min_child_weight = 50,
  subsample        = 0.5,
  colsample_bytree = 0.5,
  alpha            = 10.0,
  lambda           = 20.0,
  nthread          = parallel::detectCores(),
  verbosity        = 0
)
NROUNDS <- 50

# ---------- utils ----------
rmse_fn <- function(actual, pred) {
  m <- is.finite(actual) & is.finite(pred)
  sqrt(mean((actual[m] - pred[m])^2))
}

load_turbine <- function(tid, year) {
  path <- file.path(DATA_DIR, sprintf("Turbine%d_%d.csv", tid, year))
  if (!file.exists(path)) return(NULL)
  df <- fread(path, select = c(BASE_FEATURES, "wind_direction", TARGET))
  for (col in names(df)) df[[col]] <- suppressWarnings(as.numeric(df[[col]]))
  df <- df[complete.cases(df)]
  if (nrow(df) == 0) return(NULL)
  rad <- df$wind_direction * pi / 180
  df[, wind_direction_sin := sin(rad)]
  df[, wind_direction_cos := cos(rad)]
  df[, wind_direction := NULL]
  df
}

read_geo_donor_table <- function(path) {
  df <- fread(path)
  df[, target := as.integer(target)]
  if (all(c("donor", "geo_distance") %in% names(df))) {
    df[, donor        := as.integer(donor)]
    df[, geo_distance := as.numeric(geo_distance)]
    df <- df[complete.cases(df[, .(target, donor, geo_distance)])]
    return(df[order(target, geo_distance, donor)])
  }
  donor_cols <- grep("^donor", names(df), value = TRUE)
  donor_cols <- donor_cols[donor_cols != "donor"]
  score_cols <- grep("^score", names(df), value = TRUE)
  rows <- lapply(seq_along(donor_cols), function(i) {
    data.table(
      target       = df$target,
      donor        = suppressWarnings(as.integer(df[[donor_cols[i]]])),
      geo_distance = suppressWarnings(as.numeric(df[[score_cols[i]]]))
    )
  })
  long <- rbindlist(rows)
  long <- long[complete.cases(long)]
  long[order(target, geo_distance, donor)]
}

top_k_geo_donors <- function(geo_df, target_id, k, allowed_ids) {
  sub <- geo_df[target == target_id & donor != target_id & donor %in% allowed_ids]
  sub <- sub[order(geo_distance, donor)]
  as.integer(head(sub$donor, k))
}

append_row <- function(row_list, path) {
  row_df <- as.data.frame(row_list)
  if (!file.exists(path) || file.info(path)$size == 0) {
    write.csv(row_df, path, row.names = FALSE)
  } else {
    write.table(row_df, path, sep = ",", col.names = FALSE,
                row.names = FALSE, append = TRUE)
  }
}

# ---------- load donor table ----------
cat("[INFO] Reading geographic donor table ...\n")
geo_df    <- read_geo_donor_table(DONOR_FILE)
n_targets <- length(TEST_IDS)

# ---------- resume ----------
done_keys <- character(0)
if (file.exists(DETAIL_FILE) && file.info(DETAIL_FILE)$size > 0) {
  prev <- tryCatch(read.csv(DETAIL_FILE), error = function(e) NULL)
  if (!is.null(prev) && all(c("target", "year") %in% names(prev))) {
    done_keys <- paste0(prev$target, "|", prev$year)
    cat(sprintf("[INFO] Resuming — %d pairs already done.\n", length(done_keys)))
  }
}

cat(sprintf("[INFO] Starting G_XGBoost (DFP) — %d targets x %d years\n", n_targets, length(TEST_YEARS)))

# ---------- main loop ----------
for (t_idx in seq_along(TEST_IDS)) {
  target_id <- TEST_IDS[t_idx]
  donors    <- top_k_geo_donors(geo_df, target_id, K, TRAIN_IDS)
  if (length(donors) == 0) next

  if (all(paste0(target_id, "|", TEST_YEARS) %in% done_keys)) {
    cat(sprintf("  [SKIP] Target %d — all years done.\n", target_id))
    next
  }

  cat(sprintf("[%d/%d] Target %d | Donors: %s\n",
              t_idx, n_targets, target_id, paste(donors, collapse = ",")))

  frames <- list()
  for (d in donors) {
    df <- load_turbine(d, TRAIN_YEAR)
    if (!is.null(df)) frames[[length(frames) + 1]] <- df
  }
  if (length(frames) == 0) next
  train_df <- rbindlist(frames)
  X_train  <- as.matrix(train_df[, ..FEATURES])
  y_train  <- train_df[[TARGET]]
  dtrain   <- xgb.DMatrix(data = X_train, label = y_train)

  set.seed(target_id)
  t0       <- proc.time()["elapsed"]
  model    <- xgb.train(params = XGB_PARAMS, data = dtrain, nrounds = NROUNDS, verbose = 0)
  fit_time <- proc.time()["elapsed"] - t0
  cat(sprintf("  Fit done — %.1f sec\n", fit_time))

  for (test_year in TEST_YEARS) {
    pair_key <- paste0(target_id, "|", test_year)
    if (pair_key %in% done_keys) next

    test_df <- load_turbine(target_id, test_year)
    if (is.null(test_df)) next

    X_test <- as.matrix(test_df[, ..FEATURES])
    y_test <- test_df[[TARGET]]
    dtest  <- xgb.DMatrix(data = X_test)

    t1        <- proc.time()["elapsed"]
    pred      <- predict(model, dtest)
    pred_time <- proc.time()["elapsed"] - t1

    err <- rmse_fn(y_test, pred)
    cat(sprintf("  -> Year %d RMSE: %.4f\n", test_year, err))

    append_row(list(
      method      = "G_XGBoost",
      target      = target_id,
      year        = test_year,
      donors_used = paste(donors, collapse = ","),
      n_models    = length(donors),
      rmse        = err,
      runtime_sec = fit_time + pred_time
    ), DETAIL_FILE)

    done_keys <- c(done_keys, pair_key)
  }
}

# ---------- summary ----------
if (file.exists(DETAIL_FILE) && file.info(DETAIL_FILE)$size > 0) {
  detail_df  <- read.csv(DETAIL_FILE)
  summary_df <- data.frame(
    method            = "G_XGBoost",
    year              = sort(unique(detail_df$year)),
    avg_rmse          = tapply(detail_df$rmse,        detail_df$year, mean),
    total_runtime_sec = tapply(detail_df$runtime_sec, detail_df$year, sum)
  )
  write.csv(summary_df, SUMMARY_FILE, row.names = FALSE)
  cat("[DONE] Summary saved.\n")
  print(summary_df)
}
