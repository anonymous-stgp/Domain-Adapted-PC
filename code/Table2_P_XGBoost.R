suppressPackageStartupMessages({
  library(xgboost)
  library(data.table)
})

# ---------- paths ----------
ROOT <- normalizePath(".")
if (!dir.exists(file.path(ROOT, "data"))) ROOT <- dirname(ROOT)

DATA_DIR    <- file.path(ROOT, "data")
RESULTS_DIR <- file.path(ROOT, "results", "intermediate")
dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

DETAIL_FILE  <- file.path(RESULTS_DIR, "Table2_P_XGBoost_detail.csv")
SUMMARY_FILE <- file.path(RESULTS_DIR, "Table2_P_XGBoost_summary.csv")

# ---------- config ----------
TARGET        <- "power"
BASE_FEATURES <- c("wind_speed", "temperature", "turbulence_intensity", "std_wind_direction")
FEATURES      <- c(BASE_FEATURES, "wind_direction_sin", "wind_direction_cos")
TRAIN_YEAR    <- 2017
TEST_YEARS    <- c(2017, 2018)
SEED          <- 2026

ALL_IDS     <- 1:66
# 2018 test set matches Python: 1-46, 48-50, 52, 54-60, 62-66
TESTSET_2018 <- c(1:46, 48, 49, 50, 52, 54:60, 62:66)

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

append_row <- function(row_list, path) {
  row_df <- as.data.frame(row_list)
  if (!file.exists(path) || file.info(path)$size == 0) {
    write.csv(row_df, path, row.names = FALSE)
  } else {
    write.table(row_df, path, sep = ",", col.names = FALSE,
                row.names = FALSE, append = TRUE)
  }
}

# ---------- load and cache all data ----------
cat("[INFO] Loading data cache ...\n")
data_cache <- list()
for (tid in ALL_IDS) {
  for (yr in c(2017, 2018)) {
    data_cache[[paste0(tid, "_", yr)]] <- load_turbine(tid, yr)
  }
}

# ---------- build full pooled 2017 ----------
cat("[INFO] Building pooled 2017 dataset ...\n")
frames_2017 <- list()
tid_vec     <- c()
for (tid in ALL_IDS) {
  df <- data_cache[[paste0(tid, "_2017")]]
  if (!is.null(df)) {
    df[, turbine_id := tid]
    frames_2017[[length(frames_2017) + 1]] <- df
    tid_vec <- c(tid_vec, rep(tid, nrow(df)))
  }
}
pool_2017    <- rbindlist(frames_2017)
X_pool_2017  <- as.matrix(pool_2017[, ..FEATURES])
y_pool_2017  <- pool_2017[[TARGET]]
tid_pool_vec <- tid_vec
cat(sprintf("[INFO] Pooled 2017: %d rows from %d turbines.\n",
            length(y_pool_2017), length(frames_2017)))

# ---------- resume ----------
done_keys <- character(0)
if (file.exists(DETAIL_FILE) && file.info(DETAIL_FILE)$size > 0) {
  prev <- tryCatch(read.csv(DETAIL_FILE), error = function(e) NULL)
  if (!is.null(prev) && all(c("target", "year") %in% names(prev))) {
    done_keys <- paste0(prev$target, "|", prev$year)
    cat(sprintf("[INFO] Resuming — %d rows already done.\n", length(done_keys)))
  }
}

# ---------- main loop ----------
for (test_year in TEST_YEARS) {
  test_ids <- if (test_year == 2017) ALL_IDS else TESTSET_2018

  for (target_id in test_ids) {
    pair_key <- paste0(target_id, "|", test_year)
    if (pair_key %in% done_keys) {
      cat(sprintf("  [SKIP] Turbine %d Year %d\n", target_id, test_year))
      next
    }

    test_df <- data_cache[[paste0(target_id, "_", test_year)]]
    if (is.null(test_df)) next

    # leave-one-out: exclude target turbine from training
    mask    <- tid_pool_vec != target_id
    X_train <- X_pool_2017[mask, , drop = FALSE]
    y_train <- y_pool_2017[mask]
    X_test  <- as.matrix(test_df[, ..FEATURES])
    y_test  <- test_df[[TARGET]]

    dtrain <- xgb.DMatrix(data = X_train, label = y_train)
    dtest  <- xgb.DMatrix(data = X_test)

    cat(sprintf("Turbine %d Year %d — P_XGBoost\n", target_id, test_year))
    set.seed(target_id)
    t0      <- proc.time()["elapsed"]
    model   <- xgb.train(params = XGB_PARAMS, data = dtrain, nrounds = NROUNDS, verbose = 0)
    pred    <- predict(model, dtest)
    runtime <- proc.time()["elapsed"] - t0

    err <- rmse_fn(y_test, pred)
    cat(sprintf("  -> RMSE: %.4f\n", err))

    append_row(list(
      method      = "P_XGBoost",
      target      = target_id,
      year        = test_year,
      rmse        = err,
      runtime_sec = runtime
    ), DETAIL_FILE)

    done_keys <- c(done_keys, pair_key)
  }
}

# ---------- summary ----------
if (file.exists(DETAIL_FILE) && file.info(DETAIL_FILE)$size > 0) {
  detail_df  <- read.csv(DETAIL_FILE)
  summary_df <- data.frame(
    method            = "P_XGBoost",
    year              = sort(unique(detail_df$year)),
    avg_rmse          = tapply(detail_df$rmse,        detail_df$year, mean),
    total_runtime_sec = tapply(detail_df$runtime_sec, detail_df$year, sum)
  )
  write.csv(summary_df, SUMMARY_FILE, row.names = FALSE)
  cat("[DONE] Summary saved.\n")
  print(summary_df)
}
