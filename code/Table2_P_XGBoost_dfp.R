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

DETAIL_FILE  <- file.path(RESULTS_DIR, "Table2_P_XGBoost_detail_dfp.csv")
SUMMARY_FILE <- file.path(RESULTS_DIR, "Table2_P_XGBoost_summary_dfp.csv")

# ---------- config ----------
TARGET        <- "power"
BASE_FEATURES <- c("wind_speed", "temperature", "turbulence_intensity", "std_wind_direction")
FEATURES      <- c(BASE_FEATURES, "wind_direction_sin", "wind_direction_cos")
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

append_row <- function(row_list, path) {
  row_df <- as.data.frame(row_list)
  if (!file.exists(path) || file.info(path)$size == 0) {
    write.csv(row_df, path, row.names = FALSE)
  } else {
    write.table(row_df, path, sep = ",", col.names = FALSE,
                row.names = FALSE, append = TRUE)
  }
}

# ---------- resume ----------
done_keys <- character(0)
if (file.exists(DETAIL_FILE) && file.info(DETAIL_FILE)$size > 0) {
  prev <- tryCatch(read.csv(DETAIL_FILE), error = function(e) NULL)
  if (!is.null(prev) && all(c("target", "year") %in% names(prev))) {
    done_keys <- paste0(prev$target, "|", prev$year)
    cat(sprintf("[INFO] Resuming — %d rows already done.\n", length(done_keys)))
  }
}

# ---------- build pooled training data (TRAIN_IDS, 2017 only) ----------
cat("[INFO] Loading pooled training data ...\n")
train_frames <- list()
for (tid in TRAIN_IDS) {
  df <- load_turbine(tid, TRAIN_YEAR)
  if (!is.null(df)) train_frames[[length(train_frames) + 1]] <- df
}
train_all <- rbindlist(train_frames)
X_train   <- as.matrix(train_all[, ..FEATURES])
y_train   <- train_all[[TARGET]]
dtrain    <- xgb.DMatrix(data = X_train, label = y_train)
cat(sprintf("[INFO] Pooled training set: %d rows from %d turbines.\n",
            nrow(X_train), length(train_frames)))

# ---------- fit one model on pooled data ----------
cat("[INFO] Fitting XGBoost model ...\n")
set.seed(SEED)
t0       <- proc.time()["elapsed"]
model    <- xgb.train(params = XGB_PARAMS, data = dtrain, nrounds = NROUNDS, verbose = 0)
fit_time <- proc.time()["elapsed"] - t0
cat(sprintf("[INFO] Model fit done — %.1f sec\n", fit_time))

# ---------- predict on each test turbine / year ----------
cat(sprintf("[INFO] Starting P_XGBoost (DFP) — %d targets x %d years\n",
            length(TEST_IDS), length(TEST_YEARS)))

for (target_id in TEST_IDS) {
  for (test_year in TEST_YEARS) {
    pair_key <- paste0(target_id, "|", test_year)
    if (pair_key %in% done_keys) {
      cat(sprintf("  [SKIP] Turbine %d Year %d\n", target_id, test_year))
      next
    }

    test_df <- load_turbine(target_id, test_year)
    if (is.null(test_df)) next

    X_test <- as.matrix(test_df[, ..FEATURES])
    y_test <- test_df[[TARGET]]
    dtest  <- xgb.DMatrix(data = X_test)

    t1        <- proc.time()["elapsed"]
    pred      <- predict(model, dtest)
    pred_time <- proc.time()["elapsed"] - t1

    err <- rmse_fn(y_test, pred)
    cat(sprintf("  Turbine %d Year %d — RMSE: %.4f\n", target_id, test_year, err))

    append_row(list(
      method      = "P_XGBoost",
      target      = target_id,
      year        = test_year,
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
    method            = "P_XGBoost",
    year              = sort(unique(detail_df$year)),
    avg_rmse          = tapply(detail_df$rmse,        detail_df$year, mean),
    total_runtime_sec = tapply(detail_df$runtime_sec, detail_df$year, sum)
  )
  write.csv(summary_df, SUMMARY_FILE, row.names = FALSE)
  cat("[DONE] Summary saved.\n")
  print(summary_df)
}
