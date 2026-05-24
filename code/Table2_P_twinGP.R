suppressPackageStartupMessages({
  library(data.table)
  library(twingp)
})

args       <- commandArgs(trailingOnly = TRUE)
YEARS_RUN  <- if (length(args) >= 1L) as.integer(args[1]) else c(2017L, 2018L)
year_suffix <- if (length(args) >= 1L) paste0("_", args[1]) else ""

ROOT <- normalizePath(getwd())
if (!dir.exists(file.path(ROOT, "data"))) {
  ROOT <- normalizePath(file.path(ROOT, ".."))
}

DATA_DIR    <- file.path(ROOT, "data")
RESULTS_DIR <- file.path(ROOT, "results", "intermediate")
dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

OUTPUT_DETAIL  <- file.path(RESULTS_DIR, paste0("Table2_P_twinGP_detail",  year_suffix, ".csv"))
OUTPUT_SUMMARY <- file.path(RESULTS_DIR, paste0("Table2_P_twinGP_summary", year_suffix, ".csv"))

TURBINE_IDS  <- 1:66
TESTSET_2018 <- c(1:46, 48:50, 52, 54:60, 62:66)
TARGET       <- "power"

BASE_FEATURES <- c("wind_speed", "temperature", "turbulence_intensity", "std_wind_direction")
ANGLE_FEATURE <- "wind_direction"

rmse_vec <- function(y, yhat) sqrt(mean((y - yhat)^2, na.rm = TRUE))
nlpd_vec <- function(y, mu, sigma) {
  mean(0.5 * log(2 * pi * sigma^2) + 0.5 * ((y - mu)^2) / (sigma^2), na.rm = TRUE)
}
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

cache_key <- function(id, year) sprintf("T%02d_%d", id, year)

# --- load data cache ---
cat("[INFO] Loading data cache ...\n"); flush.console()
data_cache <- vector("list", length(TURBINE_IDS) * 2L)
names(data_cache) <- as.vector(outer(sprintf("T%02d", TURBINE_IDS), c("2017", "2018"), paste, sep = "_"))
for (id in TURBINE_IDS)
  for (yr in c(2017L, 2018L))
    data_cache[[cache_key(id, yr)]] <- load_turbine_year(id, yr)

get_data <- function(id, year) data_cache[[cache_key(id, year)]]

# --- build pooled 2017 ---
feats <- feature_names()
X2017_list <- y2017_list <- tid2017_list <- vector("list", length(TURBINE_IDS))
for (id in TURBINE_IDS) {
  d <- get_data(id, 2017L)
  if (is.null(d)) {
    X2017_list[[id]]   <- matrix(numeric(0), ncol = length(feats))
    y2017_list[[id]]   <- numeric(0)
    tid2017_list[[id]] <- integer(0)
  } else {
    X2017_list[[id]]   <- as.matrix(d[, ..feats])
    y2017_list[[id]]   <- d[[TARGET]]
    tid2017_list[[id]] <- rep(id, nrow(d))
  }
}
X2017   <- do.call(rbind, X2017_list)
y2017   <- as.numeric(unlist(y2017_list,   use.names = FALSE))
tid2017 <- as.integer(unlist(tid2017_list, use.names = FALSE))

# --- resume ---
done_keys <- character(0)
if (file.exists(OUTPUT_DETAIL) && file.size(OUTPUT_DETAIL) > 0) {
  prev <- tryCatch(fread(OUTPUT_DETAIL, showProgress = FALSE), error = function(e) data.table())
  if (nrow(prev) && all(c("target", "year") %in% names(prev))) {
    done_keys <- unique(sprintf("%d|%d", as.integer(prev$target), as.integer(prev$year)))
    cat(sprintf("[INFO] Resuming — %d pairs already done.\n", length(done_keys)))
    flush.console()
  }
}

# --- main loop ---
for (year in YEARS_RUN) {
  test_ids <- if (year == 2017L) TURBINE_IDS else TESTSET_2018

  for (target_id in test_ids) {
    pair_key <- sprintf("%d|%d", target_id, year)
    if (pair_key %in% done_keys) {
      cat(sprintf("  [SKIP] Turbine %d Year %d\n", target_id, year))
      next
    }

    cat(sprintf("TwinGP - Turbine %d Year %d\n", target_id, year))
    flush.console()

    test_dt <- get_data(target_id, year)
    if (is.null(test_dt)) next

    mask    <- tid2017 != target_id
    x_train <- X2017[mask, , drop = FALSE]
    y_train <- y2017[mask]
    x_test  <- as.matrix(test_dt[, ..feats])
    y_test  <- test_dt[[TARGET]]

    set.seed(target_id)
    t0       <- proc.time()[["elapsed"]]
    twin_out <- twingp(x = x_train, y = y_train, x_test = x_test)
    runtime  <- proc.time()[["elapsed"]] - t0

    mu    <- as.numeric(twin_out$mu)
    sigma <- as.numeric(twin_out$sigma)

    row_dt <- data.table(
      method      = "P_twinGP",
      target      = target_id,
      year        = year,
      rmse        = rmse_vec(y_test, mu),
      nlpd        = nlpd_vec(y_test, mu, sigma),
      runtime_sec = runtime
    )

    # incremental save
    fwrite(row_dt, OUTPUT_DETAIL, append = file.exists(OUTPUT_DETAIL))
    done_keys <- c(done_keys, pair_key)

    cat(sprintf("  -> Saved. RMSE: %.4f | %.1f sec\n", row_dt$rmse, runtime))
    flush.console()
  }
}

# --- final summary ---
if (file.exists(OUTPUT_DETAIL) && file.size(OUTPUT_DETAIL) > 0) {
  detail_dt  <- fread(OUTPUT_DETAIL, showProgress = FALSE)
  summary_dt <- detail_dt[, .(
    avg_rmse          = mean(rmse, na.rm = TRUE),
    avg_nlpd          = mean(nlpd, na.rm = TRUE),
    total_runtime_sec = sum(runtime_sec, na.rm = TRUE)
  ), by = .(method, year)]
  fwrite(summary_dt, OUTPUT_SUMMARY)
  cat("[DONE] Summary saved.\n")
}