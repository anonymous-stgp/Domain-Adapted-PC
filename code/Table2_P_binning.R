suppressPackageStartupMessages({
  library(data.table)
})

ROOT <- normalizePath(getwd())
if (!dir.exists(file.path(ROOT, "data"))) {
  ROOT <- normalizePath(file.path(ROOT, ".."))
}

DATA_DIR <- file.path(ROOT, "data")
RESULTS_DIR <- file.path(ROOT, "results", "intermediate")
dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

OUTPUT_DETAIL <- file.path(RESULTS_DIR, "Table2_P_Binning_detail.csv")
OUTPUT_SUMMARY <- file.path(RESULTS_DIR, "Table2_P_Binning_summary.csv")

TURBINE_IDS <- 1:66
TESTSET_2018 <- c(1:46, 48:50, 52, 54:60, 62:66)
TARGET <- "power"

rmse_vec <- function(y, yhat) sqrt(mean((y - yhat)^2, na.rm = TRUE))

binning_predict <- function(train_x, train_y, test_x, bin_width = 0.5) {
  train_x <- as.numeric(train_x)
  train_y <- as.numeric(train_y)
  test_x <- as.numeric(test_x)
  
  start <- 0
  end <- round(max(train_x, na.rm = TRUE))
  n_bins <- round((end - start) / bin_width, 0) + 1
  
  x_bin <- numeric(n_bins)
  y_bin <- numeric(n_bins)
  
  for (n in 2:n_bins) {
    idx <- which(train_x > (start + (n - 1) * bin_width) &
                   train_x < (start + n * bin_width))
    x_bin[n] <- mean(train_x[idx], na.rm = TRUE)
    y_bin[n] <- mean(train_y[idx], na.rm = TRUE)
  }
  
  binned <- data.frame(x_bin = x_bin, y_bin = y_bin)
  binned <- binned[!is.na(binned$y_bin), , drop = FALSE]
  
  fit <- smooth.spline(x = binned$x_bin, y = binned$y_bin, all.knots = TRUE)
  pred <- predict(fit, test_x)$y
  pred[pred < 0] <- 0
  pred
}

load_turbine_year <- function(turbine_id, year) {
  path <- file.path(DATA_DIR, sprintf("Turbine%d_%d.csv", as.integer(turbine_id), as.integer(year)))
  if (!file.exists(path)) return(NULL)
  
  dt <- fread(path, showProgress = FALSE)
  need <- c("wind_speed", TARGET)
  miss <- setdiff(need, names(dt))
  if (length(miss)) stop("Missing columns in ", path, ": ", paste(miss, collapse = ", "))
  
  dt <- dt[, ..need]
  dt[, (need) := lapply(.SD, as.numeric), .SDcols = need]
  dt <- na.omit(dt)
  if (!nrow(dt)) return(NULL)
  dt
}

cache_key <- function(id, year) sprintf("T%02d_%d", id, year)

data_cache <- vector("list", length = length(TURBINE_IDS) * 2L)
names(data_cache) <- as.vector(outer(sprintf("T%02d", TURBINE_IDS), c("2017", "2018"), paste, sep = "_"))

for (id in TURBINE_IDS) {
  for (yr in c(2017L, 2018L)) {
    data_cache[[cache_key(id, yr)]] <- load_turbine_year(id, yr)
  }
}

get_data <- function(id, year) data_cache[[cache_key(id, year)]]

speed_2017 <- list()
power_2017 <- list()
tid_2017 <- list()

for (id in TURBINE_IDS) {
  d <- get_data(id, 2017L)
  if (is.null(d)) {
    speed_2017[[id]] <- numeric(0)
    power_2017[[id]] <- numeric(0)
    tid_2017[[id]] <- integer(0)
  } else {
    speed_2017[[id]] <- d$wind_speed
    power_2017[[id]] <- d[[TARGET]]
    tid_2017[[id]] <- rep(id, nrow(d))
  }
}

train_speed_all <- unlist(speed_2017, use.names = FALSE)
train_power_all <- unlist(power_2017, use.names = FALSE)
train_tid_all <- unlist(tid_2017, use.names = FALSE)

detail_rows <- list()
idx <- 1L

for (year in c(2017L, 2018L)) {
  test_ids <- if (year == 2017L) TURBINE_IDS else TESTSET_2018
  
  for (target_id in test_ids) {
    cat("Binning - Turbine", target_id, "Year", year, "\n")
    
    test_dt <- get_data(target_id, year)
    if (is.null(test_dt)) next
    
    mask <- train_tid_all != target_id
    x_train <- train_speed_all[mask]
    y_train <- train_power_all[mask]
    
    x_test <- test_dt$wind_speed
    y_test <- test_dt[[TARGET]]
    
    set.seed(target_id)
    t0 <- proc.time()[["elapsed"]]
    pred <- binning_predict(x_train, y_train, x_test)
    runtime <- proc.time()[["elapsed"]] - t0
    
    detail_rows[[idx]] <- data.table(
      method = "P_Binning",
      target = target_id,
      year = year,
      rmse = rmse_vec(y_test, pred),
      runtime_sec = runtime
    )
    idx <- idx + 1L
  }
}

detail_dt <- rbindlist(detail_rows, fill = TRUE)
fwrite(detail_dt, OUTPUT_DETAIL)

summary_dt <- detail_dt[, .(
  avg_rmse = mean(rmse, na.rm = TRUE),
  total_runtime_sec = sum(runtime_sec, na.rm = TRUE)
), by = .(method, year)]

fwrite(summary_dt, OUTPUT_SUMMARY)