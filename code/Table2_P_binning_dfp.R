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

TEST_IDS <- c(38:44)
ALL_IDS <- 1:66
TRAIN_IDS <- setdiff(ALL_IDS, TEST_IDS)
SETUP_SUFFIX <- "_dfp"

OUTPUT_DETAIL <- file.path(RESULTS_DIR, paste0("Table2_P_Binning_detail", SETUP_SUFFIX, ".csv"))
OUTPUT_SUMMARY <- file.path(RESULTS_DIR, paste0("Table2_P_Binning_summary", SETUP_SUFFIX, ".csv"))

TARGET <- "power"
TEST_YEARS <- c(2017L, 2018L)

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

data_cache <- vector("list", length = length(ALL_IDS) * 2L)
names(data_cache) <- as.vector(outer(sprintf("T%02d", ALL_IDS), c("2017", "2018"), paste, sep = "_"))

for (id in ALL_IDS) {
  for (yr in c(2017L, 2018L)) {
    data_cache[[cache_key(id, yr)]] <- load_turbine_year(id, yr)
  }
}

get_data <- function(id, year) data_cache[[cache_key(id, year)]]

speed_train <- list()
power_train <- list()

for (j in seq_along(TRAIN_IDS)) {
  id <- TRAIN_IDS[j]
  d <- get_data(id, 2017L)
  if (is.null(d)) {
    speed_train[[j]] <- numeric(0)
    power_train[[j]] <- numeric(0)
  } else {
    speed_train[[j]] <- d$wind_speed
    power_train[[j]] <- d[[TARGET]]
  }
}

train_speed_all <- unlist(speed_train, use.names = FALSE)
train_power_all <- unlist(power_train, use.names = FALSE)

detail_rows <- list()
idx <- 1L

for (year in TEST_YEARS) {
  for (target_id in TEST_IDS) {
    cat("P_Binning DFP - Turbine", target_id, "Year", year, "\n")
    
    test_dt <- get_data(target_id, year)
    if (is.null(test_dt)) next
    if (!length(train_speed_all)) next
    
    x_train <- train_speed_all
    y_train <- train_power_all
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
      train_turbines = paste(TRAIN_IDS, collapse = ","),
      n_train_turbines = length(TRAIN_IDS),
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