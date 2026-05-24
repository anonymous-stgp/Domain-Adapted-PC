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

TEST_IDS <- c(38:44)
ALL_IDS <- 1:66
TRAIN_IDS <- setdiff(ALL_IDS, TEST_IDS)
SETUP_SUFFIX <- "_dfp"

OUTPUT_DETAIL <- file.path(RESULTS_DIR, paste0("Table2_P_twinGP_detail", SETUP_SUFFIX, ".csv"))
OUTPUT_SUMMARY <- file.path(RESULTS_DIR, paste0("Table2_P_twinGP_summary", SETUP_SUFFIX, ".csv"))

TARGET <- "power"
TEST_YEARS <- c(2017L, 2018L)

BASE_FEATURES <- c("wind_speed", "temperature", "turbulence_intensity", "std_wind_direction")
ANGLE_FEATURE <- "wind_direction"

rmse_vec <- function(y, yhat) sqrt(mean((y - yhat)^2, na.rm = TRUE))

nlpd_vec <- function(y, mu, sigma) {
  mean(
    0.5 * log(2 * pi * sigma^2) + 0.5 * ((y - mu)^2) / (sigma^2),
    na.rm = TRUE
  )
}

feature_names <- function() {
  c(BASE_FEATURES, "wind_direction_sin", "wind_direction_cos")
}

load_turbine_year <- function(turbine_id, year) {
  path <- file.path(DATA_DIR, sprintf("Turbine%d_%d.csv", as.integer(turbine_id), as.integer(year)))
  if (!file.exists(path)) return(NULL)
  
  dt <- fread(path, showProgress = FALSE)
  need <- c(BASE_FEATURES, ANGLE_FEATURE, TARGET)
  miss <- setdiff(need, names(dt))
  if (length(miss)) stop("Missing columns in ", path, ": ", paste(miss, collapse = ", "))
  
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

data_cache <- vector("list", length = length(ALL_IDS) * 2L)
names(data_cache) <- as.vector(outer(sprintf("T%02d", ALL_IDS), c("2017", "2018"), paste, sep = "_"))

for (id in ALL_IDS) {
  for (yr in c(2017L, 2018L)) {
    data_cache[[cache_key(id, yr)]] <- load_turbine_year(id, yr)
  }
}

get_data <- function(id, year) data_cache[[cache_key(id, year)]]

feats <- feature_names()

Xtrain_list <- vector("list", length(TRAIN_IDS))
ytrain_list <- vector("list", length(TRAIN_IDS))

for (j in seq_along(TRAIN_IDS)) {
  id <- TRAIN_IDS[j]
  d <- get_data(id, 2017L)
  if (is.null(d)) {
    Xtrain_list[[j]] <- matrix(numeric(0), ncol = length(feats))
    ytrain_list[[j]] <- numeric(0)
  } else {
    Xtrain_list[[j]] <- as.matrix(d[, ..feats])
    ytrain_list[[j]] <- d[[TARGET]]
  }
}

Xtrain2017 <- do.call(rbind, Xtrain_list)
ytrain2017 <- as.numeric(unlist(ytrain_list, use.names = FALSE))

detail_rows <- list()
idx <- 1L

for (year in TEST_YEARS) {
  for (target_id in TEST_IDS) {
    cat("P_twinGP DFP - Turbine", target_id, "Year", year, "\n")
    
    test_dt <- get_data(target_id, year)
    if (is.null(test_dt)) next
    if (!nrow(Xtrain2017)) next
    
    x_test <- as.matrix(test_dt[, ..feats])
    y_test <- test_dt[[TARGET]]
    
    set.seed(target_id)
    t0 <- proc.time()[["elapsed"]]
    twin_out <- twingp(x = Xtrain2017, y = ytrain2017, x_test = x_test)
    runtime <- proc.time()[["elapsed"]] - t0
    
    mu <- as.numeric(twin_out$mu)
    sigma <- as.numeric(twin_out$sigma)
    
    detail_rows[[idx]] <- data.table(
      method = "P_twinGP",
      target = target_id,
      year = year,
      train_turbines = paste(TRAIN_IDS, collapse = ","),
      n_train_turbines = length(TRAIN_IDS),
      rmse = rmse_vec(y_test, mu),
      nlpd = nlpd_vec(y_test, mu, sigma),
      runtime_sec = runtime
    )
    idx <- idx + 1L
  }
}

detail_dt <- rbindlist(detail_rows, fill = TRUE)
fwrite(detail_dt, OUTPUT_DETAIL)

summary_dt <- detail_dt[, .(
  avg_rmse = mean(rmse, na.rm = TRUE),
  avg_nlpd = mean(nlpd, na.rm = TRUE),
  total_runtime_sec = sum(runtime_sec, na.rm = TRUE)
), by = .(method, year)]

fwrite(summary_dt, OUTPUT_SUMMARY)