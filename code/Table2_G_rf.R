suppressPackageStartupMessages({
  library(data.table)
  library(ranger)
})

ROOT <- normalizePath(getwd())
if (!dir.exists(file.path(ROOT, "data"))) {
  ROOT <- normalizePath(file.path(ROOT, ".."))
}

DATA_DIR <- file.path(ROOT, "data")
PROCESSED_DIR <- file.path(DATA_DIR, "processed_data")
RESULTS_DIR <- file.path(ROOT, "results", "intermediate")
dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

DONOR_FILE <- file.path(PROCESSED_DIR, "matching_geographic_distance.csv")
OUTPUT_DETAIL <- file.path(RESULTS_DIR, "Table2_G_random_forest_detail.csv")
OUTPUT_SUMMARY <- file.path(RESULTS_DIR, "Table2_G_random_forest_summary.csv")

TARGET <- "power"
BASE_FEATURES <- c("wind_speed", "temperature", "turbulence_intensity", "std_wind_direction")
ANGLE_FEATURE <- "wind_direction"

K <- 7L
TRAIN_YEAR <- 2017L
TEST_YEARS <- c(2017L, 2018L)

RF_NUM_TREES <- 50L
RF_NUM_THREADS <- max(1L, parallel::detectCores() - 1L)
RF_SEED <- 15L

rmse_vec <- function(y, yhat) sqrt(mean((y - yhat)^2, na.rm = TRUE))

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

read_geo_donor_table <- function(path) {
  dt <- fread(path, showProgress = FALSE)
  
  if (all(c("donor", "geo_distance") %in% names(dt))) {
    # already long format
    dt[, target := as.integer(target)]
    dt[, donor := as.integer(donor)]
    dt[, geo_distance := as.numeric(geo_distance)]
    dt <- dt[!is.na(target) & !is.na(donor) & !is.na(geo_distance)]
    setorder(dt, target, geo_distance, donor)
    return(dt)
  }
  
  # wide format: target, donor_1, score_1, donor_2, score_2 ...
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

build_pooled_training <- function(donor_ids) {
  frames <- lapply(donor_ids, function(d) load_turbine_year(d, TRAIN_YEAR))
  frames <- Filter(Negate(is.null), frames)
  if (!length(frames)) return(NULL)
  rbindlist(frames, use.names = TRUE, fill = TRUE)
}

run_target_year <- function(target_id, year, donor_ids) {
  feats <- feature_names()
  
  train_dt <- build_pooled_training(donor_ids)
  test_dt <- load_turbine_year(target_id, year)
  if (is.null(train_dt) || is.null(test_dt)) return(NULL)
  
  x_train <- train_dt[, ..feats]
  y_train <- train_dt[[TARGET]]
  x_test <- test_dt[, ..feats]
  y_test <- test_dt[[TARGET]]
  
  t0 <- proc.time()[["elapsed"]]
  model <- ranger(
    dependent.variable.name = TARGET,
    data = data.frame(power = y_train, x_train),
    num.trees = RF_NUM_TREES,
    num.threads = RF_NUM_THREADS,
    seed = RF_SEED
  )
  fit_time <- proc.time()[["elapsed"]] - t0
  
  t1 <- proc.time()[["elapsed"]]
  pred <- predict(model, data = as.data.frame(x_test))$predictions
  pred_time <- proc.time()[["elapsed"]] - t1
  
  data.table(
    method = "G_random_forest",
    target = target_id,
    year = year,
    donors_used = paste(donor_ids, collapse = ","),
    n_models = 1L,
    rmse = rmse_vec(y_test, pred),
    runtime_sec = fit_time + pred_time,
    fit_time_sec = fit_time,
    pred_time_sec = pred_time
  )
}

main <- function() {
  geo_dt <- read_geo_donor_table(DONOR_FILE)
  targets <- sort(unique(geo_dt$target))
  
  rows <- list()
  idx <- 1L
  
  for (target_id in targets) {
    donor_ids <- get_top_k_donors(geo_dt, target_id, K)
    if (!length(donor_ids)) next
    
    for (year in TEST_YEARS) {
      cat("G_random_forest - Turbine", target_id, "Year", year, "\n")
      out <- tryCatch(run_target_year(target_id, year, donor_ids), error = function(e) NULL)
      if (is.null(out)) next
      rows[[idx]] <- out
      idx <- idx + 1L
    }
  }
  
  detail_dt <- rbindlist(rows, fill = TRUE)
  fwrite(detail_dt, OUTPUT_DETAIL)
  
  summary_dt <- detail_dt[, .(
    avg_rmse = mean(rmse, na.rm = TRUE),
    total_runtime_sec = sum(runtime_sec, na.rm = TRUE)
  ), by = .(method, year)]
  
  fwrite(summary_dt, OUTPUT_SUMMARY)
}

main()