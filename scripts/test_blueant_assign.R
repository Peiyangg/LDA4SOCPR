#!/usr/bin/env Rscript
# Quick TIMED end-to-end test for assigning daily SST and daily sea-ice
# concentration to 5 CPR segments.
#
# Implementation note:
# - NOAA OI 0.25-deg daily SST v2.1 (NCEI: "oisst-avhrr-v02r01.YYYYMMDD.nc")
#   files already contain a daily sea-ice concentration variable (`ice`).
#   This is the same daily ice field that is fed into the OISST analysis and
#   is in fact one of the NSIDC CDR series (NRT or final, depending on era).
#   So we fetch ONE file per day and extract BOTH variables from it.
#
# Run from project root:
#   Rscript scripts/test_blueant_assign.R

suppressPackageStartupMessages({
  library(terra)
})

t_total_start <- Sys.time()

# ---------------- 0. config ----------------
proj_root <- normalizePath(".", mustWork = TRUE)
cache_dir <- file.path(proj_root, "data", "blueant_cache")
out_csv   <- file.path(proj_root, "data", "metadata", "blueant_test_assignment.csv")
dir.create(file.path(cache_dir, "sst"), showWarnings = FALSE, recursive = TRUE)

TEST_YEAR <- 2010L
N_SAMPLES <- 5L

cat("Cache dir:", cache_dir, "\n")
cat("Test year:", TEST_YEAR, "  n samples:", N_SAMPLES, "\n")

# ---------------- 1. pick the tiny test set ----------------
meta_path <- file.path(proj_root, "data", "metadata", "segment_metadata.csv")
meta <- read.csv(meta_path, stringsAsFactors = FALSE)
meta$Date <- as.Date(meta$Date, format = "%d-%b-%Y")

set.seed(42)
test_meta <- subset(meta, format(Date, "%Y") == as.character(TEST_YEAR))
test_meta <- test_meta[sample(nrow(test_meta), N_SAMPLES), ]
test_meta <- test_meta[order(test_meta$Date), ]
test_meta <- test_meta[, c("Segment_ID", "Date", "Latitude", "Longitude")]
cat("\nTest samples:\n"); print(test_meta)

needed_dates <- sort(unique(test_meta$Date))
cat("\nNeeded days:", length(needed_dates), "\n")

# ---------------- 2. build URLs ----------------
sst_url <- function(d) {
  ym  <- format(d, "%Y%m")
  ymd <- format(d, "%Y%m%d")
  sprintf("https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/%s/oisst-avhrr-v02r01.%s.nc", ym, ymd)
}

# ---------------- 3. download (timed) ----------------
fetch_one <- function(url, dst) {
  if (file.exists(dst) && file.info(dst)$size > 0) return(invisible(TRUE))
  tryCatch({
    download.file(url, destfile = dst, method = "libcurl", quiet = TRUE, mode = "wb")
    invisible(TRUE)
  }, error = function(e) {
    cat("  FAILED ", url, "\n  reason:", conditionMessage(e), "\n")
    if (file.exists(dst)) file.remove(dst)
    invisible(FALSE)
  })
}

cat("\n--- downloading OISST v2.1 (", length(needed_dates), "files; SST + ice in each) ---\n", sep = "")
t0 <- Sys.time()
sst_paths <- vapply(needed_dates, function(d) {
  url <- sst_url(d)
  dst <- file.path(cache_dir, "sst", basename(url))
  fetch_one(url, dst); dst
}, character(1))
t_dl <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
sst_ok <- file.exists(sst_paths) & file.info(sst_paths)$size > 0
sst_size_mb <- sum(file.info(sst_paths[sst_ok])$size) / 1024^2
cat(sprintf("OISST: %d/%d ok, %.1f MB, %.1f s\n",
            sum(sst_ok), length(sst_paths), sst_size_mb, t_dl))

stopifnot(any(sst_ok))

# Map Date -> file path lookup (one file per day; same file used for both vars)
sst_lookup <- setNames(sst_paths, format(needed_dates, "%Y-%m-%d"))

# ---------------- 4. extraction ----------------
# Note: OISST v2.1 netCDFs use longitudes 0..360, so shift western lons.
extract_var <- function(date, lon, lat, varname) {
  f <- sst_lookup[[format(date, "%Y-%m-%d")]]
  if (!file.exists(f)) return(NA_real_)
  r <- terra::rast(paste0("NETCDF:\"", f, "\":", varname))
  lon360 <- ifelse(lon < 0, lon + 360, lon)
  v <- terra::extract(r, cbind(lon360, lat))[, 1]
  as.numeric(v)
}

extract_sst <- function(date, lon, lat) extract_var(date, lon, lat, "sst")
extract_sic <- function(date, lon, lat) extract_var(date, lon, lat, "ice")

cat("\n--- extracting ---\n")
t0 <- Sys.time()
test_meta$sst_c    <- mapply(extract_sst,
                             test_meta$Date, test_meta$Longitude, test_meta$Latitude)
test_meta$sic_frac <- mapply(extract_sic,
                             test_meta$Date, test_meta$Longitude, test_meta$Latitude)
t_extract <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("Extraction finished in %.2f s\n", t_extract))

# ---------------- 5. report ----------------
cat("\n--- RESULT ---\n")
print(test_meta)
write.csv(test_meta, out_csv, row.names = FALSE)
cat("\nWrote ->", out_csv, "\n")

t_total <- as.numeric(difftime(Sys.time(), t_total_start, units = "secs"))

cat("\n===== TIMING SUMMARY =====\n")
cat(sprintf("  download   : %.1f s  (%.1f MB, %d file%s)\n",
            t_dl, sst_size_mb, sum(sst_ok), ifelse(sum(sst_ok)==1, "", "s")))
cat(sprintf("  extraction : %.2f s\n", t_extract))
cat(sprintf("  TOTAL      : %.1f s\n", t_total))

# ---------------- 6. extrapolation ----------------
n_dates_full <- length(unique(meta$Date))
n_segments_full <- nrow(meta)
n_dates_test <- length(needed_dates)

if (n_dates_test > 0) {
  rate_sec_per_date <- t_dl / n_dates_test
  est_full <- rate_sec_per_date * n_dates_full
  cat("\n===== EXTRAPOLATION =====\n")
  cat(sprintf("Full dataset: %d segments across %d unique dates.\n",
              n_segments_full, n_dates_full))
  cat(sprintf("This test: %.2f s/date (download).\n", rate_sec_per_date))
  cat(sprintf("Estimated full download (sequential): %.1f min (%.1f h).\n",
              est_full / 60, est_full / 3600))
  cat("Note: parallelizing with e.g. {curl}::multi_download can easily 5-10x this.\n")
}
