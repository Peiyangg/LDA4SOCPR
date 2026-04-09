#!/usr/bin/env Rscript
# Download ALL BlueAnt data sources
# This script systematically downloads all available data from BlueAnt
# WARNING: This will download a large amount of data (potentially hundreds of GBs)
# Author: Generated for comprehensive data retrieval
# Date: 2026-04-07

# Install and load required packages
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}
remotes::install_github("AustralianAntarcticDivision/blueant")
library(blueant)

# Suppress warnings for cleaner output
options(warn = -1)

# Main data directory
main_data_dir <- "BlueAnt_AllData"
dir.create(main_data_dir, showWarnings = FALSE, recursive = TRUE)

# Define all BlueAnt source categories
categories <- list(
  biological = sources_biological,
  seaice = sources_seaice,
  sst = sources_sst,
  ocean_color = sources_ocean_color,
  topography = sources_topography,
  oceanographic = sources_oceanographic,
  altimetry = sources_altimetry,
  meteorological = sources_meteorological,
  reanalysis = sources_reanalysis,
  sdm = sources_sdm
)

# Initialize tracking
download_log <- list()
total_sources <- 0
successful_downloads <- 0
failed_downloads <- 0

cat("\n")
cat("========================================\n")
cat("  BlueAnt Complete Data Download\n")
cat("========================================\n")
cat("\nThis will download ALL available BlueAnt data sources.\n")
cat(sprintf("Data will be saved to: %s/\n", main_data_dir))
cat("\n")

# First, count total sources
for (cat_name in names(categories)) {
  sources_df <- tryCatch(
    categories[[cat_name]](),
    error = function(e) data.frame()
  )
  total_sources <- total_sources + nrow(sources_df)
}

cat(sprintf("Total data sources to download: %d\n", total_sources))
cat("========================================\n\n")

# Process each category
counter <- 1

for (cat_name in names(categories)) {
  cat(sprintf("\n>>> CATEGORY: %s <<<\n", toupper(cat_name)))
  
  # Create category-specific directory
  category_dir <- file.path(main_data_dir, cat_name)
  dir.create(category_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Get all sources in this category
  sources_df <- tryCatch(
    categories[[cat_name]](),
    error = function(e) {
      cat(sprintf("ERROR: Could not load sources for category %s\n", cat_name))
      return(data.frame())
    }
  )
  
  if (nrow(sources_df) == 0) {
    cat("No sources found in this category\n")
    next
  }
  
  cat(sprintf("Found %d sources in %s\n\n", nrow(sources_df), cat_name))
  
  # Download each source in the category
  for (i in 1:nrow(sources_df)) {
    source_name <- sources_df$name[i]
    safe_name <- gsub("[^A-Za-z0-9_-]", "_", source_name)
    
    cat(sprintf("[%d/%d] Downloading: %s\n", counter, total_sources, source_name))
    cat(sprintf("        Category: %s\n", cat_name))
    
    # Create source-specific directory
    source_dir <- file.path(category_dir, safe_name)
    dir.create(source_dir, showWarnings = FALSE, recursive = TRUE)
    
    # Configure bb_config for this source
    config <- bb_config(local_file_root = source_dir)
    
    # Add the source to config
    config <- tryCatch(
      config %>% bb_add(sources_df[i, ]),
      error = function(e) {
        cat(sprintf("        ERROR adding source: %s\n", e$message))
        return(NULL)
      }
    )
    
    if (is.null(config)) {
      download_log[[counter]] <- list(
        index = counter,
        category = cat_name,
        name = source_name,
        status = "FAILED (config error)",
        directory = source_dir
      )
      failed_downloads <- failed_downloads + 1
      counter <- counter + 1
      next
    }
    
    # Attempt to download/sync the data
    start_time <- Sys.time()
    status <- tryCatch(
      bb_sync(config, verbose = FALSE),
      error = function(e) {
        cat(sprintf("        ERROR downloading: %s\n", e$message))
        return(NULL)
      }
    )
    end_time <- Sys.time()
    
    if (!is.null(status)) {
      time_taken <- difftime(end_time, start_time, units = "secs")
      cat(sprintf("        SUCCESS (%.1f seconds)\n", time_taken))
      cat(sprintf("        Location: %s\n", source_dir))
      
      download_log[[counter]] <- list(
        index = counter,
        category = cat_name,
        name = source_name,
        status = "SUCCESS",
        directory = source_dir,
        time_seconds = as.numeric(time_taken)
      )
      successful_downloads <- successful_downloads + 1
    } else {
      cat("        FAILED\n")
      download_log[[counter]] <- list(
        index = counter,
        category = cat_name,
        name = source_name,
        status = "FAILED (download error)",
        directory = source_dir
      )
      failed_downloads <- failed_downloads + 1
    }
    
    cat("\n")
    counter <- counter + 1
    
    # Brief pause to avoid overwhelming servers
    Sys.sleep(2)
  }
}

# Save download log
log_df <- do.call(rbind, lapply(download_log, function(x) {
  data.frame(
    index = x$index,
    category = x$category,
    name = x$name,
    status = x$status,
    directory = x$directory,
    time_seconds = ifelse(is.null(x$time_seconds), NA, x$time_seconds),
    stringsAsFactors = FALSE
  )
}))

log_file <- file.path(main_data_dir, "download_log.csv")
write.csv(log_df, log_file, row.names = FALSE)

# Final summary
cat("\n")
cat("========================================\n")
cat("  DOWNLOAD COMPLETE - SUMMARY\n")
cat("========================================\n")
cat(sprintf("Total sources processed: %d\n", total_sources))
cat(sprintf("Successful downloads: %d\n", successful_downloads))
cat(sprintf("Failed downloads: %d\n", failed_downloads))
cat(sprintf("Success rate: %.1f%%\n", 100 * successful_downloads / total_sources))
cat(sprintf("\nData saved to: %s/\n", main_data_dir))
cat(sprintf("Download log: %s\n", log_file))
cat("========================================\n")

# Display failed downloads if any
if (failed_downloads > 0) {
  cat("\nFailed downloads:\n")
  failed <- log_df[log_df$status != "SUCCESS", ]
  for (i in 1:nrow(failed)) {
    cat(sprintf("  - [%s] %s\n", failed$category[i], failed$name[i]))
  }
}

cat("\nDone!\n")
