#!/usr/bin/env Rscript
# Download SST and Sea Ice data via BlueAnt
# Targeted download — only SST and sea ice sources (not the full BlueAnt catalogue)
#
# Run: Rscript notebooks/download_sst_seaice.R

if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
if (!requireNamespace("blueant", quietly = TRUE)) {
  remotes::install_github("AustralianAntarcticDivision/blueant")
}
library(blueant)

data_dir <- "BlueAnt_AllData"
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)

# --- SST sources ---
cat("\n=== SST SOURCES ===\n")
sst <- sources_sst()
cat(sprintf("Available SST sources (%d):\n", nrow(sst)))
for (i in seq_len(nrow(sst))) cat(sprintf("  %d. %s\n", i, sst$name[i]))

sst_dir <- file.path(data_dir, "sst")
dir.create(sst_dir, showWarnings = FALSE, recursive = TRUE)

for (i in seq_len(nrow(sst))) {
  cat(sprintf("\nDownloading SST [%d/%d]: %s\n", i, nrow(sst), sst$name[i]))
  cfg <- bb_config(local_file_root = sst_dir) %>% bb_add(sst[i, ])
  tryCatch(
    bb_sync(cfg, verbose = FALSE),
    error = function(e) cat(sprintf("  FAILED: %s\n", e$message))
  )
}

# --- Sea ice sources ---
cat("\n=== SEA ICE SOURCES ===\n")
ice <- sources_seaice()
cat(sprintf("Available sea ice sources (%d):\n", nrow(ice)))
for (i in seq_len(nrow(ice))) cat(sprintf("  %d. %s\n", i, ice$name[i]))

ice_dir <- file.path(data_dir, "seaice")
dir.create(ice_dir, showWarnings = FALSE, recursive = TRUE)

for (i in seq_len(nrow(ice))) {
  cat(sprintf("\nDownloading sea ice [%d/%d]: %s\n", i, nrow(ice), ice$name[i]))
  cfg <- bb_config(local_file_root = ice_dir) %>% bb_add(ice[i, ])
  tryCatch(
    bb_sync(cfg, verbose = FALSE),
    error = function(e) cat(sprintf("  FAILED: %s\n", e$message))
  )
}

cat("\n=== DONE ===\n")
cat(sprintf("SST data: %s/\n", sst_dir))
cat(sprintf("Sea ice data: %s/\n", ice_dir))
