# -----------------------------------------------------------------------------
# 0. MEMORY CLEANUP
# -----------------------------------------------------------------------------
rm(list = ls()) 
gc()

# -----------------------------------------------------------------------------
# 1. LIBRARIES
# -----------------------------------------------------------------------------
packages <- c("foreach", "doParallel", "lpSolve", "readr", "dplyr", "aricode", "cluster")
to_install <- setdiff(packages, rownames(installed.packages()))
if (length(to_install) > 0) install.packages(to_install)

library(foreach)
library(doParallel)
library(lpSolve)
library(dplyr)
library(readr)
library(aricode)
library(cluster)

# -----------------------------------------------------------------------------
# 2. UTILITY AND PRE-PROCESSING FUNCTIONS
# -----------------------------------------------------------------------------

#Separates the ground truth label column from the feature space
remove_label_column <- function(df, label_col_name) {
  if(!label_col_name %in% names(df)) {
    warning("Label column does not exist.")
    return(list(data = df, labels = NULL))
  }
  labels <- df[[label_col_name]]
  data_only <- df %>% select(-all_of(label_col_name))
  return(list(data = data_only, labels = labels))
}

#Removes ID/empty columns, imputes missing values using the mean, and encodes categorical variables into numeric format.
preprocess_dataset <- function(df) {
  df <- as.data.frame(df)
  id_cols <- grepl("^id$", colnames(df), ignore.case = TRUE)
  if (any(id_cols)) df <- df[, !id_cols, drop = FALSE]
  
  # Remove empty columns
  df <- df[, colSums(is.na(df)) < nrow(df), drop = FALSE]
  
  # Convert Factor -> Numeric
  df <- as.data.frame(lapply(df, function(x) {
    if(is.numeric(x)) return(x)
    else return(as.numeric(as.factor(x)))
  }))
  
  # Impute NA with mean
  for (col in colnames(df)) {
    if (any(is.na(df[[col]]))) {
      col_mean <- mean(df[[col]], na.rm = TRUE)
      if(is.nan(col_mean)) col_mean <- 0 
      df[[col]][is.na(df[[col]])] <- col_mean
    }
  }
  return(as.data.frame(df))
}

#Calculates the exact theoretical number of valid cardinality partitions
count_combinations_exact <- function(n, ranges) {
  lower <- ranges$lower; upper <- ranges$upper
  k <- length(lower); S <- n - sum(lower); R <- upper - lower 
  if (S < 0) return(0)
  dp <- rep(0, S + 1); dp[1] <- 1 
  for (i in 1:k) { 
    limit <- R[i]; new_dp <- rep(0, S + 1)
    for (s in 0:S) { 
      if (dp[s+1] > 0) {
        for (v in 0:limit) {
          if (s + v <= S) new_dp[s + v + 1] <- new_dp[s + v + 1] + dp[s+1]
        }
      }
    }
    dp <- new_dp
  }
  return(dp[S + 1])
}

#Computes the allowable lower and upper bounds [L, U] for each cluster's size based on the user-defined target cardinality
compute_cardinality_ranges <- function(target, delta) {
  lower <- floor(target * (1 - delta))
  upper <- round(target * (1 + delta))
  return(list(lower = lower, upper = upper))
}

#Generates a valid and efficient random pool of candidate cardinality vectors that strictly satisfy 
#the total instance count and the computed range constraints.
generate_smart_random_pool <- function(n, ranges, n_samples) {
  lower <- ranges$lower
  upper <- ranges$upper
  k <- length(lower)
  pool <- list()
  
  seen <- new.env(hash = TRUE, parent = emptyenv())
  
  attempts <- 0
  max_attempts <- n_samples * 20 
  
  while(length(pool) < n_samples && attempts < max_attempts) {
    attempts <- attempts + 1
    current_vec <- numeric(k)
    current_sum <- 0
    valid_path <- TRUE
    
    for (i in 1:(k-1)) {
      min_needed_future <- sum(lower[(i+1):k])
      max_possible_future <- sum(upper[(i+1):k])
      
      safe_min <- max(lower[i], n - current_sum - max_possible_future)
      safe_max <- min(upper[i], n - current_sum - min_needed_future)
      
      if (safe_min > safe_max) {
        valid_path <- FALSE
        break 
      }
      
      if (safe_min == safe_max) {
        val <- safe_min
      } else {
        val <- sample(safe_min:safe_max, 1)
      }
      
      current_vec[i] <- val
      current_sum <- current_sum + val
    }
    
    if (valid_path) {
      remainder <- n - current_sum
      if (remainder >= lower[k] && remainder <= upper[k]) {
        current_vec[k] <- remainder
        
        cand_str <- paste(current_vec, collapse = "-")
        
        if (is.null(seen[[cand_str]])) {
          seen[[cand_str]] <- TRUE
          pool[[length(pool) + 1]] <- current_vec
        }
      }
    }
  }
  
  return(pool)
}

# -----------------------------------------------------------------------------
# 3. CORE CLUSTERING LOGIC
# -----------------------------------------------------------------------------

#Computes the cost matrix between data points and centroids
get_cost_matrix <- function(X_norm, centers) {
  centers_norms <- sqrt(rowSums(centers^2)) + 1e-10
  centers_norm <- centers / centers_norms
  sim <- tcrossprod(X_norm, centers_norm)
  dists <- 1 - sim
  dists[dists < 0] <- 0
  return(dists)
}

#Solves the constrained clustering assignment problem using Linear Programming to minimize global cost.
optimize_constrained_clustering <- function(X, X_norm, card_constraints, max_iter = 20, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  k <- length(card_constraints)
  n <- nrow(X)
  
  idx <- sample(1:n, k)
  centroids <- X[idx, , drop = FALSE]
  
  old_p <- integer(n); p <- integer(n); valid <- TRUE
  row_signs <- rep("=", n); row_rhs <- rep(1, n)
  col_signs <- rep("=", k); col_rhs <- card_constraints
  
  for (iter in 1:max_iter) {
    costs <- get_cost_matrix(X_norm, centroids)
    
    res <- lp.transport(costs, "min", row_signs, row_rhs, col_signs, col_rhs)
    if(res$status != 0) { valid <- FALSE; break }
    
    p <- apply(res$solution, 1, which.max)
    if (all(p == old_p)) break
    old_p <- p
    
    for (j in 1:k) {
      mask <- (p == j)
      if (any(mask)) {
        centroids[j, ] <- colMeans(X[mask, , drop = FALSE])
      } else {
        centroids[j, ] <- X[sample(1:n, 1), ]
      }
    }
  }
  return(list(p = p, centroids = centroids, valid = valid))
}

#Wrapper function that executes the optimization for a specific cardinality candidate and calculates performance metrics
evaluate_solution <- function(X, X_norm, target_card, card, true_labels = NULL, seed_val = NULL) {
  res <- optimize_constrained_clustering(X, X_norm, card, max_iter = 15, seed = seed_val)
  if (!res$valid) return(NULL)
  
  dists <- get_cost_matrix(X_norm, res$centroids)
  idx_mat <- cbind(1:nrow(X), res$p)
  a_i <- dists[idx_mat]
  
  dists[idx_mat] <- Inf
  b_i <- do.call(pmin, as.data.frame(dists))
  
  sil_mean <- mean((b_i - a_i) / pmax(a_i, b_i), na.rm = TRUE)
  
  counts <- tabulate(res$p, nbins = length(card))
  ilvc <- sum(abs(sort(counts) - sort(target_card)))
  clvc <- sum(sort(counts) != sort(target_card))
  csvi <- 0.5 * (ilvc / nrow(X)) + 0.5 * (clvc / length(card))
  
  ami_val <- NA
  if (!is.null(true_labels)) {
    tl <- trimws(as.character(true_labels))
    if (!anyNA(tl) && length(tl) == length(res$p) && length(unique(tl)) >= 2) {
      tl_int <- as.integer(factor(tl))     
      pred_int <- as.integer(res$p)    
      ami_val <- aricode::AMI(pred_int, tl_int)
    }
  }
  c(sil_mean, ilvc, clvc, csvi, ami_val)
}

# -----------------------------------------------------------------------------
# 4. PARALLEL SEARCH
# -----------------------------------------------------------------------------

#Distributes the evaluation of the cardinality pool across processor cores to efficiently explore the solution space in parallel.
run_parallel_search <- function(dataset, target_card, delta, true_labels = NULL, n_BA = 100) {
  
  X <- as.matrix(dataset)
  X_norms <- sqrt(rowSums(X^2)) + 1e-10
  X_norm <- X / X_norms
  n <- nrow(X)
  
  ranges <- compute_cardinality_ranges(target_card, delta)
  max_teorico <- count_combinations_exact(n, ranges)
  n_samples_final <- min(n_BA, max_teorico)
  set.seed(124) 
  pool <- generate_smart_random_pool(n, ranges, n_samples_final)
  n_jobs <- length(pool)
  
  cat(sprintf(" - Combinations to explore: %d\n\n", n_jobs))
  
  cores <- parallel::detectCores() - 1
  cl <- makeCluster(cores)
  registerDoParallel(cl)
  
  MASTER_SEED <- 777
  
  results <- foreach(i = 1:n_jobs, .combine = rbind, 
                     .packages = c("lpSolve", "aricode"), 
                     .export = c("optimize_constrained_clustering", "get_cost_matrix", "evaluate_solution")) %dopar% {
                       
                       card_candidate <- pool[[i]]
                       current_seed <- MASTER_SEED + (i * 7)
                       
                       metrics <- evaluate_solution(X, X_norm, target_card, card_candidate, true_labels, seed_val = current_seed)
                       
                       if (!is.null(metrics)) {
                         data.frame(
                           solution_id = i, 
                           silhouette = metrics[1], 
                           ILVC = metrics[2], 
                           CLVC = metrics[3], 
                           CSVI = metrics[4], 
                           AMI = metrics[5],
                           cardinality = paste(card_candidate, collapse = "-"),
                           saved_seed = current_seed
                         )
                       }
                     }
  
  stopCluster(cl)
  return(results)
}

# Identifies non-dominated solutions (Pareto Front) and selects the optimal trade-off solution using the Knee Point method.
analyze_pareto_exact <- function(df_input) {
  if (is.null(df_input) || nrow(df_input) == 0) return(NULL)
  
  df <- df_input
  
  # Ensure column names are standardized for internal logic
  if("silhouette" %in% names(df)) df$s <- df$silhouette
  if("CSVI" %in% names(df)) df$csvi <- df$CSVI
  
  if(!"s" %in% names(df) || !"csvi" %in% names(df)) {
    stop("Error: Dataframe must contain columns 's' (or 'silhouette') and 'csvi' (or 'CSVI')")
  }
  
  # 1. Identify Non-Dominated Solutions
  is_pareto_maxmin <- function(df) {
    n <- nrow(df)
    pareto <- rep(TRUE, n)
    
    for (i in 1:n) {
      for (j in 1:n) {
        dominates <- (df$s[j] >= df$s[i]) &&
          (df$csvi[j] <= df$csvi[i]) &&
          ((df$s[j] > df$s[i]) || (df$csvi[j] < df$csvi[i]))
        
        if (dominates) {
          pareto[i] <- FALSE
          break
        }
      }
    }
    pareto
  }
  
  df$pareto <- is_pareto_maxmin(df)
  
  pf <- df[df$pareto, ]
  pf <- pf[order(-pf$s), ]
  
  # 2. Identify Knee Point
  knee_point <- function(pf) {
    if (nrow(pf) < 3) return(pf[1, , drop = FALSE])
    
    x <- pf$s
    y <- pf$csvi
    
    x1 <- x[1];  y1 <- y[1]
    x2 <- x[nrow(pf)]; y2 <- y[nrow(pf)]
    
    denom <- sqrt((y2 - y1)^2 + (x2 - x1)^2)
    if (denom == 0) return(pf[1, , drop = FALSE])
    
    dist <- abs((y2 - y1) * x -
                  (x2 - x1) * y +
                  x2 * y1 -
                  y2 * x1) / denom
    
    # Exclude extremes
    dist[1] <- -Inf
    dist[nrow(pf)] <- -Inf
    
    pf[which.max(dist), , drop = FALSE]
  }
  
  kp <- knee_point(pf)
  
  return(list(pareto_full = pf, knee_full = kp))
}

# -----------------------------------------------------------------------------
# 5. MAIN EXECUTION
# -----------------------------------------------------------------------------

dataset_path <- "C:/Users/emont/OneDrive/Escritorio/Tesis/MILP/Datasets/Datasets/Small/Iris.csv"
dataset_name <- tools::file_path_sans_ext(basename(dataset_path))
raw_data <- read_csv(dataset_path, show_col_types = FALSE)

cat("Loading dataset", dataset_path, "...\n")

split_data <- remove_label_column(raw_data, "Species")
true_labels <- split_data$labels
input_data <- preprocess_dataset(split_data$data)

######################
#Hyperparameters     #
######################
target_cardinality <- c(50,50,50)
delta_value <- 0.1
user_max_iter <- NULL

k_groups <- length(target_cardinality)
ranges <- compute_cardinality_ranges(target_cardinality, delta_value)
max_teorico <- count_combinations_exact(nrow(input_data), ranges)

if (!is.null(user_max_iter)) {
  n_iteraciones_max <- min(user_max_iter, max_teorico)
  if (user_max_iter > max_teorico) {
    cat(sprintf("WARNING: Only %d possible solutions exist.\n", 
                user_max_iter, max_teorico))
    cat("Limit adjusted to theoretical maximum.\n")
  }
  
} else {
  n_iter_heuristic <- 50 * (k_groups^2)
  n_iteraciones_max <- min(n_iter_heuristic, max_teorico)
  n_iteraciones_max <- min(10000, n_iteraciones_max)
}

cat(sprintf(">>> Configuration:\n - Target: %s\n - Delta: %.2f\n - Possible Combinations: %.0f \n", 
            paste(target_cardinality, collapse="-"), delta_value, max_teorico))

results <- run_parallel_search(input_data, target_cardinality, delta_value, true_labels, n_BA = n_iteraciones_max)

if(!is.null(results) && nrow(results) > 0) {
  
  analisis <- analyze_pareto_exact(results)
  
  if(!is.null(analisis)) {
    df_pareto_print <- analisis$pareto_full
    df_knee_print   <- analisis$knee_full
    
    rename_cols <- function(df) {
      if(is.null(df)) return(NULL)
      col_names <- names(df)
      col_names[col_names == "s"] <- "Silhouette"
      col_names[col_names == "silhouette"] <- "Silhouette"
      col_names[col_names == "csvi"] <- "CSVI"
      names(df) <- col_names
      return(df)
    }
    
    df_pareto_print <- rename_cols(df_pareto_print)
    df_knee_print   <- rename_cols(df_knee_print)
    cols_ver <- c("solution_id", "Silhouette", "ILVC", "CLVC", "CSVI", "AMI", "cardinality")
    
    cols_finales_pareto <- intersect(cols_ver, names(df_pareto_print))
    cols_finales_knee   <- intersect(cols_ver, names(df_knee_print))
    
    cat("\n==============================================================================\n")
    cat(" PARETO FRONT\n")
    cat("==============================================================================\n")
    print(df_pareto_print[, cols_finales_pareto])
  }
  
} else {
  warning("No results to display.")
}

# -----------------------------------------------------------------------------
# 6. FINAL PHASE (RECONSTRUCTION & SAVING)
# -----------------------------------------------------------------------------

if(!is.null(results) && nrow(results) > 0 && exists("analisis") && !is.null(analisis$knee_full)) {
  
  candidate_row <- analisis$knee_full
  
  best_card_vec <- as.numeric(unlist(strsplit(candidate_row$cardinality, "-")))
  
  if(length(best_card_vec) != length(target_cardinality)) {
    stop("ERROR: Clusters in memory do not match current target. Run 'rm(list=ls())' to clear memory.")
  }
  
  X_mat <- as.matrix(input_data)
  X_norms <- sqrt(rowSums(X_mat^2)) + 1e-10
  X_norm <- X_mat / X_norms
  
  seed_key <- candidate_row$saved_seed
  final_model <- optimize_constrained_clustering(X_mat, X_norm, best_card_vec, max_iter = 100, seed = seed_key)
  
  if(final_model$valid) {
    centroids <- final_model$centroids
    cent_norms <- sqrt(rowSums(centroids^2)) + 1e-10
    centroids_norm <- centroids / cent_norms
    dists <- 1 - (X_norm %*% t(centroids_norm))
    idx_matrix <- cbind(1:nrow(X_mat), final_model$p)
    a_i <- dists[idx_matrix]
    dists[idx_matrix] <- Inf
    b_i <- do.call(pmin, as.data.frame(dists))
    
    final_sil_ahc_style <- mean((b_i - a_i) / pmax(a_i, b_i), na.rm = TRUE)
    final_ami <- NA
    if(!is.null(true_labels)) final_ami <- aricode::AMI(final_model$p, true_labels)
    
    real_counts <- tabulate(final_model$p, nbins = length(best_card_vec))
    final_ilvc <- sum(abs(sort(real_counts) - sort(target_cardinality)))
    final_clvc <- sum(sort(real_counts) != sort(target_cardinality))
    final_csvi <- 0.5 * (final_ilvc / nrow(X_mat)) + 0.5 * (final_clvc / length(target_cardinality))
    
    cat("\n===================================================\n")
    cat("            KNEE POINT SOLUTION\n")
    cat("===================================================\n")
    cat(sprintf("Solution ID                : %s\n", candidate_row$solution_id))
    cat(sprintf("Silhouette                 : %.4f\n", final_sil_ahc_style))
    cat(sprintf("AMI                        : %.4f\n", final_ami))
    cat(sprintf("ILVC                       : %.0f\n", final_ilvc))
    cat(sprintf("CLVC                       : %.0f\n", final_clvc))
    cat(sprintf("CSVI                       : %.4f\n", final_csvi))
    cat(sprintf("Cardinality                : %s\n", paste(real_counts, collapse="-")))
    cat("===================================================\n")
    
    # Organize Final Dataset
    final_dataset <- input_data
    if(!is.null(true_labels)) final_dataset$True_Labels <- true_labels
    final_dataset$Cluster <- final_model$p 
    
    file_name <- paste0("results_", dataset_name, ".csv")
    write.csv(final_dataset, file_name, row.names = FALSE)
    cat(sprintf("File saved: %s\n", file_name))
  } else {
    cat("Critical error: Could not reconstruct final model.\n")
  }
}