# Load required libraries
library(caret)
library(dplyr)
library(rpart)

# Function to calculate Gini index using the simplified formula
calculate_gini <- function(data) {
  # Calculate proportion of 'Yes' instances
  p <- mean(data == "Yes")
  
  # Calculate Gini index using the simplified formula
  gini <- 2 * p * (1 - p)
  
  return(gini)
}

# Function to perform random forest training
random_forest <- function(X, y, num_trees = 100, max_depth = 10, subsample_size = 0.8) {
  decision_trees <- list()
  
  # Ensure X is a data frame
  if (!is.data.frame(X)) {
    X <- as.data.frame(X)
  }
  
  for (i in 1:num_trees) {
    # Sample with replacement
    n_rows <- nrow(X)
    sample_indices <- sample.int(n_rows, size = round(n_rows * subsample_size), replace = TRUE)
    
    # Subsample data
    X_subsample <- X[sample_indices, ]
    y_subsample <- y[sample_indices]
    
    # Check if there are any missing values in the subsample
    if (any(is.na(X_subsample))) {
      cat("Skipping tree", i, "due to missing values in the subsample\n")
      next
    }
    
    # Train a decision tree
    # Train a decision tree with custom Gini index
    clf <- tryCatch(
      rpart::rpart(y_subsample ~ ., data = as.data.frame(cbind(X_subsample, y_subsample)),
                   control = rpart.control(splitfunc = "calculate_gini")),
      warning = function(w) {
        cat("Warning during tree training:", w$message, "\n")
        return(NULL)
      }
    )
    
    
    # Check if clf is NULL (indicating an error) and skip to the next iteration
    if (is.null(clf)) {
      cat("Skipping tree", i, "due to training error\n")
      next
    }
    
    decision_trees[[i]] <- clf
    cat("Tree", i, "trained successfully\n")
  }
  
  return(decision_trees)
}

# Function to make predictions using the random forest
predict_random_forest <- function(decision_trees, X) {
  num_trees <- length(decision_trees)
  predictions <- matrix(0, nrow = nrow(X), ncol = num_trees)
  
  for (i in 1:num_trees) {
    clf <- decision_trees[[i]]
    predictions[, i] <- predict(clf, newdata = X, type = "class")
  }
  
  # Use majority voting for the final prediction
  final_predictions <- apply(predictions, 1, function(row) {
    if (length(unique(row)) == 1) {
      return(unique(row))
    } else {
      return(names(sort(table(row), decreasing = TRUE)[1]))
    }
  })
  
  return(final_predictions)
}

# Set the seed for reproducibility
set.seed(1023)

# Load your dataset (replace with your dataset path)
weather_data <- read.csv("D:\\Data Science Data sheets\\pune2.csv", header = TRUE, sep = ",", stringsAsFactors = TRUE)

# Preprocess your data (subset, remove columns, etc.)
weather_data2 <- subset(weather_data, select = -c(Date, Location, MinTemp, MaxTemp, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Cloud9am, Cloud3pm, Temp3pm, RainToday))

# Remove rows with missing values
training_set <- weather_data2[complete.cases(weather_data2),]

# Separate predictors (X_train) and target (y_train)
X_train <- training_set[, -ncol(training_set)]
target <- training_set$RainTomorrow
y_train_dataframe <- as.factor(as.numeric(target == "Yes")) # Convert to a factor with 2 levels

# Calculate Gini index using the simplified formula
gini <- calculate_gini(target)
cat("Original Gini Index:", gini, "\n")

# Train the random forest
decision_trees <- random_forest(X_train, y_train_dataframe, num_trees = 10, max_depth = 10, subsample_size = 0.8)

# Load your test data (replace with your test dataset path)
X_test <- weather_data2[-which(complete.cases(weather_data2)), -ncol(weather_data2)]

# Ensure the structure of X_test matches X_train
X_test <- X_test[1:nrow(X_train), ]

# Make predictions
y_pred <- predict_random_forest(decision_trees, X_test)

# Load your test set (interactively choose the file)
test_set <-  read.csv("D:\\Data Science Data sheets\\pune2.csv", header = TRUE, sep = ",", stringsAsFactors = TRUE)

# Ensure the structure of the test set matches the training set
test_set <- test_set[1:nrow(X_train), ]

# Calculate the confusion matrix manually
conf_matrix <- table(test_set$RainTomorrow, y_pred)

# Calculate the confusion matrix manually
conf_matrix <- table(test_set$RainTomorrow, y_pred)

# Print the confusion matrix
print(conf_matrix)

TN <- conf_matrix[1]
FN <- conf_matrix[2]
FP <- conf_matrix[3]
TP <- conf_matrix[4]

accu_mod <- (TP + TN) / (TP + TN + FN + FP)
print(paste("Accuracy:", round(accu_mod * 100, 2), "%"))