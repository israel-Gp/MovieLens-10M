# EDX Provided Data Download Code -----------------------------------------

##########################################################
# Create edx and final_holdout_test sets
##########################################################

# Note: this process could take a couple of minutes

if (!require(tidyverse))
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(caret))
  install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if (!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip",
                dl)

ratings_file <- "ml-10M100K/ratings.dat"
if (!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if (!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <-
  as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(
    userId = as.integer(userId),
    movieId = as.integer(movieId),
    rating = as.numeric(rating),
    timestamp = as.integer(timestamp)
  )

movies <-
  as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind = "Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <-
  createDataPartition(
    y = movielens$rating,
    times = 1,
    p = 0.1,
    list = FALSE
  )
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Save Dataset as R file --------------------------------------------------

# For effective memory management data will be saved as an RDS when not in use
# In the Data directory, created if not found
# Garbage is collected afterwards

dir.create(file.path(getwd(), 'Data'), showWarnings = FALSE)
saveRDS(edx, file = file.path('Data', 'edx.rds'))
saveRDS(final_holdout_test, file = file.path('Data', 'final_holdout_test.rds'))

# Set Custom Project Themes and Functions ---------------------------------

theme_set(theme_bw())

## Define color palette for expected factors ------------------------------

scale_color_movielens_months <- function(extra = NULL) {
  scale_color_manual(
    values = c(
      '#f8766d',
      # January
      '#00ba39',
      # February
      '#649dff',
      # March
      '#3f0116',
      # April
      '#116966',
      # May
      '#ab3141',
      # June
      '#112834',
      # July
      '#7d525f',
      # August
      '#0b0862',
      # September
      '#516615',
      # October
      '#5e4db9',
      # November
      '#7d4400',
      # December,
      extra
    )
  )
}

## Define Memory Cleaning Function ----------------------------------------

clear_memory <- function(keep = NULL) {
  # Clear Global Environment
  # Keep Functions, exceptions and excluded items
  if (is.null(keep))
    remove <-
      setdiff(ls(envir = .GlobalEnv), lsf.str(envir = .GlobalEnv))
  else
    remove <-
      str_subset(negate = TRUE, setdiff(ls(envir = .GlobalEnv),
                                        lsf.str(envir = .GlobalEnv)),
                 paste0('^(', paste(keep, collapse = '|'), ')$'))
  rm(list = remove,
     envir = .GlobalEnv)
  
  # Clear RStudio Plots
  tryCatch({
    dev.off(dev.list()["RStudioGD"])
    message('All plots cleared')
  },
  error = function(e) {
    message('No plots to clear')
  })
  
  # Clear Memory
  invisible(gc(reset = TRUE, full = TRUE))
  message('Memory Cleared')
}

# Data Partitioning -------------------------------------------------------

library(janitor)

# Partition using an 80/20 split
# TO ensure that categorical features that are extracted are consistent
# users and film should be included in both the training and test set
# nested features and a custom map partition function is an option but
# partitioning followed by semi_join and anti_join is a more efficient option

# Load Data from File
edx <- readRDS('~/Data Projects/MovieLens-10M/Data/edx.rds')

# Convert Data Frame to Tibble and clean names
edx <- as_tibble(edx) %>%
  clean_names()

# Set Split Index
set.seed(2213, sample.kind = 'Rounding')
train_index <-
  createDataPartition(edx$rating,
                      times = 1,
                      p = 0.8,
                      list = FALSE) %>%
  as.vector()

# Split Data
edx_train <- slice(edx, train_index)
edx_test_alpha <- slice(edx, -train_index)

# Balance users and Films in Test Set
edx_test <- edx_test_alpha %>%
  semi_join(edx_train, by = 'user_id') %>%
  semi_join(edx_train, by = 'movie_id')

# ID observations removed from the test set
edx_removed <- anti_join(edx_test_alpha, edx_test)

# Attach removed observations to the Training Set
edx_train <- bind_rows(edx_train, edx_removed)

# Save Train and Test Set to File for storage
saveRDS(edx_train, file = file.path('Data', 'edx_train.rds'))
saveRDS(edx_test, file = file.path('Data', 'edx_test.rds'))

# Clear Memory
# Keep Training Set
clear_memory(keep = 'edx_train')

# Model Considerations ----------------------------------------------------

# Recommendation systems are a type of information filtering system,
# a system that remopves redundant iformatio, in this case recommendations
# are performed by predictive film rating
# Recommendation systems come in three basic filtering categories
# Content
# Collaborative
#  & Contextual
# with hybrid categories a 4th possible

# Content utilizes item (in this case film) attributes to recommend similar items
# Collaborative utilizes user (film reviewers) similarity for recommendations
# Contextual filtering utilizes user-item-enviorment context for recommendations

# Content based filtering can utilize film features for prediction
# Collaborative filtering can utilize a matrix factorization for predictions
# Contextual based filtering can utilize review enviorment data 

