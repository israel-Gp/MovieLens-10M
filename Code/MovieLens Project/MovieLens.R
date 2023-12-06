# Libraries --------------------------------------------------------------

## Required Libraries ------------------------------------------------------

packages.to.install <- c(
  # General Purpose
  
  'tidyverse',
  # Tidyverse
  'janitor',
  # Data Cleaning
  'caret',
  # for machine learning
  'furrr',
  # Parallel mapping
  'broom',
  # For tidying base and tidyverse operations
  'lubridate',
  # For date-time variable manipulation
  'ggcorrplot',
  # For ggplot2 correlation plots
  'ggrepel',
  # GGplot2 repel texts
  'doParallel',
  # Parrallel Processing with CARET
  'ggridges', # Plot ridges
  'klaR', # Categorical CLustering
  'ggh4x', # GGplot ais
  'patchwork', #GGplot aid
  
  # Feature Selection
  
  'Boruta',
  # Variable Importance wrappe
  'xgboost',
  # Boruta xgboost parsing in Boruta
  'varrank',
  #Variable rank based on mutual information
  'infotheo', # Entropy & Mutual Information
  
  # R Markdown
  'knitr', # Rmarkdown aid
  'booktabs', #Rmarkdown aid
  'tinytex', # Latex aid
  'MikTeX', # Latex aid
  'kableExtra',# Rmarkdown tables
  'devtools' # For pathced libraries
)

## Download Missing Libraries ----------------------------------------------

# Parse missing packages
missing.packages <-
  packages.to.install[!(packages.to.install %in% installed.packages()[, "Package"])]

if (length(missing.packages))
  install.packages(missing.packages, dependencies = TRUE, repos = 'http://cran.us.r-project.org')

#Install Patched kableExtra 
devtools::install_github("kupietz/kableExtra")

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
edx <- movielens[-test_index, ]
temp <- movielens[test_index, ]

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

library(scales)
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
# a system that removes redundant information, in this case recommendations
# are performed by predictive film rating
# Recommendation systems come in three basic filtering categories
# Content
# Collaborative
#  & Contextual
# with hybrid categories a 4th possible

# Content utilizes item (in this case film) attributes to recommend similar items
# Collaborative utilizes user (film reviewers) similarity for recommendations
# Contextual filtering utilizes user-item-environment context for recommendations

# Content based filtering can utilize film features for prediction
# Collaborative filtering can utilize a matrix factorization for predictions
# Contextual based filtering can utilize review environment data

# These models lack independent film input, films of similar context or environment
<<<<<<< HEAD
# will differ in popularity (e.g. The Dark Knight vs Batman & Robin)
=======
# will differ in popularity (eg The Dark Knight vs Batman & Robin)
>>>>>>> 2f642d08513c447d1894bd7b728cfe9bea190c62
# A film based model is required to account for this missing input

# A ensemble model can be utilized to merge the different model
# for simplicity a linear regression stacked model will be used

<<<<<<< HEAD
# There's no accounting for taste?
# In order to explore that taste can be accounted for user based predictions will include
# a running count of the amount of reviews a user has made, this is to see whether a
# user has developed a pallet and therefore becomes more strict or looser in their
# ratings

# In contrast a running count of film reviews will be added as a measure of viewing popularity
# this can be used for both film and user individualized predictions given that, at extremes,
# some people are trend followers and others may actively avoid it.
# For appropriate use of counts these will be lagged measures as to account for data leakage

=======
>>>>>>> 2f642d08513c447d1894bd7b728cfe9bea190c62
glimpse(edx_train)

# this requires removal of film ID and title
# year of release to be separated from title
# title words may be used to cluster films using K-Modes utilizing title words
# as categories
# genres are to be separated and used as individual contexts
# as with film words these can be clustered using K-Modes
# Users can develop a pallet as time progresses, therefore timestamps in
# their raw form may be required if user reviews by time are a relevant feature
# this will require sorting by timestamp

# Data Cleaning -----------------------------------------------------------

# This data cleaning should be performed for all model features

# Determine empty data
edx_train %>%
  summarise(across(everything(), \(x) sum(is.na(x))))

# No data imputation or removal required

glimpse(edx_train)

# Date-time data to be prepared and previously mentioned contexts to be extracted
# and individual features engineered

edx_train <- edx_train %>%
  # Clean timestamp
  # Relocate response variable as preferred
  relocate(rating) %>%
  # Clean timestamp
  mutate(across(timestamp, as_datetime)) %>%
  # sort by time
  arrange(timestamp) %>%
  # Separate Title Features
  separate_wider_regex(
    title,
    patterns = c(
      title = '[:print:]+(?=(?:[:space:]\\([:digit:]{4}\\)))',
      ' \\(',
      film_year_of_release = '.*',
      '\\)'
    )
  ) %>%
  # Separate genres
  separate_wider_delim(genres,
                       delim = '|',
                       names_sep = '_',
                       too_few = 'align_start') %>%
  # Rename genres to singular
<<<<<<< HEAD
  rename_with(~ str_replace(.x, '^genres', 'genre')) %>%
=======
  rename_with( ~ str_replace(.x, '^genres', 'genre')) %>%
>>>>>>> 2f642d08513c447d1894bd7b728cfe9bea190c62
  # Replace NA genres with None
  mutate(across(starts_with('genre'), \(x) replace_na(x, 'None'))) %>%
  # Engineer date-time features
  mutate(
    film_year_of_release = as.numeric(str_extract(film_year_of_release, '[:digit:]{4}')),
    # Convert timestamp to date-time
    timestamp = as_datetime(timestamp),
    # Extract date features
    year = year(timestamp),
    month = month(timestamp, label = TRUE, abbr = FALSE),
    day = day(timestamp),
    day_of_the_year = yday(timestamp),
    day_of_the_quarter = qday(timestamp),
    # Day of the Week, weeks starts on Monday
    weekday = wday(
      timestamp,
      label = TRUE,
      abbr = FALSE,
      week_start = 1
    ),
    # Extract Time Features
    hour = hour(timestamp),
    minute = minute(timestamp),
    second = second(timestamp),
    # Calculate Film Age
    film_age = year - film_year_of_release,
<<<<<<< HEAD
    film_year_of_release = as_factor(as.character(film_year_of_release))
  ) %>%
  # Relocate features as preferred
  relocate(film_year_of_release, .before = timestamp) %>%
  relocate(film_age, .after = film_year_of_release) %>%
  # Genres as Factors
  mutate(across(starts_with('genre'), as.factor)) %>%
  # IDs as Factors
  mutate(across(ends_with('_id'), as.factor)) %>%
  group_by(user_id) %>%
  mutate(user_reviews = lag(row_number(), default = 0)) %>%
  group_by(movie_id) %>%
  mutate(movie_reviews = lag(row_number(), default = 0)) %>%
  ungroup()
=======
  ) %>%
  # Relocate features as preffered
  relocate(film_year_of_release, .before = timestamp) %>%
  relocate(film_age, .after = film_year_of_release) %>%
  # Genres as Factors
  mutate(across(starts_with('genre'), as.factor))
>>>>>>> 2f642d08513c447d1894bd7b728cfe9bea190c62

# Save Train and Test Set to File for storage
saveRDS(edx_train, file = file.path('Data', 'edx_train.rds'))

# Clear Memory
# Keep Training Set
clear_memory(keep = 'edx_train')

<<<<<<< HEAD
# Data Exploration & Feature Engineering ----------------------------------

library(broom)

# Define an error proof t-test function for numeric summaries
# standard t-test function returns errors for constant values

t_test <-
  function(x,
           y = NULL,
           alternative = c("two.sided", "less", "greater"),
           mu = 0,
           paired = FALSE,
           var.equal = FALSE,
           conf.level = 0.95,
           ...) {
    t.test_result <- tryCatch(
      # T-test
      {
        tidy(
          t.test(
            x = x,
            y = y,
            alternative = alternative,
            mu = mu,
            paired = paired,
            var.equal = var.equal,
            conf.level = conf.level,
            ...
          )
        )
      },
      # If Error
      error = function(e) {
        tibble(
          estimate = unique(x),
          statistic = NA,
          p.value = NA,
          parameter = length(x),
          conf.low = NA,
          conf.high = NA,
          method = 'Constant Value',
          alternative = 'Constant Value'
        )
      }
    )
    
    # Return Test Result
    return(t.test_result)
    
  }

edx_train_stats <- edx_train %>%
  pivot_longer(cols = where(is.numeric),
               names_to = 'variable') %>%
  group_by(variable) %>%
  summarise(t.test = t_test(value)) %>%
  unnest(cols = starts_with('t.test'))

## Rating Distribution ----------------------------------------------------

edx_train %>%
  ggplot(aes(rating)) +
  geom_density(fill = '#FF0000', alpha = 0.5) +
  geom_vline(
    data = filter(edx_train_stats, variable == 'rating'),
    aes(xintercept = estimate),
    color = '#000000',
    linetype = 'dashed'
  ) +
  geom_label(
    data = filter(edx_train_stats, variable == 'rating'),
    aes(
      x = estimate,
      y = -0.15,
      label = paste0('mu==', as.character(round(estimate, 2)))
    ),
    parse = TRUE
  ) +
  scale_x_continuous(
    'Rating',
    labels = unique(edx_train$rating),
    breaks = unique(edx_train$rating)
  ) +
  guides(color = 'none') +
  ylab('Rating Density') +
  ggtitle('Film Rating Distribution')

# Mean Rating is 3.51
# The plot suggests that the response variable may actually be categorical
# For simplicity of the model with a data set this large a numeric response
# is used, this also aligns with the required loss function being RMSE
# using a categorical prediction model would approximate to a ± 0.5 Rating
# the response variable rating is therefore kept as numeric

## Rating Distribution by Date --------------------------------------------

edx_train_daily_mean_ratings <- edx_train %>%
  mutate(
    year = year(timestamp),
    month = month(timestamp, label = TRUE, abbr = FALSE),
    date = date(timestamp)
  ) %>%
  group_by(date) %>%
  summarise(
    t.test = t_test(rating),
    year = first(year),
    month = first(month)
  ) %>%
  unnest(cols = starts_with('t.test'))

edx_train_daily_mean_ratings %>%
  ggplot(aes(date, estimate)) +
  geom_line(aes(group = 1, color = month)) +
  geom_smooth(
    se = FALSE,
    span = 365.256366 / 12,
    color = '#FF0000',
    linewidth = 1,
    aes(group = interaction(year, month),
        linetype = 'Local Mean')
  ) +
  geom_hline(
    show.legend = FALSE,
    color = '#000000',
    aes(
      yintercept = filter(edx_train_stats, variable == 'rating')$estimate,
      linetype = 'Rating Mean'
    )
  ) +
  scale_color_movielens_months() +
  scale_x_continuous(
    'Date',
    labels = seq(
      min(edx_train_daily_mean_ratings$year),
      max(edx_train_daily_mean_ratings$year)
    ),
    breaks = seq.Date(
      floor_date(min(edx_train_daily_mean_ratings$date), unit = 'month'),
      ceiling_date(max(edx_train_daily_mean_ratings$date), unit = 'month'),
      by = 'year'
    )
  ) +
  scale_linetype_manual('Statistics',
                        values = c('dashed', 'solid')) +
  guides(color = guide_legend('Month'),
         linetype = guide_legend(override.aes = list(
           color = c('#FF0000', '#000000'),
           values = c('dashed', 'solid')
         ))) +
  ylab('Mean Rating') +
  ggtitle('Mean Ratings though time', subtitle = 'MovieLens daily mean ratings with Local Daily Mean and overall Rating means') +
  ylim(
    min(edx_train_daily_mean_ratings$estimate),
    max(edx_train_daily_mean_ratings$estimate)
  )

# Year and Month have an influence on local means compared to the rating mean
# Day might have an influence but is expected to be smaller
# There is high volatility on daily means before the latter half of 1999

## Rating Distribution by Year --------------------------------------------

library(ggridges)

edx_train %>%
  mutate(year = year(timestamp)) %>%
  ggplot(aes(rating, as.factor(year))) +
  stat_density_ridges(
    geom = 'density_ridges',
    calc_ecdf = TRUE,
    quantiles = 2,
    quantile_lines = TRUE,
    fill = '#FF0000',
    alpha = 0.5
  ) +
  geom_vline(
    data = filter(edx_train_stats, variable == 'rating'),
    aes(xintercept = estimate),
    color = '#000000',
    linetype = 'dashed'
  ) +
  geom_label(
    data = filter(edx_train_stats, variable == 'rating'),
    aes(
      x = estimate,
      y = 1,
      label = paste0('mu==', as.character(round(estimate, 2)))
    ),
    parse = TRUE
  ) +
  scale_x_continuous(
    'Rating',
    labels = unique(edx_train$rating),
    breaks = unique(edx_train$rating)
  ) +
  ylab('Year') +
  ggtitle('Film Rating Distribution by Review Year')

# Half star ratings are zero or near zero for years 1995 to 2002
# after which densities for 3.5 increases, peaking at between 3 and 4's densities
# Year is expected to have a major impact on rating prediction
# Time features will be kept as is and filtered out in the feature selection process
# Genre features will be processed to reduce the feature space

## Predictor Distributions ------------------------------------------------

### ID Factor Distributions -----------------------------------------------

edx_train_id_means <- edx_train %>%
  pivot_longer(cols = ends_with('_id'),
               names_to = 'id_type',
               values_to = 'id') %>%
  group_by(id_type, id) %>%
  summarise(t.test = t_test(rating)) %>%
  unnest(cols = starts_with('t.test')) %>%
  ungroup()

edx_train_id_means_summary <- edx_train_id_means %>%
  group_by(id_type) %>%
  summarise(t.test = t_test(estimate)) %>%
  unnest(cols = starts_with('t.test'))

edx_train_id_means %>%
  ggplot(aes(estimate, fill = id_type)) +
  geom_density(alpha = 0.25) +
  geom_vline(
    data = filter(edx_train_stats, variable == 'rating'),
    aes(xintercept = estimate),
    color = '#000000',
    linetype = 'dashed'
  ) +
  geom_label(
    inherit.aes = FALSE,
    data = filter(edx_train_stats, variable == 'rating'),
    aes(
      x = estimate,
      y = -0.15,
      label = paste0('mu==', as.character(round(estimate, 2)))
    ),
    parse = TRUE
  ) +
  geom_vline(
    data = filter(edx_train_id_means_summary, id_type == 'movie_id'),
    aes(xintercept = estimate),
    color = '#000000',
    linetype = 'dashed'
  ) +
  geom_label(
    inherit.aes = FALSE,
    data = filter(edx_train_id_means_summary, id_type == 'movie_id'),
    aes(
      x = estimate,
      y = -0.05,
      label = paste0('mu[movie]==', as.character(round(estimate, 2)))
    ),
    parse = TRUE
  ) +
  geom_vline(
    data = filter(edx_train_id_means_summary, id_type == 'user_id'),
    aes(xintercept = estimate),
    color = '#000000',
    linetype = 'dashed'
  ) +
  geom_label(
    inherit.aes = FALSE,
    data = filter(edx_train_id_means_summary, id_type == 'user_id'),
    aes(
      x = estimate,
      y = -0.05,
      label = paste0('mu[user]==', as.character(round(estimate, 2)))
    ),
    parse = TRUE
  ) +
  guides(fill = guide_legend('ID Type')) +
  xlab('Mean Rating') +
  ylab('Rating Density') +
  ggtitle('Mean Rating Distribution by ID Factors')

# On average Films have a slightly lower mean rating than Users ratings
# User mean ratings are approximately normally distributed
# For films there is a pronounced negative skew
# Both are noticeably different to the overall mean

tidy(t.test(
  filter(edx_train_id_means, id_type == 'movie_id')$estimate,
  filter(edx_train_id_means, id_type == 'user_id')$estimate
))

# Both mean distributions are statistically different

# tidy(stats::chisq.test(edx_train$user_id, edx_train$movie_id))

# Both Variables are independent of each other

# Users on average rate higher than the overall mean while films are rated lower
# than the overall mean

edx_train_id_means %>%
  count(id_type) %>%
  mutate(prop = label_percent()(n / sum(n))) %>%
  arrange(desc(n))

# There are more users than films in the data
# They are independent of each other, interaction potential for regression low
# A significant amount of films are rated far lower than the mean rating

edx_train %>%
  count(user_id, movie_id) %>%
  ggplot(aes(user_id, movie_id, fill = n)) +
  geom_tile(show.legend = FALSE) +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank()
  ) +
  ggtitle('ID Feature Interactions')

# Data is sparse, interaction for predictors will not be applied for prediction
# Alternative methods must be applied to predict an interaction.

## Genre Distribution -----------------------------------------------------

library(ggrepel)

edx_genre_wide <- edx_train %>%
  distinct(movie_id, .keep_all = TRUE) %>%
  select(c(movie_id, starts_with('genre')))

edx_genre_tidy <- edx_genre_wide %>%
  pivot_longer(
    cols = starts_with('genre_'),
    names_to = 'genre_level',
    values_to = 'genre',
    names_transform = list(genre_level = ~ as_factor(str_remove(.x, 'genre_')))
  )

edx_genre_tidy %>%
  count(genre_level, genre) %>%
  group_by(genre_level) %>%
  mutate(prop = label_percent(accuracy = 1)(n / sum(n))) %>%
  ungroup() %>%
  ggplot(aes(genre_level, n, fill = genre, label = prop)) +
  geom_col(color = '#000000') +
  geom_label_repel(position = position_stack(vjust = 0.5),
                   show.legend = FALSE) +
  scale_y_continuous('Count', labels = comma) +
  xlab('Genre Level') +
  guides(fill = guide_legend('Genre')) +
  ggtitle('MovieLens Genre Distribution by Level')

# Levels 3 to 4 have too high a proportion of Genre "None"
# Will use a cutoff of 80/20 to the next proportions

edx_genre_nzv <-
  nearZeroVar(
    select(edx_genre_wide, contains('genre')),
    freqCut = 80 / 20,
    uniqueCut = 10,
    saveMetrics = TRUE,
    names = TRUE,
    allowParallel = TRUE
  )

genre_remove <-
  names(select(edx_genre_wide, contains('genre')))[which(edx_genre_nzv$nzv)]

edx_train <- edx_train %>%
  select(-all_of(genre_remove))

edx_genre_tidy_2 <- edx_train %>%
  distinct(movie_id, .keep_all = TRUE) %>%
  select(c(movie_id, starts_with('genre'))) %>%
  pivot_longer(
    cols = starts_with('genre_'),
    names_to = 'genre_level',
    values_to = 'genre',
    names_transform = list(genre_level = ~ as_factor(str_remove(.x, 'genre_')))
  )

edx_genre_tidy_2 %>%
  count(genre_level, genre) %>%
  group_by(genre_level) %>%
  mutate(prop = label_percent(accuracy = 1)(n / sum(n))) %>%
  ungroup() %>%
  ggplot(aes(genre_level, n, fill = genre, label = prop)) +
  geom_col(color = '#000000') +
  geom_label_repel(position = position_stack(vjust = 0.5),
                   show.legend = FALSE) +
  scale_y_continuous('Count', labels = comma) +
  xlab('Genre Level') +
  guides(fill = guide_legend('Genre')) +
  ggtitle('MovieLens Genre Distribution by Level')

edx_genre_means <- edx_train %>%
  select(c('rating', starts_with('genre'))) %>%
  group_by(genre_1, genre_2) %>%
  reframe(
    t.test = t_test(rating),
    sd = sd(rating),
    quantile_value = quantile(rating),
    quantile = c('min', 'first', 'median', 'third', 'max')
  ) %>%
  pivot_wider(names_from = quantile,
              values_from = quantile_value) %>%
  unnest(cols = contains('t.test')) %>%
  mutate(iqr = third - first) %>%
  ungroup()

### Genre Rating Distribution ---------------------------------------------

edx_genre_wide_2 <- edx_train %>%
  distinct(movie_id, .keep_all = TRUE) %>%
  select(c(movie_id, starts_with('genre')))

edx_genre_means <- edx_train %>%
  group_by(genre_1, genre_2) %>%
  reframe(
    t.test = t_test(rating),
    sd = sd(rating),
    quantile_value = quantile(rating),
    quantile = c('min', 'first', 'median', 'third', 'max')
  ) %>%
  pivot_wider(names_from = quantile,
              values_from = quantile_value) %>%
  unnest(cols = contains('t.test')) %>%
  mutate(iqr = third - first) %>%
  ungroup()

edx_genre_means %>%
  mutate(
    genres_1 = fct_reorder(genre_1, estimate, .desc = FALSE),
    genres_2 = fct_reorder(genre_2, estimate, .desc = FALSE),
    label = str_c(round(conf.low, 2), round(conf.high, 2), sep = ' - ')
  ) %>%
  rowwise() %>%
  mutate(label = replace_na(label, as.character(round(estimate, 2)))) %>%
  ungroup() %>%
  ggplot(aes(
    genres_1,
    genres_2,
    fill = rescale(estimate, to = c(-1, 1)),
    label = label
  )) +
  geom_tile(color = '#000000') +
  geom_text() +
  scale_fill_gradient2(
    'Scaled\nMean\nRating',
    mid = '#FFFFFF',
    low = '#5BCEFA',
    high = '#F5A9B8'
  ) +
  xlab('Genre 1') +
  ylab('Genre 2') +
  ggtitle('Genre Mean Ratings by Genre Level')

# Genre Level seems to play some role in the rating of a film
# Film-Noir as an example is very highly rated but it is less so when
# secondary to comedy
# When None is the secondary Genre we can observe the mean of each genre
# in isolation, there are a number of instances where the singular genre is
# differs from the combined genre means
# Film-Noir alone averages at 3.8, above the mean rating but 2 all combinations
# which include Film-Noir are higher than the lone mean
# T-tests show that some means have a large CI
# Genre rating dist by level should be observed to make a choice on genre variables

edx_genre_wide_num <- edx_genre_wide_2 %>%
  select(contains('genre_')) %>%
  mutate(across(everything(), as.numeric))

genre_clusters_min <- edx_genre_wide_2 %>%
  pivot_longer(
    cols = starts_with('genre_'),
    names_to = 'genre',
    names_transform = list(word = as_factor)
  ) %>%
  distinct(value) %>%
  nrow()

genre_clusters_max <- edx_genre_wide_num %>%
  distinct() %>%
  nrow()

genre_clusters_max
# 144 Maximum clusters, while complex it is a manageable analysis

#### Parallel Cluster Analysis --------------------------------------------

library(furrr)
library(doParallel)
=======
# Feature Extraction ------------------------------------------------------

### Title Clusters --------------------------------------------------------

library(tidytext)

# Individual Films and title words in tidy format
# Keep cases
title_words <- edx_train %>%
  distinct(movie_id, .keep_all = TRUE) %>%
  unnest_tokens('title_words', 'title',
                token = 'words',
                to_lower = FALSE) %>%
  select(all_of(c('movie_id', 'title_words')))

title_words_wide <- title_words %>%
  group_by(movie_id) %>% 
  mutate(word_n = paste('word',row_number(),sep = '_')) %>% 
  ungroup() %>% 
  pivot_wider(names_from = word_n, values_from = title_words,
              values_fill = 'None'
  ) %>% 
  mutate(across(starts_with('word_'),as.factor))
>>>>>>> 2f642d08513c447d1894bd7b728cfe9bea190c62

# Custom K-Modes Function
kmodes_fn <-
  function(data,
           modes,
           seed,
           iter.max = 10,
           weighted = FALSE,
           fast = TRUE) {
    set.seed(seed, sample.kind = 'Rounding')
    model <-
      klaR::kmodes(
        data = data,
        modes = modes,
        iter.max = iter.max,
        weighted = weighted,
        fast = fast
      )
    return(sum(model[4]$withindiff))
  }

<<<<<<< HEAD
# Plan Parallel
parallel_cores <- detectCores() - 1

plan(multisession, workers = parallel_cores)

kmodes_data <-
  tibble(clusters = genre_clusters_min:genre_clusters_max,
         data = list(edx_genre_wide_num)) %>%
  mutate(k_modes = future_map2(
    data,
    clusters,
    seed = 2131,
    weighted = TRUE,
    # Using Weighted distances to account for differences in genre frequencies
    kmodes_fn,
    .progress = TRUE
  ))

# Stop Parallel
plan(sequential)

kmodes_data %>%
  select(-all_of('data')) %>%
  unnest(cols = 'k_modes') %>%
  ggplot(aes(x = clusters, y = k_modes)) +
  geom_point() +
  geom_line() +
  geom_smooth(linetype = 'dashed',
              se = FALSE,
              method = 'glm') +
  geom_text_repel(aes(label = clusters)) +
  xlab('Clusters') +
  ylab('Within-Cluster Simple-Matching Distance') +
  ggtitle('MovieLens Genre Optimal Clusters',
          subtitle = 'Genre Level Clustering based on the K-Modes Algorithm')

# The optimal amount of clusters is 36

genre_clusters <-
  klaR::kmodes(
    edx_genre_wide_num,
    modes = 36,
    weighted = TRUE,
    fast = TRUE
  )

edx_genre_clusters <- edx_genre_wide %>%
  mutate(genre_cluster = as_factor(genre_clusters$cluster)) %>%
  select(all_of(c('movie_id', 'genre_cluster')))

edx_train <- edx_train %>%
  left_join(edx_genre_clusters) %>%
  relocate(genre_cluster, .before = 'genre_1') %>%
  select(-matches('genre_[[:digit:]]+'))

### User Interactions -----------------------------------------------------

edx_train %>%
  count(user_id, genre_cluster) %>%
  group_by(user_id) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup() %>%
  ggplot(aes(user_id, genre_cluster, fill = prop)) +
  geom_tile() +
  scale_fill_gradient('Scaled\nReview\nProportion',
                      # mid = '#FFFFFF',
                      low = '#5BCEFA',
                      high = '#F5A9B8') +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank())

# Users and Genre Cluster may be strongly tied to each other

tidy(chisq.test(edx_train$user_id, edx_train$genre_cluster))

# Genre Cluster is not independent of User
# This is far more likely to be result of lack of variety on the user part
# as far as predictive value their interaction may improve upon user trends
# alone as genre cluster will be tied to the type of film that will be reviewed

# Save Train and Genre Clusters
saveRDS(edx_train, file = file.path('Data', 'edx_train.rds'))
saveRDS(edx_genre_clusters, file = file.path('Data', 'edx_genre_clusters.rds'))

# Clear Memory
# Keep Training Set
clear_memory(keep = 'edx_train')

## Numeric Predictor Distributions ----------------------------------------

library(ggh4x)
library(patchwork)

edx_train_num_tidy <- edx_train %>%
  mutate(across(where(is.ordered), as.numeric)) %>%
  select(where(is.numeric)) %>%
  select(-all_of('rating')) %>%
  pivot_longer(cols = everything(),
               names_to = 'variable')

edx_train_num_tidy %>%
  ggplot(aes(value)) +
  geom_histogram(
    show.legend = FALSE,
    color = '#000000',
    aes(value, after_stat(ndensity), fill = variable)
  ) +
  geom_density(show.legend = FALSE,
               color = '#FF0000',
               aes(value, after_stat(ndensity))) +
  facet_wrap2( ~ variable, scales = 'free', axes = 'all') +
  xlab('Value') +
  ylab('Density') +
  ggtitle('MovieLens Variable Distributions')

# Minute, Second and Day of the Quarter are Nearly uniformly distributed
# By itself the lack a variance may affect linear regression models
# It also may not provide PCA methods with adequate variance for best performance
# These 3 features will be removed from consideration at this time

# Day is a close 4th to this distribution, however the large spike
# and somewhat sinusoidal distribution may allow a good use for PCA

# Movie and User reviews have an approx Negative Binomial distribution
# overall
# However users by user and film by film the count should be linear as
# reviews are expected top increase as time passes and never decrease
# considering the distribution should these features be required in the final
# model they'll need to be trained as a linear model
# While the final model will require an analytical solution these may
# be trained using a chunk and pull method (nested models)
# as using a overall mean for IDs may prove inadequate for this distribution of values

edx_train <- edx_train %>%
  select(-all_of(c('day_of_the_quarter', 'minute', 'second')))

# Year, Month and weekday may yield better results as factors rather than
# as numeric features despite being

set.seed(1314, sample.kind = 'Rounding')
year_qq <-
  tibble(year = as.numeric(sample(edx_train$year, 100000))) %>%
  ggplot(aes(sample = year)) +
  stat_qq() +
  stat_qq_line() +
  xlab('Theoretical Quantiles') +
  ylab('Sample Quantiles') +
  ggtitle('Year as Numerical Feature QQ Plot')

set.seed(1327, sample.kind = 'Rounding')
month_qq <-
  tibble(month = as.numeric(sample(edx_train$month, 100000))) %>%
  ggplot(aes(sample = month)) +
  stat_qq() +
  stat_qq_line() +
  xlab('Theoretical Quantiles') +
  ylab('Sample Quantiles') +
  ggtitle('Month as Numerical Feature QQ Plot')

set.seed(1353, sample.kind = 'Rounding')
weekday_qq <-
  tibble(weekday = as.numeric(sample(edx_train$weekday, 100000))) %>%
  ggplot(aes(sample = weekday)) +
  stat_qq() +
  stat_qq_line() +
  xlab('Theoretical Quantiles') +
  ylab('Sample Quantiles') +
  ggtitle('Weekday as Numerical Feature QQ Plot')

set.seed(1353, sample.kind = 'Rounding')
fyor_qq <-
  tibble(weekday = as.numeric(sample(edx_train$film_year_of_release, 100000))) %>%
  ggplot(aes(sample = weekday)) +
  stat_qq() +
  stat_qq_line() +
  xlab('Theoretical Quantiles') +
  ylab('Sample Quantiles') +
  ggtitle('Film Year of Release as Numerical Feature QQ Plot')


(year_qq + month_qq) / (weekday_qq + fyor_qq)

# The QQ plots for these features demonstrate a tendency for the variables to be
# categorical
# Film Year of Release seems like the outlier as they are approximately numeric in nature
# The feature is not normally distributed and may require scaling and centering prior to use
# If treated as numerical we lose the capacity to be able to gauge mean regressions
# for each variable
# If treated as categorical PCA will have less numerical features to use for capturing variance
# For the following analyses these features will be treated as categorical as PCA may not be required
# at this point except for correlation analysis and detecting linear combinations

edx_train <- edx_train %>%
  mutate(across(year, as.ordered)) %>%
  mutate(across(film_year_of_release, \(x) as.numeric(as.character(x))))

# Film Age, Year of Release and both review count features will require transformations
# In order to improve non-tree based predictions.

# Film Age presents an odd behavior, the density distribution line
# seemingly extends beyond zero

edx_train %>%
  count(film_age)

# A small amount of films display negative film ages
# This could be an error or a type of early screening

edx_train %>%
  filter(film_age < 0) %>%
  count(user_id, sort = TRUE)

# The amount of users and lower than zero film age reviews may suggest
# early reviews but film id exploration is required

edx_train %>%
  filter(film_age < 0) %>%
  count(movie_id, title, sort = TRUE)

# Movie ID suggests that this is limited to a few titles, the discrepancy
# may be due to release discrepancies instead of errors

film_negative_age <- edx_train %>%
  filter(film_age < 0) %>%
  count(movie_id, title) %>%
  pull(movie_id)

edx_train %>%
  filter(movie_id %in% film_negative_age) %>%
  ggplot(aes(year, movie_id, fill = film_age, label = label_comma()(film_age))) +
  geom_tile() +
  geom_text() +
  scale_fill_gradient2('Film\nAge',
                       low = '#5BCEFA',
                       high = '#F5A9B8') +
  ggtitle('Negative Film Age Distribution')

# When using film to detect potential errors there are no indication of input errors
# At most a film has a two year discrepancy with year of release
# Negative film ages suggest that a user is an early reviewer or film critic
# an additional feature could be added but it will not be distinguishable from
# user_id unless paired with film_id being reviewed by a regular user vs an early
# reviewer (critic?)

### Critic Rating Discrepancy ---------------------------------------------

critics <- edx_train %>%
  filter(film_age < 0) %>%
  pull(user_id) %>%
  unique()

edx_train %>%
  mutate(user_type = ifelse(user_id %in% critics, 'Critic', 'Regular') %>% as_factor()) %>%
  count(year, user_type) %>%
  group_by(year) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup() %>%
  ggplot(aes(
    year,
    prop,
    label = label_percent(accuracy = 0.001)(prop),
    color = user_type,
    group = user_type
  )) +
  geom_point() +
  geom_line(linetype = 'dashed', show.legend = FALSE) +
  geom_text_repel(show.legend = FALSE) +
  scale_y_continuous('Proportion', labels = percent) +
  ggtitle('Yearly User Type Review Proportions')

# There are very few critic accounts but the proportion is stable across years

user_type_means <- edx_train %>%
  mutate(user_type = ifelse(user_id %in% critics, 'Critic', 'Regular') %>% as_factor()) %>%
  group_by(user_type, user_id) %>%
  summarise(t.test = t_test(rating)) %>%
  unnest(cols = starts_with('t.test')) %>%
  ungroup()

user_type_means_summary <- user_type_means %>%
  group_by(user_type) %>%
  summarise(t.test = t_test(estimate)) %>%
  unnest(cols = starts_with('t.test'))

user_type_means %>%
  ggplot(aes(estimate, fill = user_type)) +
  geom_density(alpha = 0.25) +
  geom_vline(
    data = filter(user_type_means_summary, user_type == 'Regular'),
    aes(xintercept = estimate),
    color = '#000000',
    linetype = 'dashed'
  ) +
  geom_label(
    inherit.aes = FALSE,
    data = filter(user_type_means_summary, user_type == 'Regular'),
    aes(
      x = estimate,
      y = -0.05,
      label = paste0('mu[Regular]==', as.character(round(estimate, 2)))
    ),
    parse = TRUE
  ) +
  geom_vline(
    data = filter(user_type_means_summary, user_type == 'Critic'),
    aes(xintercept = estimate),
    color = '#000000',
    linetype = 'dashed'
  ) +
  geom_label(
    inherit.aes = FALSE,
    data = filter(user_type_means_summary, user_type == 'Critic'),
    aes(
      x = estimate,
      y = 0,
      label = paste0('mu[Critic]==', as.character(round(estimate, 2)))
    ),
    parse = TRUE
  ) +
  guides(fill = guide_legend('User Type')) +
  xlab('Mean Rating') +
  ylab('Rating Density') +
  ggtitle('Mean Rating Distribution by User Type')

t_test(
  filter(user_type_means, user_type == 'Critic')$estimate,
  filter(user_type_means, user_type == 'Regular')$estimate
)

# There is a statistical difference in means between user_types
# critics have a thicker distribution tails than regular users
# To solve the issue of user_type being tied to user_id the new feature can be applied
# to movie_id alone
# film effects may differ between regular users and critical users
# with come film being oscar bait films and other being crowd pleasers
# and some having overlap
# The academy clearly increased best picture nominiees just cause it wanted to
# and not cause of pushback from audiences on snubbed bat films, possible but timing...
# I mean just look at the tomatometer and audience score on rotten tomatoes
# the discrepancy required them to implement this at some point
# some critics are just oo out there and sometimes audiences are too large and
# popularity and other factors (FMAB like review bombing not being a factor of study in this case)
# come into effect

# Add user_type to training set
user_type <- edx_train %>%
  mutate(user_type = ifelse(user_id %in% critics, 'Critic', 'Regular') %>% as_factor()) %>%
  distinct(user_id, user_type)

edx_train <- edx_train %>%
  left_join(user_type) %>%
  relocate(user_type, .after = 'user_id')

## Batched Reviews --------------------------------------------------------

### Users -----------------------------------------------------------------

# Barbenheimer ~ why only one?
# Some reviews can be events

user_batched_reviews_ymdh <- edx_train %>%
  count(user_id, year, month, day, hour) %>%
  ggplot(aes(n)) +
  geom_histogram(aes(n, after_stat(ndensity)),
                 fill = '#FF0000',
                 alpha = 0.25) +
  geom_density(aes(n, after_stat(ndensity)),
               color = '#000000',
               alpha = 0.25) +
  scale_x_log10()

user_batched_reviews_ymdh

# As expected 1 is the most common value
# However there is a non-insignificant amount of batched reviews
# adding daily amount may be beneficial, it will be very strongly tied to the
# running count, the correlation analysis in feature selection may yield
# the choice of one

user_batched_reviews_ymd <- edx_train %>%
  count(user_id, year, month, day) %>%
  ggplot(aes(n)) +
  geom_histogram(aes(n, after_stat(ndensity)),
                 fill = '#FF0000',
                 alpha = 0.25) +
  geom_density(aes(n, after_stat(ndensity)),
               color = '#000000',
               alpha = 0.25) +
  scale_x_log10()

user_batched_reviews_ym <- edx_train %>%
  count(user_id, year, month) %>%
  ggplot(aes(n)) +
  geom_histogram(aes(n, after_stat(ndensity)),
                 fill = '#FF0000',
                 alpha = 0.25) +
  geom_density(aes(n, after_stat(ndensity)),
               color = '#000000',
               alpha = 0.25) +
  scale_x_log10()

user_batched_reviews_y <- edx_train %>%
  count(user_id, year) %>%
  ggplot(aes(n)) +
  geom_histogram(aes(n, after_stat(ndensity)),
                 fill = '#FF0000',
                 alpha = 0.25) +
  geom_density(aes(n, after_stat(ndensity)),
               color = '#000000',
               alpha = 0.25) +
  scale_x_log10()

user_batched_reviews_ymdh / user_batched_reviews_ymd / user_batched_reviews_ym / user_batched_reviews_y

# Slicing to all levels of time available we can see that there a
# number of users which only reviewed for 1 year
# hour may be too fine a time slice to be useful for estimating review counts
# year-month may be too large of a time slice
# adding a feature for ymd will be used

user_batched_reviews <- edx_train %>%
  count(user_id, year, month, day, name = 'user_reviews_day') %>%
  group_by(user_id) %>%
  mutate(user_reviews_day_accumulated = lag(cumsum(user_reviews_day), default = 0)) %>%
  ungroup()

edx_train <- left_join(edx_train, user_batched_reviews) %>%
  relocate(user_reviews_day, .after = 'user_reviews')

### Films -----------------------------------------------------------------

movie_batched_reviews_ymdh <- edx_train %>%
  count(movie_id, year, month, day, hour) %>%
  ggplot(aes(n)) +
  geom_histogram(aes(n, after_stat(ndensity)),
                 fill = '#FF0000',
                 alpha = 0.25) +
  geom_density(aes(n, after_stat(ndensity)),
               color = '#000000',
               alpha = 0.25) +
  scale_x_log10()

movie_batched_reviews_ymdh

# As expected 1 is the most common value

movie_batched_reviews_ymd <- edx_train %>%
  count(movie_id, year, month, day) %>%
  ggplot(aes(n)) +
  geom_histogram(aes(n, after_stat(ndensity)),
                 fill = '#FF0000',
                 alpha = 0.25) +
  geom_density(aes(n, after_stat(ndensity)),
               color = '#000000',
               alpha = 0.25) +
  scale_x_log10()

movie_batched_reviews_ym <- edx_train %>%
  count(movie_id, year, month) %>%
  ggplot(aes(n)) +
  geom_histogram(aes(n, after_stat(ndensity)),
                 fill = '#FF0000',
                 alpha = 0.25) +
  geom_density(aes(n, after_stat(ndensity)),
               color = '#000000',
               alpha = 0.25) +
  scale_x_log10()

movie_batched_reviews_y <- edx_train %>%
  count(movie_id, year) %>%
  ggplot(aes(n)) +
  geom_histogram(aes(n, after_stat(ndensity)),
                 fill = '#FF0000',
                 alpha = 0.25) +
  geom_density(aes(n, after_stat(ndensity)),
               color = '#000000',
               alpha = 0.25) +
  scale_x_log10()

movie_batched_reviews_ymdh / movie_batched_reviews_ymd / movie_batched_reviews_ym / movie_batched_reviews_y

# Films have an apparent better grouping as year-month as even day has too fine a
# time slice to create adequate binning
# will use ym for films

film_batched_reviews <- edx_train %>%
  count(movie_id, year, month, name = 'movie_reviews_year_month') %>%
  group_by(movie_id) %>%
  mutate(movie_reviews_day_accumulated = lag(cumsum(movie_reviews_year_month), default = 0)) %>%
  ungroup()

edx_train <- left_join(edx_train, film_batched_reviews) %>%
  relocate(movie_reviews_year_month, .after = 'movie_reviews')


## Accumulated Batches ----------------------------------------------------

edx_train

# Save User Types List and Train Set
saveRDS(user_type, file = file.path('Data', 'user_type.rds'))
saveRDS(edx_train, file = file.path('Data', 'edx_train.rds'))

# Clear Memory
# Keep Training Set
clear_memory(keep = 'edx_train')

# Feature Selection -------------------------------------------------------

# Remove non-predictors
edx_train <- edx_train %>%
  select(-all_of(c('timestamp', 'title')))

## Filter Methods ---------------------------------------------------------

### Near Zero Variance Predictors -----------------------------------------

edx_train_nzv <- nearZeroVar(
  select(edx_train, -all_of('rating')),
  freqCut = 80 / 20,
  uniqueCut = 20,
  saveMetrics = TRUE,
  names = TRUE,
  foreach = TRUE,
  allowParallel = TRUE
)

edx_train_nzv

# User Type, and accumulated reviews have near zero variance
# Considering what has been observed this selection will not be taken
# No other predictor exhibits this nzv

### Correlated Predictors -------------------------------------------------

library(ggcorrplot)

edx_train_num_predictors <- edx_train %>%
  # In order to capture any correlations convert ordered features
  # to numeric temporary for this study
  mutate(across(year, \(x) as.numeric(as.character(x)))) %>%
  mutate(across(where(is.ordered), as.numeric)) %>%
  select(where(is.numeric))

edx_train_num_cor_pearson <-
  cor(edx_train_num_predictors, method = 'pearson')
edx_train_num_cor_spearman <-
  cor(edx_train_num_predictors, method = 'spearman')

# edx_train_num_cor_pearson_pmat <-
#   cor_pmat(edx_train_num_predictors, method = 'pearson')
# edx_train_num_cor_spearman_pmat <-
#   cor_pmat(edx_train_num_predictors, method = 'spearman')

# Plot Pearson
edx_train_num_cor_pearson_plot <- ggcorrplot(
  edx_train_num_cor_pearson,
  type = 'lower',
  ggtheme = ggplot2::theme_bw,
  title = 'MovieLens Numeric Variable Pearson Correlations',
  hc.order = FALSE,
  lab = TRUE#,
  # p.mat = edx_train_num_cor_pearson_pmat
)

# Plot Spearman
edx_train_num_cor_spearman_plot <- ggcorrplot(
  edx_train_num_cor_spearman,
  type = 'lower',
  ggtheme = ggplot2::theme_bw,
  title = 'MovieLens Numeric Variable Spearman Correlations',
  hc.order = FALSE,
  lab = TRUE#,
  # p.mat = edx_train_num_cor_spearman_pmat
)

edx_train_num_cor_pearson_plot + edx_train_num_cor_spearman_plot

edx_train_num_cor_pearson_remove <-
  findCorrelation(
    edx_train_num_cor_pearson,
    cutoff = 0.8,
    verbose = TRUE,
    names = TRUE
  )

edx_train_num_cor_spearman_remove <-
  findCorrelation(
    edx_train_num_cor_spearman,
    cutoff = 0.8,
    verbose = TRUE,
    names = TRUE
  )

edx_train_num_cor_remove_all <-
  unique(c(
    edx_train_num_cor_pearson_remove,
    edx_train_num_cor_spearman_remove
  ))

# edx_train_num_cor_remove_all

# day of the year, Film age and base reviews were selected for removal
# Since PCA can use both year and year of release which calculate film age
# this option will be taken at this time in order to avoid linear combinations

edx_train <- edx_train %>%
  select(-all_of(edx_train_num_cor_remove_all))

# Of note are the features with some, abeit small correlation to rating
# film year of release, film age, user reviews and movie reviews have some
# direct correlation.
# While correlation is not causation these may warrant further exploration for
# Effects with other variables
# as it stands now the hypothesis is
# rating increases with film age
# rating decreases as film year of release increases
# rating increases as user's review more films

### Linear Dependencies ---------------------------------------------------

edx_train_num_predictors <- edx_train %>%
  # In order to capture any hidden combinations ordered features
  # to numeric temporary for this study
  mutate(across(year, \(x) as.numeric(as.character(x)))) %>%
  mutate(across(where(is.ordered), as.numeric)) %>%
  select(where(is.numeric))

edx_train_num_lcombos <- findLinearCombos(edx_train_num_predictors)

edx_train_num_lcombos

# There are no linear combinations within the numeric features

# Clear Memory
# Keep Training Set
clear_memory(keep = 'edx_train')

## Boruta Selection -------------------------------------------------------

library(Boruta)

# Given the size of the data set the importance function will be based on xgboost
# in contrast to the default of random forest

# Also, the current model has the form Y = user_effects + film_effects
# some of the predictors used in one effect bin may be confounders in the other
# therefore individual boruta analysis per each will be required
# in parallel there may be some confounding with review counts and time variables
# separate analysis will be used to see the relative performance of
# using either as effects
# weekday in not considered a time variable in this context but an enviorment variable
# as it does not directly influence reviews

edx_train_user_reviews <- edx_train %>%
  select(-all_of(c(
    'user_type', 'movie_id', 'year', 'month', 'day', 'hour'
  )))

edx_train_user_time <- edx_train %>%
  select(-all_of(c('user_type', 'movie_id')) & -contains('reviews'))

edx_train_film_reviews <- edx_train %>%
  select(-all_of(
    c(
      'user_id',
      'film_year_of_release',
      'genre_cluster',
      'year',
      'month',
      'day',
      'hour'
    )
  ) & -contains('user_reviews'))

edx_train_film_time <- edx_train %>%
  select(-all_of(c(
    'user_id', 'film_year_of_release', 'genre_cluster'
  )) & -contains('reviews'))

boruta_interpret <-
  function(x, title = NULL, subtitle = NULL) {
    decisions <- tibble(variable = names(x$finalDecision),
                        decision = as.character(x$finalDecision))
    
    importance <- as_tibble(x$ImpHistory) %>%
      pivot_longer(cols = everything(),
                   names_to = 'variable')
    
    data <- left_join(importance, decisions) %>%
      replace_na(list(decision = 'Metric')) %>%
      mutate(across(where(is.character), as.factor)) %>%
      mutate(variable = fct_reorder(variable, value, .desc = FALSE))
    
    plot <- data %>%
      ggplot(aes(variable, value, fill = decision)) +
      geom_boxplot(alpha = 0.25) +
      geom_jitter(position = position_jitterdodge()) +
      scale_y_continuous('Importance') +
      xlab('Predictor') +
      guides(fill = guide_legend('Decision')) +
      ggtitle(title, subtitle = subtitle) +
      coord_flip()
    
    return(plot)
    
  }

set.seed(756, sample.kind = 'Rounding')
boruta_user_reviews <- Boruta(
  rating ~ .,
  data = edx_train_user_reviews,
  doTrace = 3,
  getImp = getImpXgboost,
  maxRuns = 10000
)

set.seed(1956, sample.kind = 'Rounding')
boruta_user_time <- Boruta(
  rating ~ .,
  data = edx_train_user_time,
  doTrace = 3,
  getImp = getImpXgboost,
  maxRuns = 10000
)

set.seed(1300, sample.kind = 'Rounding')
boruta_film_reviews <- Boruta(
  rating ~ .,
  data = edx_train_film_reviews,
  doTrace = 3,
  getImp = getImpXgboost,
  maxRuns = 10000
)

set.seed(1385, sample.kind = 'Rounding')
boruta_film_time <- Boruta(
  rating ~ .,
  data = edx_train_film_time,
  doTrace = 3,
  getImp = getImpXgboost,
  maxRuns = 10000
)

edx_train_boruta_user_reviews <- boruta_interpret(
  boruta_user_reviews,
  'MovieLens User Effects with Reviews',
  'MovieLens predictor xgboost boruta importance'
)

edx_train_boruta_user_time <- boruta_interpret(
  boruta_user_time,
  'MovieLens User Effects with Time Variables',
  'MovieLens predictor xgboost boruta importance'
)

edx_train_boruta_film_reviews <- boruta_interpret(
  boruta_film_reviews,
  'MovieLens Film Effects with Reviews',
  'MovieLens predictor xgboost boruta importance'
)

edx_train_boruta_film_time <- boruta_interpret(
  boruta_film_time,
  'MovieLens Film Effects with Time Variables',
  'MovieLens predictor xgboost boruta importance'
)

edx_train_boruta_user_reviews / edx_train_boruta_user_time

# The users effect model would be simplified (greatly) by using time variables
# instead of counts with other variables overtaking the relative importance to a larger degree

getSelectedAttributes(boruta_user_time)

# Month and hour are better than noise as predictors but considering their
# low relative importance the model will be simplified to
# user_id (considered not important but this option will not be taken)
# film year of release, genre cluster and year

user_effect_predictors <-
  c('user_id', 'film_year_of_release', 'genre_cluster', 'year')

edx_train_boruta_film_reviews / edx_train_boruta_film_time

# Film effects are mostly centered on the individual film than with any other feature
# using time variables instead of counts the model simplified with movie_id and year
# taking the bulk of the variable importance

getSelectedAttributes(boruta_film_time)

# While user_type and other time variables are considered better than noise
# as predictors, their relative low importance will allow the further simplification
# to just movie_id and year

film_effect_predictors <- c('movie_id', 'year')

edx_train <- edx_train %>%
  select(all_of(c(
    'rating', user_effect_predictors, film_effect_predictors
  )))

# Save User Types List and Train Set
saveRDS(edx_train, file = file.path('Data', 'edx_train.rds'))

# Clear Memory
# Keep Training Set
clear_memory(keep = 'edx_train')

## Mutual Information -----------------------------------------------------

### Variable Rank ---------------------------------------------------------

library(varrank)

edx_train <-
  edx_train %>% mutate(across(contains('year'), \(x) as.numeric(as.character(x))))

edx_train_varrank <-
  varrank(
    edx_train,
    method = 'estevez',
    variable.important = 'rating',
    discretization.method = "sturges",
    algorithm = "forward",
    scheme = "mid",
    verbose = TRUE
  )

summary(edx_train_varrank)

plot(edx_train_varrank)

edx_train_varrank$ordered.var

# Redundant Interactions:
# Film:Year
# User:NONE
# Year functions as a non-interaction
# genre may not function significantly well as a non-interaction however
# the slope difference may be required for the user interaction

# Expected model:
# rating = b0 + b1*user + b2*genre + b3*user:genre + b4*year + b5*user:year + b6*film

edx_train <- edx_train %>%
  select(all_of(c('rating', edx_train_varrank$ordered.var)) &
           -all_of('film_year_of_release'))

### Mutual Information between Selected Predictors ------------------------

library(infotheo)

# Rating & Years as ordered factors
edx_train_fct <- edx_train %>%
  mutate(across(contains(c('year', 'rating')), as.ordered))

edx_train_mutual_info <- mutinformation(edx_train_fct)

mutual_info_max <- edx_train_mutual_info %>%
  as.data.frame() %>%
  rownames_to_column(var = 'var1') %>%
  pivot_longer(cols = -all_of('var1')) %>%
  filter(var1 != name) %>%
  pull(value) %>%
  max() %>%
  ceiling()

edx_train_mutual_info_plot <- ggcorrplot(
  edx_train_mutual_info,
  type = 'lower',
  ggtheme = ggplot2::theme_bw,
  title = 'MovieLens Factor Converted Variables Mutual Information',
  hc.order = FALSE,
  lab = TRUE
) +
  scale_fill_gradient2(
    'Mutual\nInformation',
    breaks = c(0, mutual_info_max),
    limit = c(0, mutual_info_max),
    low = '#000FFF',
    mid = '#FFFFFF',
    high = '#FF0000'
  )

edx_train_mutual_info_plot

# Potentially Collinear Interactions
# User ~ Genre + Year + Film
# Genre ~ Film (expected)
# Users have potential collinearity with all other predictors
# As expected Genre Clusters have collinear effects with Film
# As the base model has the form Rating ~ User Effects + Film Effects
# This should be fine as long as there is no interaction term between them
# and genre is used as an interaction term with users
#
# ridge regression (L2/Squared Regularization) will be applied
# in order to curve this effects

# Save User Types List and Train Set
saveRDS(edx_train, file = file.path('Data', 'edx_train.rds'))

# Clear Memory
# Keep Training Set
clear_memory(keep = 'edx_train')

# Train Model -------------------------------------------------------------

# As described the model will be trained by individual effects
# do to the large size of observations and categorical features
# an analytically solution to Ordinary Least Squares will be used
# to ensure that overfitting is not an issue
# ridge regression (L2/Squared Regularization) will be applied

# The core model is T = user effects + movie effects
# considering what was observed in the mean review by day
# the mean review could be used as the intercept and all other features
# causing deviations from this mean, then all predictors are 0
# the mean is the prediction

# The expected model form is:
# rating = b0 + b1*user + b2*genre + b3*user:genre + b4*year + b5*user:year + b6*film

# In cases where the coefficients are NA they'll be replaced with 0 for
# interactions

# A quasi-stepwise selection model will be trained for simplicity
# The main mode will be taken by forward selection with baclwards selection
# being implemented when a model exhibits negative RMSE effects
# in lieu of p-values the test set RMSE will be used if a predictor
# does not improve the model it will be taken out otherwise the
# selection will proccede

# numerical predictors will all be centered and scaled

# for the intercept the mean of the response will be used
# all other estimations will ignore the intercept calculation
# for numeric predictors the calculus derived formula:
# b_hat = sum(x*y)/sum(x^2)
#  based on minimized ordinary least squares (OLS) will be used
# for categorical predictors the grouped mean of the categories will be used
# as derived from the matrix operations from the matrix approach to OLS
# interactions will handled be multiple group_by variables
# L2 penalties will be calculated at each training step
# note that having an optimal penaty term of 0 (zero)
# will revert the coefficient back to the non-regularized term

## Center and Scale Predictors --------------------------------------------

numeric_prep_model <- preProcess(edx_train,
                                 method = c('center', 'scale'))

edx_train <- predict(numeric_prep_model, edx_train)

saveRDS(numeric_prep_model,
        file = file.path('Data', 'numeric_prep_model.rds'))

# Since Centering and Scaling is the only preprocessing preformed this
# can be reversed after model training using a custom function
# This is expected to improve training by centering and normalizing the
# response and then reverting back into a interpretable result

reverse_prep <- function(prep_model, data, digits = 0) {
  data <- data %>%
    select(one_of(prep_model$mean %>% names)) %>%
    map2_df(prep_model$std, ., function(sig, dat)
      dat * sig) %>%
    map2_df(prep_model$mean, ., function(mu, dat)
      dat + mu)
  
  return(data)
  
}

## Prepare Test Set -------------------------------------------------------

edx_data_prep <- function(data, keep_cols = NULL) {
  # Load Genre Cluster
  genres <-
    readRDS('~/Data Projects/MovieLens-10M/Data/edx_genre_clusters.rds')
  
  # Load Scaling and Centering Models
  numeric_prep_model <-
    readRDS('~/Data Projects/MovieLens-10M/Data/numeric_prep_model.rds')
  
  # Mutate Data
  data <- data %>%
    # Relocate response variable as preferred
    relocate(rating) %>%
    # Clean timestamp
    mutate(across(timestamp, as_datetime)) %>%
    # sort by time
    arrange(timestamp) %>%
    # Separate Title Features
    separate_wider_regex(
      title,
      patterns = c(
        title = '[:print:]+(?=(?:[:space:]\\([:digit:]{4}\\)))',
        ' \\(',
        film_year_of_release = '.*',
        '\\)'
      )
    ) %>%
    # Engineer date-time features
    mutate(
      film_year_of_release = as.numeric(str_extract(film_year_of_release, '[:digit:]{4}')),
      # Convert timestamp to date-time
      timestamp = as_datetime(timestamp),
      # Extract date features
      year = year(timestamp),
      month = month(timestamp, label = TRUE, abbr = FALSE),
      day = day(timestamp),
      # Day of the Week, weeks starts on Monday
      weekday = wday(
        timestamp,
        label = TRUE,
        abbr = FALSE,
        week_start = 1
      ),
      # Extract Time Features
      hour = hour(timestamp),
      minute = minute(timestamp),
      second = second(timestamp)
    ) %>%
    # Remove Genres & Titles
    select(-all_of(c('genres', 'title', 'timestamp'))) %>%
    # clean Names
    clean_names() %>%
    # IDs as Factors
    mutate(across(ends_with('_id'), as.factor)) %>%
    left_join(genres) %>%
    mutate(across(year, \(x) as.numeric(as.character(x)))) %>%
    mutate(across(where(is.ordered), as.numeric))
  
  # Center and Scale
  data <- as_tibble(predict(numeric_prep_model, data))
  
  # Select Predictors
  data <- data %>%
    select(all_of(keep_cols))
  
  return(data)
  
}

# Load Test Set
edx_test <-
  readRDS('~/Data Projects/MovieLens-10M/Data/edx_test.rds')

# Prepare
edx_test <- edx_data_prep(edx_test, keep_cols = names(edx_train))

# Validate Preparation
edx_train
edx_test

## Train Intercept --------------------------------------------------------

# The expected model form is:
# rating = b0 + b1*user + b2*genre + b3*user:genre + b4*year + b5*user:year + b6*film

# Set Model Coefficient as tibbles
user_model <- tibble(user_id = sort(unique(edx_train$user_id)))
film_model <- tibble(movie_id = sort(unique(edx_train$movie_id)))

model_intercept <- edx_train %>%
  summarise(b0 = mean(rating))

model_rmse <- edx_test %>%
  cross_join(model_intercept) %>%
  mutate(y_hat = b0) %>%
  summarise(model = 'Intercept',
            rmse = RMSE(y_hat, rating),)

model_rmse

# Intercept Model has an RMSE of 1.00

edx_train <- edx_train %>%
  cross_join(model_intercept) %>%
  mutate(y_hat = b0)

## Train User Model -------------------------------------------------------

# The expected model form is:
# rating = b0 + b1*user + b2*genre + b3*user:genre + b4*year + b5*user:year + b6*film

### Train B1 --------------------------------------------------------------

user_model <- edx_train %>%
  group_by(user_id) %>%
  summarise(n = n(),
            b1 = mean(rating - y_hat),) %>%
  left_join(user_model, .) %>%
  relocate(starts_with('b'), .after = everything())

#### Regularize -----------------------------------------------------------

lambda <- tibble(lambda = seq(0, 10, length.out = 101))

user_model_reg_prime <- edx_train %>%
  left_join(user_model) %>%
  mutate(y_hat = rating - b0) %>%
  group_by(user_id) %>%
  summarise(n = n(),
            y_hat = mean(y_hat)) %>%
  cross_join(lambda) %>%
  mutate(b1_reg = y_hat / (n + lambda)) %>%
  select(all_of(c('user_id', 'lambda', 'b1_reg')))

user_model_reg_prime_rmse <- edx_test %>%
  cross_join(model_intercept) %>%
  left_join(user_model_reg_prime,
            relationship = 'many-to-many') %>%
  mutate(y_hat = b0 + b1_reg) %>%
  group_by(lambda) %>%
  summarise(rmse = RMSE(y_hat, rating))

user_model_reg_prime_rmse %>%
  slice_min(rmse)

# Optimal L2 Penalty Term is 0
# model will use previous b1 calculation

user_model <- user_model %>%
  select(-all_of(c('n')))

clear_memory(keep = c(
  'edx_train',
  'edx_test',
  'model_intercept',
  'user_model',
  'lambda'
))

### Train B2 --------------------------------------------------------------

genre_model <- edx_train %>%
  left_join(user_model) %>%
  mutate(y_hat = b0 + b1) %>%
  group_by(genre_cluster) %>%
  summarise(n = n(),
            b2 = mean(rating - y_hat))

#### Regularize -----------------------------------------------------------

genre_model_reg_prime <- edx_train %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  mutate(y_hat = rating - b0 - b1) %>%
  group_by(genre_cluster) %>%
  summarise(n = first(n),
            y_hat = mean(y_hat)) %>%
  cross_join(lambda) %>%
  mutate(b2 = y_hat / (n + lambda)) %>%
  select(all_of(c('genre_cluster', 'lambda', 'b2')))

genre_model_reg_prime_rmse <- edx_test %>%
  cross_join(model_intercept) %>%
  left_join(user_model) %>%
  left_join(genre_model_reg_prime,
            relationship = 'many-to-many') %>%
  mutate(y_hat = b0 + b1 + b2) %>%
  group_by(lambda) %>%
  summarise(rmse = RMSE(y_hat, rating))

genre_model_reg_lambda <- genre_model_reg_prime_rmse %>%
  slice_min(rmse)

genre_model_reg_lambda

# Optimal L2 Penalty Term is 0
# model will use previous b2 calculation

genre_model <- genre_model %>%
  select(-all_of(c('n')))

clear_memory(
  keep = c(
    'edx_train',
    'edx_test',
    'model_intercept',
    'lambda',
    'user_model',
    'genre_model'
  )
)

### Train B3 --------------------------------------------------------------

user_genre_n <- edx_train %>%
  count(user_id, genre_cluster)

user_genre_model <- edx_train %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  mutate(y_hat = b0 + b1 + b2) %>%
  group_by(user_id, genre_cluster) %>%
  summarise(b3 = mean(rating - y_hat)) %>%
  ungroup()

#### Regularize -----------------------------------------------------------

user_genre_model_reg_prime <- edx_train %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  left_join(user_genre_model) %>%
  left_join(user_genre_n) %>%
  mutate(y_hat = rating - b0 - b1 - b2) %>%
  group_by(user_id, genre_cluster) %>%
  summarise(n = first(n),
            y_hat = mean(y_hat)) %>%
  cross_join(lambda) %>%
  mutate(b3 = y_hat / (n + lambda)) %>%
  ungroup()

user_genre_model_reg_prime_rmse <- edx_test %>%
  cross_join(model_intercept) %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  left_join(user_genre_model_reg_prime,
            relationship = 'many-to-many') %>%
  mutate(y_hat = b0 + b1 + b2 + b3) %>%
  group_by(lambda) %>%
  summarise(rmse = RMSE(y_hat, rating))

user_genre_model_reg_lambda <- user_genre_model_reg_prime_rmse %>%
  slice_min(rmse)

user_genre_model_reg_lambda

# Optimal L2 Penalty Term is 4.8

user_genre_model_reg_lambda <- user_genre_model_reg_lambda %>%
  pull(lambda)

user_genre_model <- user_genre_model %>%
  left_join(user_genre_n) %>%
  mutate(b3 = b3 / (n + user_genre_model_reg_lambda)) %>%
  select(-all_of(c('n')))

clear_memory(
  keep = c(
    'edx_train',
    'edx_test',
    'model_intercept',
    'lambda',
    'user_model',
    'genre_model',
    'user_genre_model'
  )
)

### Train B4 --------------------------------------------------------------

year_model <- edx_train %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  left_join(user_genre_model) %>%
  mutate(y_hat = b0 + b1 + b2 + b3) %>%
  summarise(b4 = sum((year) * (rating - y_hat)) / sum(year ^ 2))

year_model

#### Regularize -----------------------------------------------------------

year_model_reg_prime <- edx_train %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  left_join(user_genre_model) %>%
  cross_join(year_model) %>%
  mutate(y_hat = rating - b0 - b1 - b2 - b3) %>%
  summarise(n = n(),
            y_hat = mean(y_hat)) %>%
  cross_join(lambda) %>%
  mutate(b4 = y_hat / (n + lambda))

year_model_reg_prime_rmse <- edx_test %>%
  cross_join(model_intercept) %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  left_join(user_genre_model) %>%
  cross_join(year_model_reg_prime) %>%
  mutate(across(starts_with('b'), \(x) replace_na(x, 0))) %>%
  mutate(y_hat = b0 + b1 + b2 + b3 + b4) %>%
  group_by(lambda) %>%
  summarise(rmse = RMSE(y_hat, rating)) %>%
  ungroup()

year_model_reg_lambda <- year_model_reg_prime_rmse %>%
  slice_min(rmse, with_ties = FALSE)

year_model_reg_lambda

# Optimal L2 Penalty is 0
# model will use previous b4 calculation

clear_memory(
  keep = c(
    'edx_train',
    'edx_test',
    'model_intercept',
    'lambda',
    'user_model',
    'genre_model',
    'user_genre_model',
    'year_model'
  )
)

### Train B5 --------------------------------------------------------------

user_year_model <- edx_train %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  left_join(user_genre_model) %>%
  cross_join(year_model) %>%
  mutate(y_hat = b0 + b1 + b2 + b3 + b4) %>%
  group_by(user_id) %>%
  summarise(b5 = sum((year) * (rating - y_hat)) / sum(year ^ 2)) %>%
  ungroup()

user_year_model

#### Regularize -----------------------------------------------------------

user_year_model_reg_prime <- edx_train %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  left_join(user_genre_model) %>%
  cross_join(year_model) %>%
  left_join(user_year_model) %>%
  mutate(across(starts_with('b'), \(x) replace_na(x, 0))) %>%
  mutate(y_hat = rating - b0 - b1 - b2 - b3 - b4) %>%
  group_by(user_id) %>%
  summarise(n = n(),
            y_hat = mean(y_hat)) %>%
  ungroup() %>%
  cross_join(lambda) %>%
  mutate(b5 = y_hat / (n + lambda))

user_year_reg_prime_rmse <- edx_test %>%
  cross_join(model_intercept) %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  left_join(user_genre_model) %>%
  cross_join(year_model) %>%
  left_join(user_year_model_reg_prime,
            relationship = 'many-to-many') %>%
  mutate(across(starts_with('b'), \(x) replace_na(x, 0))) %>%
  mutate(y_hat = b0 + b1 + b2 + b3 + b4) %>%
  group_by(lambda) %>%
  summarise(rmse = RMSE(y_hat, rating)) %>%
  ungroup()

user_year_reg_prime_rmse %>%
  slice_min(rmse, with_ties = FALSE)

# Optimal L2 Penalty is 0
# model will use previous b5 calculation

clear_memory(
  keep = c(
    'edx_train',
    'edx_test',
    'model_intercept',
    'lambda',
    'user_model',
    'genre_model',
    'user_genre_model',
    'year_model',
    'user_year_model'
  )
)

### Train B6 --------------------------------------------------------------

film_model <- edx_train %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  left_join(user_genre_model) %>%
  cross_join(year_model) %>%
  left_join(user_year_model) %>%
  mutate(y_hat = b0 + b1 + b2 + b3 + b4 + b5) %>%
  group_by(movie_id) %>%
  summarise(b6 = mean(rating - y_hat)) %>%
  ungroup()

film_model

#### Regularize -----------------------------------------------------------

film_model_reg_prime <- edx_train %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  left_join(user_genre_model) %>%
  cross_join(year_model) %>%
  left_join(user_year_model) %>%
  mutate(across(starts_with('b'), \(x) replace_na(x, 0))) %>%
  mutate(y_hat = rating - b0 - b1 - b2 - b3 - b4 - b5) %>%
  group_by(movie_id) %>%
  summarise(n = n(),
            y_hat = mean(y_hat)) %>%
  ungroup() %>%
  cross_join(lambda) %>%
  mutate(b6 = y_hat / (n + lambda)) %>%
  ungroup()

film_model_reg_prime_rmse <- edx_test %>%
  cross_join(model_intercept) %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  left_join(user_genre_model) %>%
  cross_join(year_model) %>%
  left_join(user_year_model) %>%
  left_join(film_model_reg_prime,
            relationship = 'many-to-many') %>%
  mutate(across(starts_with('b'), \(x) replace_na(x, 0))) %>%
  mutate(y_hat = b0 + b1 + b2 + b3 + b4 + b5) %>%
  group_by(lambda) %>%
  summarise(rmse = RMSE(y_hat, rating)) %>%
  ungroup()

film_model_reg_prime_rmse %>%
  slice_min(rmse, with_ties = FALSE)

# Optimal L2 Penalty is 0
# model will use previous b6 calculation

clear_memory(
  keep = c(
    'edx_train',
    'edx_test',
    'model_intercept',
    'lambda',
    'user_model',
    'genre_model',
    'user_genre_model',
    'year_model',
    'user_year_model',
    'film_model'
  )
)

# Test Model --------------------------------------------------------------

# The expected model form is:
# rating = b0 + b1*user + b2*genre + b3*user:genre + b4*year + b5*user:year + b6*film

edx_test_rmse <- edx_test %>%
  cross_join(model_intercept) %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  left_join(user_genre_model) %>%
  cross_join(year_model) %>%
  left_join(user_year_model) %>%
  left_join(film_model) %>%
  mutate(across(starts_with('b'), \(x) replace_na(x, 0))) %>%
  mutate(
    y_hat_intercept = b0,
    y_hat_user = b0 + b1,
    y_hat_genre = b0 + b1 + b2,
    y_hat_user_genre = b0 + b1 + b2 + b3,
    y_hat_year = b0 + b1 + b2 + b3 + b4,
    y_hat_user_year = b0 + b1 + b2 + b3 + b4 + b5,
    y_hat_film = b0 + b1 + b2 + b3 + b4 + b5 + b6
  ) %>%
  summarise(across(starts_with('y_hat'), \(x) RMSE(x, rating))) %>%
  pivot_longer(
    cols = everything(),
    names_to = 'model',
    values_to = 'rmse',
    names_transform = list(model = \(x) as_factor(str_to_title(
      str_replace(str_remove(x, 'y_hat_'), '_', '-')
    ))),
  )

edx_test_rmse %>%
  mutate(rmse_diff = lag(rmse) - rmse)

edx_test_rmse %>%
  ggplot(aes(model, rmse, group = 1)) +
  geom_line()

# Model has improves RMSE up to the inclusion of User-Year
# This will be taken out of the model and Film will be retrained

clear_memory(
  keep = c(
    'edx_train',
    'edx_test',
    'model_intercept',
    'lambda',
    'user_model',
    'genre_model',
    'user_genre_model',
    'year_model',
    'user_year_model',
    'film_model',
    'edx_test_rmse'
  )
)

# Backwards Selection and Retrain -----------------------------------------

# This step will just require retraining film effects after not considering b5
# in the training step

## Retrain B6 -------------------------------------------------------------

film_model <- edx_train %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  left_join(user_genre_model) %>%
  cross_join(year_model) %>%
  left_join(user_year_model) %>%
  mutate(y_hat = b0 + b1 + b2 + b3 + b4) %>%
  group_by(movie_id) %>%
  summarise(b6 = mean(rating - y_hat)) %>%
  ungroup()

film_model

### Regularize ------------------------------------------------------------

film_model_reg_prime <- edx_train %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  left_join(user_genre_model) %>%
  cross_join(year_model) %>%
  left_join(user_year_model) %>%
  mutate(across(starts_with('b'), \(x) replace_na(x, 0))) %>%
  mutate(y_hat = rating - b0 - b1 - b2 - b3 - b4) %>%
  group_by(movie_id) %>%
  summarise(n = n(),
            y_hat = mean(y_hat)) %>%
  ungroup() %>%
  cross_join(lambda) %>%
  mutate(b6 = y_hat / (n + lambda)) %>%
  ungroup()

film_model_reg_prime_rmse <- edx_test %>%
  cross_join(model_intercept) %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  left_join(user_genre_model) %>%
  cross_join(year_model) %>%
  left_join(user_year_model) %>%
  left_join(film_model_reg_prime,
            relationship = 'many-to-many') %>%
  mutate(across(starts_with('b'), \(x) replace_na(x, 0))) %>%
  mutate(y_hat = b0 + b1 + b2 + b3 + b4) %>%
  group_by(lambda) %>%
  summarise(rmse = RMSE(y_hat, rating)) %>%
  ungroup()

film_model_reg_prime_rmse %>%
  slice_min(rmse, with_ties = FALSE)

# Optimal L2 Penalty is 0
# model will use previous b6 calculation

clear_memory(
  keep = c(
    'edx_train',
    'edx_test',
    'model_intercept',
    'lambda',
    'user_model',
    'genre_model',
    'user_genre_model',
    'year_model',
    'user_year_model',
    'film_model'
  )
)

# Retest Model ------------------------------------------------------------

# The expected model form is:
# rating = b0 + b1*user + b2*genre + b3*user:genre + b4*year + b6*film

edx_test_rmse <- edx_test %>%
  cross_join(model_intercept) %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  left_join(user_genre_model) %>%
  cross_join(year_model) %>%
  left_join(user_year_model) %>%
  left_join(film_model) %>%
  mutate(across(starts_with('b'), \(x) replace_na(x, 0))) %>%
  mutate(
    y_hat_intercept = b0,
    y_hat_user = b0 + b1,
    y_hat_genre = b0 + b1 + b2,
    y_hat_user_genre = b0 + b1 + b2 + b3,
    y_hat_year = b0 + b1 + b2 + b3 + b4,
    y_hat_film = b0 + b1 + b2 + b3 + b4 + b6
  ) %>%
  summarise(across(starts_with('y_hat'), \(x) RMSE(x, rating))) %>%
  pivot_longer(
    cols = everything(),
    names_to = 'model',
    values_to = 'rmse',
    names_transform = list(model = \(x) as_factor(str_to_title(
      str_replace(str_remove(x, 'y_hat_'), '_', '-')
    ))),
  )

edx_test_rmse %>%
  mutate(rmse_diff = lag(rmse) - rmse)

edx_test_rmse %>%
  ggplot(aes(model, rmse, group = 1)) +
  geom_line()

# The Model RMSE is 0.831
# While there's an improvement in RMSE using year as a predictor it is not significant
# will retrain b6 while removing this to gauge RMSE improvements

clear_memory(
  keep = c(
    'edx_train',
    'edx_test',
    'model_intercept',
    'lambda',
    'user_model',
    'genre_model',
    'user_genre_model',
    'year_model',
    'user_year_model',
    'film_model'
  )
)

# Backwards Selection and Retrain 2 ---------------------------------------

# This step will just require retraining film effects after not considering b4
# in the training step

## Retrain B6 -------------------------------------------------------------

film_model <- edx_train %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  left_join(user_genre_model) %>%
  cross_join(year_model) %>%
  left_join(user_year_model) %>%
  mutate(y_hat = b0 + b1 + b2 + b3) %>%
  group_by(movie_id) %>%
  summarise(b6 = mean(rating - y_hat)) %>%
  ungroup()

film_model

### Regularize ------------------------------------------------------------

film_model_reg_prime <- edx_train %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  left_join(user_genre_model) %>%
  cross_join(year_model) %>%
  left_join(user_year_model) %>%
  mutate(across(starts_with('b'), \(x) replace_na(x, 0))) %>%
  mutate(y_hat = rating - b0 - b1 - b2 - b3) %>%
  group_by(movie_id) %>%
  summarise(n = n(),
            y_hat = mean(y_hat)) %>%
  ungroup() %>%
  cross_join(lambda) %>%
  mutate(b6 = y_hat / (n + lambda)) %>%
  ungroup()

film_model_reg_prime_rmse <- edx_test %>%
  cross_join(model_intercept) %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  left_join(user_genre_model) %>%
  cross_join(year_model) %>%
  left_join(user_year_model) %>%
  left_join(film_model_reg_prime,
            relationship = 'many-to-many') %>%
  mutate(across(starts_with('b'), \(x) replace_na(x, 0))) %>%
  mutate(y_hat = b0 + b1 + b2 + b3) %>%
  group_by(lambda) %>%
  summarise(rmse = RMSE(y_hat, rating)) %>%
  ungroup()

film_model_reg_prime_rmse %>%
  slice_min(rmse, with_ties = FALSE)

# Optimal L2 Penalty is 0
# model will use previous b6 calculation

clear_memory(
  keep = c(
    'edx_train',
    'edx_test',
    'model_intercept',
    'lambda',
    'user_model',
    'genre_model',
    'user_genre_model',
    'year_model',
    'user_year_model',
    'film_model'
  )
)

# Retest Model 2 ----------------------------------------------------------

# The expected model form is:
# rating = b0 + b1*user + b2*genre + b3*user:genre + b6*film

edx_test_rmse <- edx_test %>%
  cross_join(model_intercept) %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  left_join(user_genre_model) %>%
  cross_join(year_model) %>%
  left_join(user_year_model) %>%
  left_join(film_model) %>%
  mutate(across(starts_with('b'), \(x) replace_na(x, 0))) %>%
  mutate(
    y_hat_intercept = b0,
    y_hat_user = b0 + b1,
    y_hat_genre = b0 + b1 + b2,
    y_hat_user_genre = b0 + b1 + b2 + b3,
    y_hat_film = b0 + b1 + b2 + b3 + b6
  ) %>%
  summarise(across(starts_with('y_hat'), \(x) RMSE(x, rating))) %>%
  pivot_longer(
    cols = everything(),
    names_to = 'model',
    values_to = 'rmse',
    names_transform = list(model = \(x) as_factor(str_to_title(
      str_replace(str_remove(x, 'y_hat_'), '_', '-')
    ))),
  )

edx_test_rmse %>%
  mutate(rmse_diff = lag(rmse) - rmse)

edx_test_rmse %>%
  ggplot(aes(model, rmse, group = 1)) +
  geom_line()

# The Model RMSE is 0.831
# These was no noticeable change in RMSE when removing year as a predictor
# This is considered the optimal model as it has minimized the amount of predictors and
# minimized RMSE

# This model will be applied to the hold-over set

# The FINAL model form is:
# rating = b0 + b1*user + b2*genre + b3*user:genre + b6*film

clear_memory(
  keep = c(
    'edx_train',
    'edx_test',
    'model_intercept',
    'lambda',
    'user_model',
    'genre_model',
    'user_genre_model',
    'year_model',
    'user_year_model',
    'film_model'
  )
)

# Apply Model to Holdover Set ---------------------------------------------

train_names <- edx_train %>%
  select(-all_of(c('b0', 'y_hat'))) %>%
  names()

# Load Holdout Set
final_holdout_test_prime <-
  readRDS('~/Data Projects/MovieLens-10M/Data/final_holdout_test.rds')

# Load Scaling and Centering Models
numeric_prep_model <-
  readRDS('~/Data Projects/MovieLens-10M/Data/numeric_prep_model.rds')

# Prepare Holdout Set
final_holdout_test <-
  edx_data_prep(final_holdout_test_prime, keep_cols = train_names)

final_holdout_test_pred <- final_holdout_test %>%
  cross_join(model_intercept) %>%
  left_join(user_model) %>%
  left_join(genre_model) %>%
  left_join(user_genre_model) %>%
  left_join(film_model) %>%
  mutate(across(starts_with('b'), \(x) replace_na(x, 0))) %>%
  mutate(y_hat = b0 + b1 + b2 + b3 + b6)

final_holdout_test_pred %>%
  summarise(rmse = RMSE(y_hat, rating))

# The final model has an RMSE of 0.831 on the final holdout set
# The FINAL model form is validated:
# rating = b0 + b1*user + b2*genre + b3*user:genre + b6*film

# For interpretable results the reverse_prep function can be used
final_holdout_test_pred %>%
  select(-all_of(c('rating'))) %>%
  rename('rating' = 'y_hat') %>%
  reverse_prep(numeric_prep_model, .) %>%
  rename_with(.cols = everything(),
              .fn = \(x) paste('reversed', x, sep = '_')) %>%
  bind_cols(final_holdout_test_pred) %>%
  select(contains('rating'))

# Report Considerations ---------------------------------------------------

# Write Report including
# Cleaning
# Feature Engineering of Genre Clusters
# Feature Selection
# Highlight Boruta & Mutual Information
# Ignore Critics and additional time values for ease of processing
# Model Training
# Highlight Coefficients which were regularized vs others
# reversal of centering and scaling
# Conclusions

options(knitr.duplicate.label = "allow")
=======
title_words_counts <- title_words %>%
  count(title_words, sort = TRUE)

mode_stat <- function(x){
  uniqv <- unique(x)
  uniqv[which.max(tabulate(match(x, uniqv)))]
}

title_words_counts %>% 
  summarise(mean = mean(n),
            sd = sd(n),
            median = median(n),
            mode = mode_stat(n),
            min = min(n),
            max = max(n))

# At least 3 clusters are required
# There are a a large number of single count words

title_modes <- tibble(clusters = 3:2489,
       data = list(select(title_words, all_of('title_words')))
       ) %>% 
  mutate(k_modes = map2(
    data,
    clusters,
    seed = 2250,
    weighted = TRUE,
    # Using Weighted distances to account for differences in genre frequencies
    kmodes_fn,
    .progress = 'K-Modes'
  ))

title_modes
>>>>>>> 2f642d08513c447d1894bd7b728cfe9bea190c62
