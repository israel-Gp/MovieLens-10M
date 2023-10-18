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
# will differ in popularity (e.g. The Dark Knight vs Batman & Robin)
# A film based model is required to account for this missing input

# A ensemble model can be utilized to merge the different model
# for simplicity a linear regression stacked model will be used

# There's no accounting for taste?
# In order to explore that taste can be accounted for user based predictions will include
# a running count of the amount of reviews a user has made, this is to see whether a
# user has developed a pallet and therefore becomes more strict or looser in their
# ratings

# In contrast a running count of film reviews will be added as a measure of viewing popularity
# this can be used for both film and user individualized predictions given that, at extremes,
# some people are trend followers and others may actively avoid it.
# For appropriate use of counts these will be lagged measures as to account for data leakage

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
  rename_with(~ str_replace(.x, '^genres', 'genre')) %>%
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
  ) %>%
  # Relocate features as preffered
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

# Save Train and Test Set to File for storage
saveRDS(edx_train, file = file.path('Data', 'edx_train.rds'))

# Clear Memory
# Keep Training Set
clear_memory(keep = 'edx_train')

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

tidy(stats::chisq.test(edx_train$user_id, edx_train$movie_id))

# Both Variables are independent of each other

# Users on average rate higher than the overall mean while films are rated lower
# than the overall mean

edx_train_id_means %>%
  count(id_type) %>%
  arrange(desc(n))

# There are more users than films in the data
# A significant amount of films are rated far lower than the mean rating

## Film Age ---------------------------------------------------------------

library(scales)

edx_train %>%
  count(film_age) %>%
  ggplot(aes(film_age, n)) +
  geom_col(fill = '#FF0000', alpha = 0.5) +
  scale_y_continuous('Count', labels = comma) +
  xlab('Film Age') +
  ggtitle('Film Age')

# Film age is approximately negative binomial

quantile(edx_train$film_age, probs = seq(0, 1, 0.1))
mean(edx_train$film_age == 0) * 100

# Only 4.23% of films reviewed are same year reviews
# 60% of reviews are done within 10 years of release
# There are two films with negative years

edx_train %>%
  distinct(movie_id, film_age) %>%
  count(preterm  = film_age < 0)

# 23 Films are premature reviews

## Film Year of Release ---------------------------------------------------

edx_train %>%
  count(film_year_of_release) %>%
  ggplot(aes(film_year_of_release, n)) +
  geom_col(fill = '#FF0000', alpha = 0.5) +
  scale_y_continuous('Count', labels = comma) +
  xlab('Film Age') +
  ggtitle('Film Age')

# Film year of release follows an inverted log-normal distribution

quantile(edx_train$film_year_of_release, probs = seq(0, 1, 0.1))

t_test(edx_train$film_year_of_release)

# The mean year of release is 1990 with a very narrow CI, both low and high being 1990

sd(edx_train$film_year_of_release)

# SD is 13.59 (~14) years

mean(
  edx_train$film_year_of_release >= 1990 - 14 &
    edx_train$film_year_of_release <= 1990 + 14
)

# 84.5% of all reviewed films were released within 14 years of 1990

## User Reviews -----------------------------------------------------------

edx_train %>%
  count(user_reviews) %>%
  ggplot(aes(user_reviews, n)) +
  geom_col(fill = '#FF0000', alpha = 0.5) +
  scale_y_continuous('Count', labels = comma) +
  scale_x_continuous('User Reivews', labels = comma) +
  ggtitle('User Reivews')

# As a lag count of previous user reviews the count will include zero for every user
# Therefore the user reviews will follow a negative binomial distribution

## Film Reviews -----------------------------------------------------------

edx_train %>%
  count(movie_reviews) %>%
  ggplot(aes(movie_reviews, n)) +
  geom_col(fill = '#FF0000', alpha = 0.5) +
  scale_y_continuous('Count', labels = comma) +
  scale_x_continuous('FIlm Reivews', labels = comma) +
  ggtitle('Film Reivews')

# As a lag count of previous user reviews the count will include zero for every user
# Therefore the user reviews will follow a negative binomial distribution

## Genres -----------------------------------------------------------------

edx_train_genre_stats <- edx_train %>%
  select(starts_with('genre_')) %>%
  pivot_longer(
    cols = everything(),
    names_to = 'level',
    values_to = 'genre',
    names_transform = list(level = ~ as_factor(str_extract(.x, '[:digit:]+')))
  ) %>%
  count(level, genre, sort = TRUE) %>%
  group_by(level) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()

edx_train_genre_stats %>%
  ggplot(aes(
    level,
    genre,
    fill = prop,
    label = label_percent(accuracy = 0.1)(prop)
  )) +
  geom_tile(color = '#000000') +
  geom_text() +
  scale_fill_continuous('Proportion',
                        high = '#F5A9B8',
                        low = '#5BCEFA') +
  xlab('Genre Level') +
  ylab('Genre') +
  ggtitle('Genre Proportions by Level')

# Levels 3 to 8 have a significant amount of 'None' genres
# These empty genres are unlikely to be significant for predictions and will be dropped

edx_train_genre_remove <- edx_train_genre_stats %>%
  filter(prop >= 0.2 & genre == 'None') %>%
  pull(level) %>%
  as.numeric() %>%
  sort() %>%
  str_c('genre', ., sep = '_')

edx_train <- edx_train %>%
  select(-all_of(edx_train_genre_remove))

edx_train_genre_stats %>%
  filter(level %in% 1:2) %>%
  ggplot(aes(
    level,
    genre,
    fill = prop,
    label = label_percent(accuracy = 0.1)(prop)
  )) +
  geom_tile(color = '#000000') +
  geom_text() +
  scale_fill_continuous('Proportion',
                        high = '#F5A9B8',
                        low = '#5BCEFA') +
  xlab('Genre Level') +
  ylab('Genre') +
  ggtitle('Genre Proportions by Level')

# For optimal feature usage genres will be clustered considered

### Genre Clusters --------------------------------------------------------

library(ggrepel)

edx_genre_wide <- edx_train %>%
  distinct(movie_id, .keep_all = TRUE) %>%
  select(c(movie_id, starts_with('genre')))

edx_genre_selections <- edx_genre_wide %>%
  pivot_longer(
    cols = starts_with('genre_'),
    names_to = 'genre',
    names_transform = list(word = as_factor)
  ) %>%
  mutate(is_na = str_detect(value, 'None')) %>%
  group_by(genre) %>%
  summarise(na_prop = mean(is_na == TRUE)) %>%
  filter(na_prop <= 0.5) %>%
  pull(genre) %>%
  as.character()

edx_genre_removal <- edx_genre_wide %>%
  pivot_longer(
    cols = starts_with('genre_'),
    names_to = 'genre',
    names_transform = list(word = as_factor)
  ) %>%
  mutate(is_na = str_detect(value, 'None')) %>%
  group_by(genre) %>%
  summarise(na_prop = mean(is_na == TRUE)) %>%
  filter(na_prop > 0.5) %>%
  pull(genre) %>%
  as.character()

edx_genre_means <- edx_train %>%
  select(all_of(c('rating', edx_genre_selections))) %>%
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

edx_genre_wide_num <- edx_genre_wide %>%
  select(all_of(edx_genre_selections)) %>%
  mutate(across(everything(), as.numeric))

genre_clusters_min <- edx_genre_wide %>%
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

# Genre Clusters Determined using K-Modes

library(furrr)
library(doParallel)

# Custom K-Modes Function for nested tibbles
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

# Save Train and Genre Clusters
saveRDS(edx_train, file = file.path('Data', 'edx_train.rds'))
saveRDS(edx_genre_clusters, file = file.path('Data', 'edx_genre_clusters.rds'))

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

edx_train_nzv_remove <- edx_train_nzv %>%
  rownames_to_column() %>%
  as_tibble() %>%
  filter(nzv == 'TRUE') %>%
  pull(rowname)

edx_train_nzv

edx_train <- edx_train %>%
  select(-all_of(edx_train_nzv_remove))

# There are no predictors with near zero variance

### Correlated Predictors -------------------------------------------------

library(ggcorrplot)
library(patchwork)

edx_train_num_predictors <- edx_train %>%
  mutate(across(where(is.ordered), as.numeric)) %>%
  select(where(is.numeric))

edx_train_num_cor_pearson <-
  cor(edx_train_num_predictors, method = 'pearson')
edx_train_num_cor_spearman <-
  cor(edx_train_num_predictors, method = 'spearman')

edx_train_num_cor_pearson_pmat <-
  cor_pmat(edx_train_num_predictors, method = 'pearson')
edx_train_num_cor_spearman_pmat <-
  cor_pmat(edx_train_num_predictors, method = 'spearman')

# Plot Pearson
edx_train_num_cor_pearson_plot <- ggcorrplot(
  edx_train_num_cor_pearson,
  type = 'lower',
  ggtheme = ggplot2::theme_bw,
  title = 'MovieLens Numeric Variable Pearson Correlations',
  hc.order = FALSE,
  lab = TRUE,
  p.mat = edx_train_num_cor_pearson_pmat
)

# Plot Spearman
edx_train_num_cor_spearman_plot <- ggcorrplot(
  edx_train_num_cor_spearman,
  type = 'lower',
  ggtheme = ggplot2::theme_bw,
  title = 'MovieLens Numeric Variable Spearman Correlations',
  hc.order = FALSE,
  lab = TRUE,
  p.mat = edx_train_num_cor_spearman_pmat
)

edx_train_num_cor_pearson_plot + edx_train_num_cor_spearman_plot

# Month and day of the year are perfectly correlated in both methods, remove one

# Film year of release and film age are very highly correlated,
# Given the exploration observed during film age the year of release year of release
# year of release should the the option to keep if selected by the caret function
# findCorrelation

edx_train_num_cor_pearson_remove <-
  findCorrelation(edx_train_num_cor_pearson,
                  verbose = TRUE,
                  names = TRUE)

edx_train_num_cor_spearman_remove <-
  findCorrelation(edx_train_num_cor_spearman,
                  verbose = TRUE,
                  names = TRUE)

edx_train_num_cor_remove_all <-
  unique(c(
    edx_train_num_cor_pearson_remove,
    edx_train_num_cor_spearman_remove
  )) %>%
  str_subset('(?i)film_(age|year_of_release)', negate = TRUE)

edx_train_num_cor_remove_all

edx_train <- edx_train %>%
  select(-all_of(edx_train_num_cor_remove_all))

# Of note are the features with some correlation to rating
# film year of release4, film age, user reviews and movie reviews have some
# direct correlation.
# While correlation is not causation these may warrant further exploration for
# Effects with other variables
# as it stands now the hypothesis is
# newer releases are slightly more strictly rated
# older films are slightly more leniently rated
# experienced reviewers are slightly more strict with their ratings
# popular films have slightly more leninet ratings

### Linear Dependencies ---------------------------------------------------

edx_train_num_predictors <- edx_train %>%
  mutate(across(where(is.ordered), as.numeric)) %>%
  select(where(is.numeric))

edx_train_num_lcombos <- findLinearCombos(edx_train_num_predictors)

edx_train_num_lcombos$remove

edx_train <- edx_train %>%
  select(-all_of(edx_train_num_lcombos$remove))

# There are no linear combinations of numeric variables

## Dimensionality Reduction ------------------------------------------------

# While not a Selection methodology Dimensinality reduction methods will be applied to the data
# These will be compared to the original features for selection

### PCA --------------------------------------------------------------------

# Principal Component Analysis will be performed in subsets

#### Date-Time subset -----------------------------------------------------

edx_train_date_time <- edx_train %>%
  select(all_of(
    c(
      'year',
      'month',
      'day',
      'day_of_the_quarter',
      'weekday',
      'hour',
      'minute',
      'second'
    )
  )) %>%
  mutate(across(where(is.ordered), as.numeric))

edx_train_date_time_pca_train <- preProcess(
  edx_train_date_time,
  method = c('YeoJohnson', 'scale', 'center', 'pca'),
  # Scale and Center
  thresh = 0.90 # Capture 90% of the variance
)

edx_train_date_time_pca_train

edx_train_date_time_pca <-
  as_tibble(predict(edx_train_date_time_pca_train, edx_train_date_time)) %>%
  rename_with(\(x) paste('date_time', x, sep = '_')) %>%
  clean_names()

#### Reviews Subset -------------------------------------------------------

edx_train_reviews <- edx_train %>%
  select(ends_with('_reviews'))

edx_train_reviews_pca_train <- preProcess(
  edx_train_reviews,
  method = c('YeoJohnson', 'scale', 'center', 'pca'),
  # Scale and Center
  thresh = 0.90 # Capture 90% of the variance
)

edx_train_reviews_pca_train

edx_train_reviews_pca <-
  as_tibble(predict(edx_train_reviews_pca_train, edx_train_reviews)) %>%
  rename_with(\(x) paste('reviews', x, sep = '_')) %>%
  clean_names()

edx_train_reviews_pca

## Data Transformations ----------------------------------------------------

# Variables that have not been used in PCA will be transformed, centered and scaled

edx_train_trans <- edx_train %>%
  select(-all_of(
    c(
      'rating',
      'year',
      'month',
      'day',
      'day_of_the_quarter',
      'weekday',
      'hour',
      'minute',
      'second',
      'user_reviews',
      'movie_reviews'
    )
  )) %>%
  select(where(is.numeric))

edx_train_trans

# Only film age is selected

edx_train_trans_train <- preProcess(edx_train_trans,
                                    method = c('YeoJohnson', 'scale', 'center'))

edx_train_trans_trans <-
  tibble(predict(edx_train_trans_train, edx_train_trans)) %>%
  rename_with(\(x) paste('trans_film_age', x, sep = '_')) %>%
  clean_names()


## Boruta -----------------------------------------------------------------

# Boruta will be used to asses relevant features and to compare PCA to non PCA
# features

### Original Data ---------------------------------------------------------

library(Boruta)

set.seed(1510, sample.kind = 'Rounding')
edx_train_boruta_og <- Boruta(
  rating ~ .,
  data = edx_train,
  doTrace = 3,
  getImp = getImpXgboost,
  maxRuns = 10000
)

edx_train_boruta_og

### Date-Time PCA ---------------------------------------------------------

set.seed(1851, sample.kind = 'Rounding')
edx_train_boruta_dt_pca <- Boruta(
  rating ~ .,
  data = select(bind_cols(edx_train, edx_train_date_time_pca),
                -all_of(
                  c(
                    'year',
                    'month',
                    'day',
                    'day_of_the_quarter',
                    'weekday',
                    'hour',
                    'minute',
                    'second'
                  )
                )),
  doTrace = 3,
  getImp = getImpXgboost,
  maxRuns = 10000
)

edx_train_boruta_dt_pca
plot(edx_train_boruta_dt_pca)

### Review PCA ------------------------------------------------------------

set.seed(1910, sample.kind = 'Rounding')
edx_train_boruta_revs_pca <- Boruta(
  rating ~ .,
  data = select(
    bind_cols(edx_train, edx_train_reviews_pca),
    -ends_with('_reviews')
  ),
  doTrace = 3,
  getImp = getImpXgboost,
  maxRuns = 10000
)

edx_train_boruta_revs_pca
plot(edx_train_boruta_revs_pca)

### Original Data Trans ---------------------------------------------------

library(Boruta)

set.seed(1919, sample.kind = 'Rounding')
edx_train_boruta_og_trans <- Boruta(
  rating ~ .,
  data = select(
    bind_cols(edx_train, edx_train_trans_trans),
    -all_of('film_age')
  ),
  doTrace = 3,
  getImp = getImpXgboost,
  maxRuns = 10000
)

edx_train_boruta_og_trans

### Date-Time PCA Trans ---------------------------------------------------

set.seed(1927, sample.kind = 'Rounding')
edx_train_boruta_dt_pca_trans <- Boruta(
  rating ~ .,
  data = select(
    bind_cols(edx_train, edx_train_date_time_pca, edx_train_trans_trans),
    -all_of(
      c(
        'year',
        'month',
        'day',
        'day_of_the_quarter',
        'weekday',
        'hour',
        'minute',
        'second',
        'film_age'
      )
    )
  ),
  doTrace = 3,
  getImp = getImpXgboost,
  maxRuns = 10000
)

edx_train_boruta_dt_pca_trans

### Review PCA Trans ------------------------------------------------------

set.seed(1934, sample.kind = 'Rounding')
edx_train_boruta_revs_pca_trans <- Boruta(
  rating ~ .,
  data = select(
    bind_cols(edx_train, edx_train_reviews_pca, edx_train_trans_trans),
    -c(ends_with('_reviews'), all_of('film_age'))
  ),
  doTrace = 3,
  getImp = getImpXgboost,
  maxRuns = 10000
)

edx_train_boruta_revs_pca_trans

### PCA & Transforms ------------------------------------------------------

set.seed(1934, sample.kind = 'Rounding')
edx_train_boruta_all_pca_trans <- Boruta(
  rating ~ .,
  data = bind_cols(
    edx_train,
    edx_train_date_time_pca,
    edx_train_reviews_pca,
    edx_train_trans_trans
  ) %>%
    select(-all_of(
      c(
        'year',
        'month',
        'day',
        'day_of_the_quarter',
        'weekday',
        'hour',
        'minute',
        'second',
        'film_age',
        'user_reviews',
        'movie_reviews'
      )
    )),
  doTrace = 3,
  getImp = getImpXgboost,
  maxRuns = 10000
)

edx_train_boruta_all_pca_trans

## Merge Importance Data --------------------------------------------------

edx_train_all_boruta_importance_tidy <- list(
  edx_train_boruta_og = edx_train_boruta_og$ImpHistory,
  edx_train_boruta_dt_pca = edx_train_boruta_dt_pca$ImpHistory,
  edx_train_boruta_revs_pca = edx_train_boruta_revs_pca$ImpHistory,
  edx_train_boruta_og_trans = edx_train_boruta_og_trans$ImpHistory,
  edx_train_boruta_dt_pca_trans = edx_train_boruta_dt_pca_trans$ImpHistory,
  edx_train_boruta_revs_pca_trans = edx_train_boruta_revs_pca_trans$ImpHistory,
  edx_train_boruta_all_pca_trans = edx_train_boruta_all_pca_trans$ImpHistory
) %>%
  map(as_tibble) %>%
  map(\(x) pivot_longer(
    x,
    cols = everything(),
    names_to = 'variable',
    values_to = 'importance'
  )) %>%
  bind_rows(.id = 'type') %>%
  group_by(type) %>%
  mutate(variable = fct_reorder(variable, importance)) %>%
  ungroup()

edx_train_all_boruta_importance_tidy %>%
  mutate(variable_level = as.numeric(variable)) %>%
  group_by(type) %>%
  filter(str_detect(variable, '(?i)shadow')) %>%
  slice_min(variable_level, with_ties = FALSE)

# All cutoffs are level 6

edx_train_all_boruta_importance_tidy %>%
  filter(as.numeric(variable) > 6) %>%
  group_by(type) %>%
  summarise(mean_importance = mean(importance)) %>%
  slice_max(mean_importance)

# "edx_train_boruta_all_pca_trans" is the favored data for modeling

getSelectedAttributes(edx_train_boruta_all_pca_trans)

# all 3 pre-processing models are to be saved as the selected features will
# require all pre-processing models

edx_train_date_time_pca_train
edx_train_reviews_pca_train
edx_train_trans_train

# Save pre-processing models
saveRDS(
  edx_train_date_time_pca_train,
  file = file.path('Data', 'edx_train_date_time_pca_train.rds')
)
saveRDS(edx_train_reviews_pca_train,
        file = file.path('Data', 'edx_train_reviews_pca_train.rds'))
saveRDS(edx_train_trans_train,
        file = file.path('Data', 'edx_train_trans_train.rds'))

# Prepare Training set for training and Save
edx_train <- bind_cols(edx_train,
                       edx_train_date_time_pca,
                       edx_train_reviews_pca,
                       edx_train_trans_trans) %>%
  select(-all_of(
    c(
      'year',
      'month',
      'day',
      'day_of_the_quarter',
      'weekday',
      'hour',
      'minute',
      'second',
      'film_age',
      'user_reviews',
      'movie_reviews'
    )
  ))

saveRDS(edx_train, file = file.path('Data', 'edx_train.rds'))

clear_memory(keep = 'edx_train')

# Train Model -------------------------------------------------------------

