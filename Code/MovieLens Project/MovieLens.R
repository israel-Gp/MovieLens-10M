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

saveRDS(edx, file = file.path('Data', 'edx.rds'))
saveRDS(final_holdout_test, file = file.path('Data', 'final_holdout_test.rds'))

# Data Load from Archives -----------------------------------------------------------

edx <- readRDS('~/Data Projects/MovieLens-10M/Data/edx.rds')
final_holdout_test <-
  readRDS('~/Data Projects/MovieLens-10M/Data/final_holdout_test.rds')

# EDX Set as tibble -------------------------------------------------------

library(tidyverse)
library(caret)
library(janitor)
# Convert Data Frame to Tibble
edx <- as_tibble(edx) %>%
  clean_names()

# For ease of memory management a chunk and pull strategy will be used

gc(reset = T)

## Set Plot Theme & Pallets ----------------------------------------------------------

theme_set(theme_bw())

## Define color palette for month factors ----------------------------------

# Datetime features will contain months so a custom palette will be prepared

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
      '#7d4400'  
      # December
    )
  )
  
}

# Data Partitioning -------------------------------------------------------

# Partition using an 80/20 split
# TO ensure that categorical features that are extracted are consistent
# users and film should be included in both the training and test set
# nested features and a custom map partition function is an option but
# partitioning followed by semi_join and anti_join is a more efficient option

# Set Split Index
set.seed(2213, sample.kind = 'Rounding')
train_index <-
  createDataPartition(edx$rating,
                      times = 1,
                      p = 0.8,
                      list = FALSE)

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

# Remove aid objects to save memeory and clean memory
rm(list = c('edx_removed', 'edx', 'train_index', 'edx_test_alpha'))

gc(reset = T)

# Data Cleaning -----------------------------------------------------------

slice_sample(edx_train, n = 10)

# Clean Time stamp to date-time
# Separate Title from year of release
# Separate genres into constituent parts, preserving the order in which
# they are pipe separated
# genres are not consistent across observations
# use "none" for missing genres
# Genre Levels should be consistent across all analysis
# a prior level determination is required
# title should be tied to movie id and as the source details it is prone to errors

genre_levels <- edx_train %>%
  separate_longer_delim(genres, delim = '|') %>%
  count(genres) %>%
  arrange(genres) %>%
  add_row(genres = 'none') %>%
  pull(genres)

library(lubridate)

edx_train <- edx_train %>%
  mutate(timestamp = as_datetime(timestamp)) %>%
  separate_wider_regex(
    title,
    patterns = c(title = '[:print:]+(?=(?:\\([:digit:]{4}\\)))',
                 year_of_release = '.*')
  ) %>%
  mutate(across(c('title', 'year_of_release'), str_squish)) %>%
  mutate(year_of_release = as.numeric(str_extract(year_of_release, '[:digit:]{4}'))) %>%
  separate_wider_delim(genres,
                       delim = '|',
                       names_sep = '_',
                       too_few = 'align_start') %>%
  mutate(across(starts_with('genres'), \(x) replace_na(x, 'none'))) %>%
  mutate(across(starts_with('genres'), \(x) factor(x, levels = genre_levels))) %>%
  mutate(across(ends_with('_id'), \(x) factor(x, levels = sort(unique(
    x
  ))))) %>% 
  select(-all_of(c('title')))

# Data Exploration --------------------------------------------------------

## Summary Statistics for Numeric Variables --------------------------------

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

edx_train_stats

# The mean rating is 3.51
# The mean year of release is 1990

gc(reset = T)

## Rating Distribution -----------------------------------------------------

# edx_train %>%
#   ggplot(aes(rating)) +
#   geom_density(fill = '#FF0000', alpha = 0.5) +
#   geom_vline(
#     data = filter(edx_train_stats, variable == 'rating'),
#     aes(xintercept = estimate),
#     color = '#000000',
#     linetype = 'dashed'
#   ) +
#   geom_label(
#     data = filter(edx_train_stats, variable == 'rating'),
#     aes(
#       x = estimate,
#       y = -0.15,
#       label = paste0('Mean\n', scales::label_comma(accuracy = 0.01)(estimate))
#     )
#   ) +
#   scale_x_continuous(
#     'Rating',
#     labels = unique(edx_train$rating),
#     breaks = unique(edx_train$rating)
#   ) +
#   guides(color = 'none') +
#   ylab('Rating Density') +
#   ggtitle('Film Rating Distribution')

# Data tends towards the mean but is mostly concentrated in ratings 3 & 4
# This may be an issue with distribution through time
# Explore rating distribution by year, chosen as it is the largest slice of time

gc(reset = T)

## Rating Distribution by year ---------------------------------------------

library(ggridges)

# edx_train %>%
#   mutate(year = year(timestamp)) %>%
#   ggplot(aes(rating, as.factor(year))) +
#   stat_density_ridges(
#     geom = 'density_ridges',
#     calc_ecdf = TRUE,
#     quantiles = 2,
#     quantile_lines = TRUE,
#     fill = '#FF0000',
#     alpha = 0.5
#   ) +
#   geom_vline(
#     data = filter(edx_train_stats, variable == 'rating'),
#     aes(xintercept = estimate, color = '#000000'),
#     linetype = 'dashed'
#   ) +
#   scale_x_continuous(
#     'Rating',
#     labels = unique(edx_train$rating),
#     breaks = unique(edx_train$rating)
#   ) +
#   scale_color_manual('Statistic',
#                      values = c('#000000'),
#                      labels = c('Overall\nMean')) +
#   ylab('Year') +
#   ggtitle('Film Rating Distribution by Review Year')

gc(reset = T)

# Rating distributions and means are affected by the year the review was made
# Years before 2003 have little to none half-star reviews
# The option may not have been available till 2003
# This may result in more variation in means with finer time slices
# Reviews in 1995 are near zero compared to other years

## Rating Means by Month and Date ------------------------------------------

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

# edx_train_daily_mean_ratings %>%
#   ggplot(aes(date, estimate)) +
#   geom_line(aes(color = month, group = 1)) +
#   geom_smooth(
#     se = TRUE,
#     span = 365.256366 / 12,
#     color = '#FF0000',
#     aes(group = interaction(year, month),
#         linetype = 'smooth')
#   ) +
#   scale_color_movielens_months() +
#   scale_linetype_manual(
#     'Statistics',
#     values = c('dashed'),
#     labels = c('Monthly\nMoving\nAverage Rating')
#   ) +
#   scale_x_continuous(
#     'Date',
#     labels = seq(
#       min(edx_train_daily_mean_ratings$year),
#       max(edx_train_daily_mean_ratings$year)
#     ),
#     breaks = seq.Date(
#       floor_date(min(edx_train_daily_mean_ratings$date), unit = 'month'),
#       ceiling_date(max(edx_train_daily_mean_ratings$date), unit = 'month'),
#       by = 'year'
#     )
#   ) +
#   guides(
#          color = guide_legend('Month')
#          ) +
#   ylab('Mean Rating') +
#   ggtitle(
#           'Daily Mean Ratings'
#           ) +
#   ylim(
#     min(edx_train_daily_mean_ratings$estimate),
#     max(edx_train_daily_mean_ratings$estimate)
#   )

gc(reset = T)

# Daily mean ratings are volatile before the `year` $2000$.
# There is a potential lack of reviews between $1995$ and $1996$.
# `year` potentially could be grouped in 5 categories
# - 1995
# - Early 1996
# - Late 1996 to Late 1997
# - Late 1997 to Late 1999
# - Post Late 1999
# Both `month` and `year` appear to have some effect on the influence of the rating.
# Spikes are consistently seen in both the Summer Months and Late November for Post Late 1999 periods.
# Some days have prominent spikes, these may not me tied to days of the month necessarily
# While holidays and weekends can produce these effects the release of a film may be a better indicator
# The feature closest to track this may be film age, while the actual release date is excluded
# the holdout dates CANNOT be accounted for and there is no garantee that a user rated a film close to the release date
# this can be engineered from review year and the year of release as a proxy for all of these features

## Genre Levels Rating Density ------------------------------------------------------------

edx_train_genre_density <- edx_train %>%
  pivot_longer(
    cols = starts_with('genres'),
    names_to = 'genre_level',
    values_to = 'genre',
    names_transform = list(genre_level = ~ as.numeric(str_extract(.x, '[:digit:]+$')))
  ) %>%
  group_by(genre_level, genre) %>%
  summarise(t.test = t_test(rating)) %>%
  unnest(cols = starts_with('t.test'))

# edx_train_genre_density %>%
#   mutate(genre = fct_reorder2(genre, genre_level, estimate, .desc = FALSE)) %>%
#   ggplot(aes(genre_level, genre)) +
#   geom_raster(aes(fill = estimate)) +
#   scale_fill_continuous('Rating', high = '#FF51EB', low = '#FFB1F6') +
#   scale_x_continuous(
#     'Genre Level',
#     labels = seq(
#       range(edx_train_genre_density$genre_level)[1],
#       range(edx_train_genre_density$genre_level)[2]
#     ),
#     breaks = seq(
#       range(edx_train_genre_density$genre_level)[1],
#       range(edx_train_genre_density$genre_level)[2]
#     )
#   ) +
#   ylab('Genre') +
#   ggtitle('Rating Density by Film Genre & Genre Level')

# Genres after level 3 become quite sparse
# Genres tend towards mean rating with the exception of a few genres at certain levels
# aka some genres have higher ratings at particular levels
# - IMAX at Level 4, Comedy at Level 5, etc.

gc(reset = T)

# Observation Counts by Genre & Level -------------------------------------

edx_train_genre_counts <- edx_train %>%
  pivot_longer(
    cols = starts_with('genres'),
    names_to = 'genre_level',
    values_to = 'genre',
    names_transform = list(genre_level = ~ as.numeric(str_extract(.x, '[:digit:]+$')))
  ) %>%
  count(genre_level, genre, sort = TRUE)

# edx_train_genre_counts %>%
#   filter(genre != 'none') %>%
#   mutate(genre = fct_reorder2(genre, genre_level, n, .desc = FALSE)) %>%
#   ggplot(aes(genre_level, genre)) +
#   geom_raster(aes(fill = n)) +
#   scale_fill_continuous(
#     'Observations',
#     high = '#FF51EB',
#     low = '#FFB1F6',
#     labels = scales::label_comma()
#   ) +
#   scale_x_continuous(
#     'Genre Level',
#     labels = seq(
#       range(edx_train_genre_counts$genre_level)[1],
#       range(edx_train_genre_counts$genre_level)[2]
#     ),
#     breaks = seq(
#       range(edx_train_genre_counts$genre_level)[1],
#       range(edx_train_genre_counts$genre_level)[2]
#     )
#   ) +
#   ylab('Genre') +
#   ggtitle('Counts Density by Film Genre & Genre Level')

gc(reset = T)

# There is a large discrepancy on review counts between genres and levels,
# Action, Drama and Comedy at Level 1 being far more prominent genres across all levels
# Passed level 3 the sparsity of observations becomes prominent hitting a major 
# drop at level 5

# Feature Engineering -----------------------------------------------------

edx_train <- edx_train %>% 
  mutate(
    year = year(timestamp),
    month = month(timestamp, label = TRUE, abbr = FALSE),
    day = day(timestamp),
    weekday = wday(timestamp, label = TRUE, abbr = FALSE),
    film_age = year - year_of_release
  ) %>% 
  select(-all_of(c('timestamp',str_c('genres_',6:8)))) %>% 
  relocate('rating') %>% 
  relocate(starts_with('genres_'),.after = 'weekday')

gc(reset = T)

# Feature Selection -------------------------------------------------------


## Remove Near Zero Variance -----------------------------------------------

edx_nzv <- nearZeroVar(edx_train[,-c(1:3)], saveMetrics = TRUE, names = TRUE)

edx_train <- edx_train %>% 
  select(-(which(edx_nzv$nzv) + 3))

gc(reset = T)

## Correlated Predictors ---------------------------------------------------

library(ggcorrplot)

edx_cor <- edx_train %>% 
  select(-c(contains('_id'),starts_with('genres_'))) %>% 
  mutate(across(where(is.ordered),as.numeric)) %>% 
  cor(method = 'spearman')

edx_cor_pmat <- edx_train %>% 
  select(-c(contains('_id'),starts_with('genres_'))) %>% 
  mutate(across(where(is.ordered),as.numeric)) %>% 
  cor_pmat(method = 'spearman')

# ggcorrplot(
#            edx_cor,
#            ggtheme = ggplot2::theme_bw,
#            type = 'lower',
#            title = 'Rating & Numeric Predictor Correlations',
#            hc.order = TRUE,
#            lab = TRUE,
#            p.mat = edx_cor_pmat
#            )

gc(reset = T)

# There are no strong correlations between numeric or ordered categorical features
# As expected there are strong correlations for film age and its original features
# No features removed at this stage

## Linear Dependencies -----------------------------------------------------

edx_lcombo <- edx_train %>% 
  select(-c(contains('_id'),starts_with('genres_'))) %>% 
  mutate(across(where(is.ordered),as.numeric)) %>% 
  as.matrix() %>% 
  findLinearCombos()

edx_train %>% 
  select(-c(contains('_id'),starts_with('genres_'))) %>% 
  names() %>% 
  .[edx_lcombo$linearCombos[[1]]]

gc(reset = T)

# The only linear dependent variables are those related to film age
# The suggestion is to remove film age, howver this will be applied in neccessary after further 
# analysis

### Boruta Wrapper ----------------------------------------------------------

# To be certain of feature interactions the Boruta Wrapper will be used to determine 
# feature importance
# for speed and resource optimization the XG Boost variation will be used
# User and film effects will be accounted for in separate analysis

library(Boruta)
library(ggrepel)

set.seed(104, sample.kind = 'Rounding')

edx_boruta <- Boruta(
  rating ~ .,
  data = edx_train,
  doTrace = 3,
  getImp = getImpXgboost,
  maxRuns = 10000
)

edx_boruta_data <- as_tibble(edx_boruta$ImpHistory) %>%
  pivot_longer(
    cols = everything(),
    names_to = 'variable',
    values_to = 'variable_importance',
    names_transform = list(variable = ~ as.factor(.x))
  ) %>%
  left_join(tibble(
    variable = names(edx_boruta$finalDecision),
    decision = as.character(edx_boruta$finalDecision)
  ),
  by = 'variable') %>%
  mutate(
    variable = fct_reorder(variable, variable_importance, .desc = FALSE),
    decision = as.factor(replace_na(decision, 'Metric'))
  )

edx_boruta_stats <- edx_boruta_data %>%
  group_by(variable) %>%
  summarise(median = median(variable_importance),
            decision = first(decision))

# edx_boruta_data %>%
#   ggplot(aes(variable, variable_importance, fill = decision)) +
#   geom_boxplot() +
#   geom_text_repel(
#     inherit.aes = FALSE,
#     data = filter(edx_boruta_stats, decision == 'Confirmed'),
#     aes(variable, median, label = scales::label_number()(median)),
#     nudge_x = 0.25,
#     nudge_y = 0.25
#   ) +
#   scale_y_log10() +
#   scale_fill_discrete('Decision') +
#   xlab('Variable') +
#   ylab('Variable Importance') +
#   coord_flip() +
#   ggtitle('Boruta Variable Importance', subtitle = 'Boruta Feature Importance wrapper variable importance values')

gc(reset = T)

# Boruta Analysis recommends removing Day, User ID & Weekday
# User Id will not be removed as it is a basis for user effects that may be utilized in models
# Removal will be expected at a per model basis

### Boruta Wrapper 2 ---------------------------------------------------------

set.seed(2243, sample.kind = 'Rounding')

edx_boruta_2_ <- Boruta(
  rating ~ .,
  data = edx_train,
  doTrace = 3,
  getImp = getImpXgboost,
  maxRuns = 10000
)

edx_boruta_2__data <- as_tibble(edx_boruta_2_$ImpHistory) %>%
  pivot_longer(
    cols = everything(),
    names_to = 'variable',
    values_to = 'variable_importance',
    names_transform = list(variable = ~ as.factor(.x))
  ) %>%
  left_join(tibble(
    variable = names(edx_boruta_2_$finalDecision),
    decision = as.character(edx_boruta_2_$finalDecision)
  ),
  by = 'variable') %>%
  mutate(
    variable = fct_reorder(variable, variable_importance, .desc = FALSE),
    decision = as.factor(replace_na(decision, 'Metric'))
  )

edx_boruta_2__stats <- edx_boruta_2__data %>%
  group_by(variable) %>%
  summarise(median = median(variable_importance),
            decision = first(decision))

# edx_boruta_2__data %>%
#   ggplot(aes(variable, variable_importance, fill = decision)) +
#   geom_boxplot() +
#   geom_text_repel(
#     inherit.aes = FALSE,
#     data = filter(edx_boruta_2__stats, decision == 'Confirmed'),
#     aes(variable, median, label = scales::label_number()(median)),
#     nudge_x = 0.25,
#     nudge_y = 0.25
#   ) +
#   scale_y_log10() +
#   scale_fill_discrete('Decision') +
#   xlab('Variable') +
#   ylab('Variable Importance') +
#   coord_flip() +
#   ggtitle('Boruta Variable Importance', subtitle = 'Boruta Feature Importance wrapper variable importance values')

gc(reset = T)

# Boruta Analysis continues to recommends removing Day, User ID & Weekday
# User Id will not be removed as it is a basis for user effects that may be utilized in models
# Removal will be done at a per model basis
# Film age is also a candidate for removal as it is by far less important that either 
# year of release or year alone in the overall analysis

quantile(edx_train$film_age)

# There are instances that produce negative numbers
# This should not be possible, this likely stems from the manual input
# This feature is top be explored further after boruta feature selection

## Boruta Clean ------------------------------------------------------------

edx_train <- edx_train %>%
  select(-all_of(str_subset(
    names(edx_boruta_2_$finalDecision)[which(edx_boruta_2_$finalDecision == "Rejected")], 'user_id', negate = TRUE
  ))) %>% 
  relocate(film_age, .after = 'year')

## Film Age ----------------------------------------------------------------

edx_train %>%
  filter(film_age < 0) %>%
  group_by(movie_id) %>%
  summarise(t.test = t_test(film_age)) %>%
  unnest(cols = starts_with('t.test')) %>% 
  count(method, sort = TRUE)

# Films with negative ages are all have the same values for age
# While this is not a real element it is however a feature that can be applied
# to tain an model

# Model Selection ---------------------------------------------------------


