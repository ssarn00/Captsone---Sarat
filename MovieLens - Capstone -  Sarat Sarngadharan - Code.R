## Sarat Sarngadharan
## Capstone HarvardX

################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Summary of the edx dataset
summary(edx)
#Rows / Columns in Dataset
dim(edx)
#Ratings
edx %>% separate_rows(rating, sep = "\\|") %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

#movies / userid

n_distinct(edx$movieId)

# distinct Userid

n_distinct(edx$userId)


# Genres

edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) 

#plot Genre
edx %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = genres, y = count)) +
  geom_line()+
  ggtitle("genres Plot")

# Highest Rated Movies

edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))


# Ratings descending
edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(5) %>%
  arrange(desc(count))  


#Ratings plot
edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()+
  ggtitle("Ratings Plot")

#mean rating
mean(edx$rating)

edx %>% group_by(userId) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = userId, y = count)) +
  geom_point(color="blue")+
  ggtitle("User Ratings Plot")

#Analysis
# Loss Function - RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

## Simple Prediction based on Mean Rating
mu <- mean(edx$rating)
mu

rmse_naive <- RMSE(validation$rating, mu)
rmse_naive

## Save Results in Data Frame
rmse_results = data_frame(method = "Naive Analysis by Mean", RMSE = rmse_naive)

## Movie Effects Model
## Simple model taking into account the movie effects, b_i

mu <- mean(edx$rating)
mu

movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu))

predicted_ratings <- mu + validation %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

rmse_model_movie_effects <- RMSE(predicted_ratings, validation$rating)
rmse_model_movie_effects
rmse_results <- bind_rows(rmse_results, 
                          data_frame(method="Movie Effects Model",
                                     RMSE = rmse_model_movie_effects))

## User Effects Model
## Simple model taking into account the user effects, b_u
user_avgs <- edx %>%
  left_join(movie_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

rmse_model_user_effects <- RMSE(predicted_ratings, validation$rating)
rmse_model_user_effects
rmse_results <- bind_rows(rmse_results, 
                          data_frame(method="User Effects Model",
                                     RMSE = rmse_model_user_effects))

## Regularisation
# Predict via regularisation, movie and user effect model
# (as per https://rafalab.github.io/dsbook 34.9 Regularization)

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n() +l))
  
  b_u <- edx %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- validation %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
  
})
rmse_regularisation <- min(rmses)
rmse_regularisation

# Plot RMSE against Lambdas to find optimal lambda
qplot(lambdas, rmses)
lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <- bind_rows(rmse_results, 
                          data_frame(method="Regularisation",
                                     RMSE = rmse_model_user_effects))

## Final results
rmse_results

## Appendix
# For future debugging purposes
print("Version Info")
version

