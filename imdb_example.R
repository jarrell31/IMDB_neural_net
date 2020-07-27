setwd("C:/Users/Owner/Desktop/The 9 Projects/Number Identification")

#install.packages("keras")
library(keras)
install_keras(tensorflow = "gpu")

#load IMDB dataset
imdb <- dataset_imdb(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

#decoding
word_index <- dataset_imdb_word_index()
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index
decoded_review <- sapply(train_data[[1]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})

#preparing data
vectorize_sequences <- function(sequences, dimension = 10000) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

xtrain <- vectorize_sequences(train_data)
xtest <- vectorize_sequences(test_data)

ytrain <- as.numeric(train_labels)
ytest <- as.numeric(test_labels)

#building model
model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

#compile model
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

#setting aside validation set
val_indices <- 1:10000
x_val <- xtrain[val_indices,]
partial_x_train <- xtrain[-val_indices,]
y_val <- ytrain[val_indices]
partial_y_train <- ytrain[-val_indices]

#training model
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

plot(history)

#remake model from scratch with only 4 epochs
model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

model %>% fit(xtrain, ytrain, epochs = 4, batch_size = 512)
results <- model %>% evaluate(xtest, ytest)

model %>% predict(xtest[1:10, ])
