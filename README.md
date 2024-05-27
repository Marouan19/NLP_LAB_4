# NLP Lab 4

This repository contains code for various Natural Language Processing (NLP) labs and experiments, including:

1. **Web Scraping and Text Classification**
   - Web scraping using Selenium
   - Data preprocessing (removing HTML tags, URLs, punctuation, tokenization, etc.)
   - Building machine learning models for text classification and regression tasks
   - Models: Recurrent Neural Network (RNN), Bidirectional RNN (Bi-RNN) with LSTM and GRU

2. **News Article Generation using GPT-2**
   - Fine-tuning a pre-trained GPT-2 transformer model on a news article dataset
   - Generating new news articles using the fine-tuned model

3. **Sentiment Analysis using BERT**
   - Building a sentiment analysis model using the BERT architecture
   - Classifying product reviews as positive or negative
   - Data preprocessing and tokenization for BERT
   - Model training, evaluation, and inference

# NLP LAB 4 - Part 1

## Introduction

This code is part of an NLP (Natural Language Processing) lab focused on web scraping, data preprocessing, and building machine learning models for text classification or regression tasks. The primary goal is to extract article content from a website, preprocess the text data, and train machine learning models to predict a score based on the presence of specific keywords.

## Dependencies

The code requires the following libraries:

- `selenium`: For web scrapping using Selenium.
- `webdriver_manager`: To manage the Chrome WebDriver automatically.
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib`: For data visualization.
- `seaborn`: For statistical data visualization.
- `nltk`: For natural language processing tasks.
- `gensim`: For Word2Vec model.
- `tensorflow`: For building and training machine learning models.

## Web Scraping
[Video Scraping]()
The code uses Selenium to scrape article titles, content URLs, and article content from the website `https://www.hespress.com`. The extracted data is stored in a CSV file named `data_scraping.csv`.

## Scoring Articles

The code defines a dictionary of keywords and their corresponding weights. It calculates a score for each article's content based on the presence of these keywords. The scores are then added as a new column to the CSV file, and a new file named `articles.csv` is created with the updated data.

## Data Preprocessing

The code provides several functions to preprocess the text data, including:

- Removing HTML tags
- Removing URLs
- Removing punctuation
- Converting text to lowercase
- Tokenizing text
- Removing stop words
- Stemming or lemmatization

The preprocessing steps are applied to the 'Content' column of the `articles.csv` file.

## Data Visualization

The code includes some visualizations to explore the dataset, such as:

- Displaying the first few rows of the dataset
- Checking for missing values and duplicates
- Plotting the distribution of scores

## Building Machine Learning Models

The code demonstrates how to build and train various machine learning models for text classification or regression tasks. It includes the following models:

1. **Recurrent Neural Network (RNN)**
  - The code tokenizes the text data, pads the sequences, and splits the data into training and testing sets.
  - It defines a function to build an RNN model using the `keras` library.
  - The model is trained and evaluated using different optimizers (Adam, SGD, and RMSprop), and the results are printed.

2. **Bidirectional Recurrent Neural Network (Bi-RNN) with LSTM and GRU**
  - The code applies Word2Vec to obtain word vectors for the text data.
  - It defines functions to create Bidirectional RNN models with LSTM and GRU layers.
  - The models are trained, and their performance is evaluated on the test set.
  - The training history is plotted for both models.

3. **Predicting Score for New Text**
  - The code demonstrates how to preprocess and predict the score for a new text input using the trained model.

## Usage

To run the code, you'll need to have all the required libraries installed. You can install them using `pip`:
```bash
pip install selenium webdriver_manager pandas numpy matplotlib seaborn nltk gensim tensorflow keras scikit-learn
```
Additionally, you'll need to download the necessary NLTK data by running the following code:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```
After installing the dependencies and downloading the required NLTK data, you can run the code in your Python environment or IDE.
Note: The code assumes that the website https://www.hespress.com is accessible and that the HTML structure and class names are consistent with the code. If the website changes, you may need to modify the web scraping part of the code accordingly.



# NLP LAB 4 - Part 2: Transformer for News Generation

## Introduction

This part of the code focuses on using the GPT-2 transformer model for generating news articles. The goal is to fine-tune a pre-trained GPT-2 model on a dataset of news articles and then use the trained model to generate new news articles.

## Dependencies

The code requires the following libraries:

- `pandas`: For data manipulation and analysis.
- `torch`: For tensor operations and deep learning.
- `transformers`: For accessing pre-trained transformer models like GPT-2.
- `numpy`: For numerical operations.

## Data Preparation

The code assumes the availability of a CSV file containing news articles. The `NewsDataset` class is defined to load and preprocess the news data from the CSV file. The class expects the CSV file to have columns for headlines and text content.

## Model Setup

1. The code loads the pre-trained `gpt2-medium` tokenizer and model from the Hugging Face Transformers library.
2. The `choose_from_top` function is defined to sample from the top-k tokens during text generation.
3. The device (CPU or GPU) is determined based on the availability of a CUDA-enabled GPU.

## Model Training

1. The news data is loaded into a PyTorch `DataLoader` for efficient batch processing.
2. The GPT-2 model is set to training mode, and an optimizer (`AdamW`) and learning rate scheduler are initialized.
3. The training loop iterates over the specified number of epochs (`EPOCHS`).
4. Within each epoch, the news data is processed in batches of a specified size (`BATCH_SIZE`).
5. The news sequences are concatenated to fit within the maximum sequence length (`MAX_SEQ_LEN`).
6. The model is trained on the concatenated sequences using the cross-entropy loss.
7. After each epoch, the trained model's state is saved to the `trained_models` directory for future use.

## Model Evaluation

1. The code lists the trained model files in the `trained_models` directory.
2. A specific model epoch (e.g., `MODEL_EPOCH = 4`) is selected for evaluation.
3. The tokenizer and pre-trained GPT-2 model are loaded, and the selected model's state is loaded from the corresponding file.
4. The model is set to evaluation mode and moved to the appropriate device (CPU or GPU).
5. The `generate` method of the GPT-2 model is used to generate news articles, with various parameters controlling the generation process (e.g., temperature, top-k, top-p, repetition_penalty).
6. The generated news articles are decoded using the tokenizer and written to a text file (`generated_news_{MODEL_EPOCH}.txt`).

## Execution

1. Install the required libraries using `pip install pandas torch transformers numpy`.
2. Run the code cells in order.
3. The output will show the training progress, including the loss values for each epoch.
4. After training, the list of trained model files will be displayed.
5. The selected model epoch will be used to generate news articles, and the generated articles will be saved to a text file.

Note: The execution may take some time, depending on the dataset size, model complexity, and available computational resources.

# NLP LAB 4 - Part 3: Sentiment Analysis using BERT

## Introduction
This part of the code focuses on building a sentiment analysis model using the BERT (Bidirectional Encoder Representations from Transformers) architecture for sequence classification. The task is to classify product reviews as either positive or negative based on the review text.

## Data Preparation
1. The code starts by importing the necessary libraries, including PyTorch, Transformers, Pandas, and JSON.
2. The device (CPU or GPU) is determined based on the availability of a CUDA-enabled GPU.
3. The Amazon Fashion product review dataset is loaded from a JSON file into a Pandas DataFrame.
4. The DataFrame is filtered to include only rows where the `reviewText` column contains strings.
5. The number of missing reviews and non-string reviews are checked and printed.

## Data Preprocessing
1. The BERT tokenizer is initialized using `BertTokenizer.from_pretrained('bert-base-uncased')`.
2. A function `tokenize_texts` is defined to tokenize the review texts using the BERT tokenizer. It handles padding and truncation of the texts to a maximum length of 128 tokens.
3. The `reviewText` column of the DataFrame is tokenized using the `tokenize_texts` function, and the corresponding input IDs and attention masks are obtained.
4. The overall rating (1-5) is converted to binary labels (0 for ratings < 4, 1 for ratings >= 4) using the `apply` method and stored in the `labels` tensor.

## Model Setup
1. The BERT model for sequence classification (`BertForSequenceClassification`) is loaded from the pre-trained `bert-base-uncased` checkpoint.
2. The model is moved to the appropriate device (CPU or GPU).
3. A TensorDataset is created using the input IDs, attention masks, and labels.
4. The dataset is split into training and validation sets (80% for training, 20% for validation) using `random_split`.
5. DataLoaders are created for the training and validation datasets with a batch size of 32.

## Model Training
1. The optimizer (AdamW) and learning rate scheduler (StepLR) are initialized.
2. The cross-entropy loss function is defined.
3. The training loop iterates over the specified number of epochs (3 in this case).
4. In each epoch, the model is trained on the training dataset, and the average training loss is calculated.
5. After each training epoch, the model is evaluated on the validation dataset, and the average validation loss and accuracy are calculated and printed.

## Model Evaluation
1. After training, the model is evaluated on the validation dataset.
2. The predictions and true labels are collected, and the accuracy and F1 scores are calculated using the `accuracy_score` and `f1_score` functions from `sklearn.metrics`.
3. The final accuracy and F1 score are printed.

## Inference
1. Example review texts are provided for inference.
2. The example reviews are tokenized using the `tokenize_texts` function.
3. The tokenized reviews are moved to the appropriate device.
4. The model is set to evaluation mode, and predictions are made on the example reviews using `model.forward`.
5. The predictions are decoded as "Positive" or "Negative" based on the logit scores.
6. The review texts and their corresponding sentiment predictions are displayed.

## Results
The code outputs the training and validation losses for each epoch, as well as the final accuracy and F1 score on the validation set. For the provided example, the model achieved an accuracy of 0.9953 and an F1 score of 0.9972, indicating good performance on the sentiment analysis task.

The sentiment predictions for the example reviews are also printed, demonstrating the model's ability to classify positive and negative reviews accurately.

Note: The performance of the model may vary depending on the dataset, hyperparameters, and the available computational resources.
