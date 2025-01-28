# AI_AC4a_Fall2023

### **Embeddings.py**
- **Purpose**: Reads CSV files containing articles and generates embeddings for them using ChromaDB.
- **Key Functionality**:
  - Converts articles into vector representations.
  - Stores the embeddings in a ChromaDB vector database.

### **PIRvsNIR.py**
- **Purpose**: Classifies articles based on their similarity to pre-defined PIR (Positive Intergroup Relations) and NIR (Negative Intergroup Relations) embeddings.
- **Key Functionality**:
  - Takes input articles and computes similarity with PIR and NIR embeddings.
  - Outputs classification results based on which embedding the article is closer to.

### **Queries.py**
- **Purpose**: Classifies articles by finding the most similar article from the vector database.
- **Key Functionality**:
  - Retrieves the most similar article from the embedded database.
  - Uses the metadata of the most similar article to classify the input article.

### **QueryBigData.py**
- **Purpose**: Evaluates the accuracy of the classification method that uses the most similar article from the vector database.
- **Key Functionality**:
  - Tests the performance of the most similar article classification method on datasets of various sample sizes.
  - Provides accuracy metrics for comparison.

### **RegressionModel.py**
- **Purpose**: Implements a Logistic Regression (LR) classifier to classify articles based on their embeddings.
- **Key Functionality**:
  - Trains an LR model on labeled embeddings.
  - Uses the trained model to classify new articles based on their embeddings.

### **Tokens.py**
- **Purpose**: Counts the number of tokens in a text file.
- **Key Functionality**:
  - Processes text files and outputs the total token count.
  - Useful for analyzing the size and complexity of text datasets.

AI Project: AC4a: Larry S. Liebovitch, Peter T. Coleman, Melissa Mannis, Kevin Lian

