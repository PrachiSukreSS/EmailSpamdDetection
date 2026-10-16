# Email Spam Detection Using Machine Learning

A machine-learning project that classifies text messages as **spam** or **ham** (legitimate messages). The project uses a Jupyter Notebook to clean and explore a labeled SMS dataset, convert message text into token-count features, train a Multinomial Naive Bayes classifier, and evaluate its predictions.

## Results

Using the workflow in `classification_of_mail.ipynb`:

- **Training accuracy:** `99.33%`
- **Test accuracy:** `98.74%`
- **Duplicate rows identified:** `403`
- **Example prediction:** `Same. Wana plan a trip sometme then` is classified as ham.

> These results are the outputs saved in the notebook. They should be revalidated after correcting the dataset path and rerunning the notebook from a clean environment.

## Project structure

```text
.
├── classification_of_mail.ipynb          # Data preparation, training, prediction, and evaluation
├── dataset for email classification.csv  # Labeled message dataset
├── outline image.png                     # Project outline graphic
└── All-About-Email-Spam-Detection        # Background notes on spam and ham messages
```

## Workflow

The notebook follows these steps:

1. Load the labeled CSV dataset with pandas.
2. Remove unused columns (`Unnamed: 2`, `Unnamed: 3`, and `Unnamed: 4`).
3. Rename the source columns to `Spam or Ham` and `Mail Message`.
4. Encode the `ham`/`spam` labels using `LabelEncoder`.
5. Separate message text (`X`) from the target label (`y`).
6. Split the data into training and test sets using an 80/20 split with `random_state=0`.
7. Convert messages into token-count vectors with `CountVectorizer`.
8. Train a `MultinomialNB` classifier.
9. Predict labels for new messages and measure model accuracy.

## Technology stack

- Python 3
- Jupyter Notebook or Google Colab
- pandas
- NumPy
- scikit-learn
  - `LabelEncoder`
  - `train_test_split`
  - `CountVectorizer`
  - `MultinomialNB`
  - classification metrics

## Getting started

### Option 1: Run in Google Colab

1. Open [`classification_of_mail.ipynb`](classification_of_mail.ipynb) in Google Colab.
2. Upload `dataset for email classification.csv` to the Colab session.
3. Update the `pd.read_csv(...)` path in the notebook to match the uploaded filename.
4. Run the cells from top to bottom.

### Option 2: Run locally

Create a virtual environment and install the required packages:

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn jupyter
jupyter notebook classification_of_mail.ipynb
```

When running locally, make sure the CSV path in the notebook points to:

```text
dataset for email classification.csv
```

## Dataset

The CSV contains labeled text messages with the following meaningful fields:

| Source column | Meaning |
| --- | --- |
| `v1` | Message label: `ham` or `spam` |
| `v2` | Message text |

The repository also includes three empty/unused columns in the raw CSV. The notebook removes them before modeling.

The messages are primarily SMS-style examples and include promotional offers, prize notifications, subscription messages, personal conversations, and other legitimate text. The dataset contains some encoding artifacts, so preprocessing and normalization could be improved for production use.

## Using the trained model

After the notebook has trained `model` and `cv`, classify a new message as follows:

```python
message = ["Congratulations! You have won a free prize. Call now!"]
message_features = cv.transform(message)
prediction = model.predict(message_features)

label = "spam" if prediction[0] == 1 else "ham"
print(label)
```

The exact numeric mapping is learned by `LabelEncoder`; inspect `encoder.classes_` rather than assuming a numeric label in a modified dataset.

## Limitations and recommended improvements

- The notebook currently reads `/content/spam.csv`, while the repository dataset is named `dataset for email classification.csv`. Update the path before running.
- The notebook link metadata references a different GitHub repository and should be updated if a Colab badge is added.
- The notebook reports accuracy but imports precision, recall, and confusion-matrix metrics without displaying their values. Add these metrics, especially because spam datasets can be imbalanced.
- Duplicate messages are detected but not removed before splitting the data. Deduplicating first can make evaluation more realistic.
- Use a scikit-learn `Pipeline` to keep vectorization and classification together and reduce data-leakage risk.
- Consider `TfidfVectorizer`, stratified splitting, cross-validation, and hyperparameter tuning.
- Normalize character encoding and add text preprocessing for URLs, phone numbers, currency symbols, and unusual Unicode characters.
- The project is an analysis notebook rather than a deployed service; it does not currently expose an API, web interface, or serialized model.

## Reproducibility notes

The notebook uses `random_state=0` for the train/test split and records the following saved outputs:

- Training accuracy: `0.9932690150325331`
- Test accuracy: `0.9874439461883409`

For a reproducible rerun, start with a fresh kernel, use the repository CSV file, execute all cells in order, and record the package versions used.

## License

No license file is currently included in the repository. Add a license before distributing or reusing the project.
