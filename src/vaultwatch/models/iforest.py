import pandas as pd
import numpy as np
import joblib 
from sklearn.ensemble import IsolationForest
from pathlib import Path


#ISOLATION FOREST MODEL CLASS 
class IsolationForestModel:
    def __init__(self): # constructor method 
        self.forest = IsolationForest(n_estimators=100, random_state=42)


    def fit(self, X, y=None): #normal transactions to train model
        if y is not None: # if no fraud label are provided, model will be trained
            labels = np.asarray(y)  
            if labels.shape != (len(X),) or not np.isin(labels, [0, 1]).all(): #error check
                raise ValueError("Provide one binary fraud label per training row.")
            X = X[labels == 0] # only accept normal transctiosn for training, so filter out fraud rows

        if len(X) == 0: #need more than 0 rows
            raise ValueError("Training requires at least one normal transaction.")

        self.forest.fit(X) #trains on filted normal x
        return self

    def predict_score(self, features): #returns anomaly score
        return -self.forest.score_samples(features)

    def save(self, model_path): 
        joblib.dump(self.forest, model_path) # saves a Python object to disk so you can reload it later with .load()

    #need this decorator or python treats this as regular instance method, so it will get passed to cls only, leaving path empty
    @classmethod 
    def load(cls, path): #loads a model from disk and returns an instance of the class with the loaded model
        model = cls()
        model.forest = joblib.load(path)
        return model

def main():  #main driver function

    transactions = pd.read_parquet("data/seed/paysim_sample.parquet") 
    transactions["hour_of_day"] = transactions["step"] % 24 #step = hours in PaySim
    feature_columns = ["amount", "hour_of_day"]

    #filter table to keep matching rows for training and validation (less than 323 hours for training, 324-378 hours for validation)
    training_rows = transactions.loc[transactions["step"] <= 323] #keeps true rows
    training_labels = training_rows["isFraud"] #pulls 0 or 1 fraud label so it can pass as y to the fit method
    validation_rows = transactions[transactions["step"].between(324, 378)].copy()


    # Train the Isolation Forest model on the normal transactions
    model = IsolationForestModel()
    model.fit(training_rows[feature_columns], training_labels)    

    #score the validation rows using the trained model and add a new column for anomaly scores
    validation_rows["anomaly_score"] = model.predict_score(validation_rows[feature_columns])
    top_five = validation_rows.nlargest(5, "anomaly_score")

    #print the number of normal transactions used for training and the top five anomaly scores with their corresponding features and fraud labels
    normal_count = (training_labels == 0).sum()
    print(f"Trained on {normal_count:,} normal transactions.")
    print(top_five[["amount", "hour_of_day", "anomaly_score", "isFraud"]].to_string(index=False))

    #review 100 largest anomaly scores in validation set
    reviewed = validation_rows.nlargest(100, "anomaly_score")
    fraud_caught = (reviewed["isFraud"] == 1).sum() 

    #now find total fraud from the whole validation_rows table instead of top 100
    total_fraud = (validation_rows["isFraud"] == 1).sum()

    precision = fraud_caught / len(reviewed) # fraction of caught
    recall = fraud_caught / total_fraud #how much is caught over total fraud in validation set

    print(f"\nFraud caught: {fraud_caught} of {total_fraud}")
    print(f"Precision among the top 100: {precision:.1%}")
    print(f"Recall on validation: {recall:.1%}")


    # Building a DataFrame with transaction IDs (6 digit) with validation anomaly scores.
    predictions = pd.DataFrame({
        "txn_id": [f"seed-{row_number:06d}" for row_number in validation_rows.index],
        "score": validation_rows["anomaly_score"].to_numpy(),
    })

    # Ensure the output directory exists and save the predictions CSV without row indices.
    output_path = Path("data/processed/iforest_seed_val.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Saved {len(predictions):,} predictions to {output_path}")

    model_path = Path("data/processed/iforest_seed.joblib") #trained forest is written to disk as a joblib file
    model.save(model_path)  

    reloaded_model = IsolationForestModel.load(model_path) #reload back from disk and take first 5 validation rows
    example_features = validation_rows[feature_columns].head(5)

    #runs prediction score on both reloaded and origional model and checks that they are the same
    np.testing.assert_allclose(reloaded_model.predict_score(example_features),model.predict_score(example_features),) 
    print(f"Reloaded {model_path} and verified the same scores on five rows.")



# tells python to run main when the file is executed directly, but not when imported as a module in another script
if __name__ == "__main__":
    main()