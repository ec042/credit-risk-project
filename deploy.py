import joblib
import os

def main():
    # Example model (replace with your trained model)
    model = ...  # Load or define your model here

    # Define deployment directory
    deployment_dir = 'deployment_dir'
    os.makedirs(deployment_dir, exist_ok=True)

    # Save model to deployment_dir as credit.pkl
    joblib.dump(model, os.path.join(deployment_dir, 'credit.pkl'))

if __name__ == "__main__":
    main()
