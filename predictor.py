import argparse
from prediction import DiseasePredictor

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Disease Prediction')

    # Add arguments for model name and image path
    parser.add_argument('--model', type=str, help='Path to the model file (e.g., fine_tuned_modelApple.h5)', required=True)
    parser.add_argument('--image', type=str, help='Path to the image file (e.g., some_image.jpg)', required=True)

    # Parse the command line arguments
    args = parser.parse_args()

    # Create an instance of the DiseasePredictor class
    predictor = DiseasePredictor(args.model, args.image)

    # Display the prediction
    predictor.display_prediction()

if __name__ == "__main__":
    main()

