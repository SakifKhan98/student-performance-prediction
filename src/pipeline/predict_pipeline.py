import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, data):
        try:
            logging.info("Entered the predict method or component")
            logging.info("Loading the trained model")
            # model_path = os.path.join("artifacts", "model.pkl")
            # preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            ARTIFACTS_DIR = "./src/components/artifacts/"
            MODEL_NAME = "model.pkl"
            PREPROCESSOR_NAME = "preprocessor.pkl"
            # df = pd.read_csv(f"{ARTIFACTS_DIR}{MODEL_NAME}")
            # model_path = "artifacts/model.pkl"
            # preprocessor_path = "artifacts/preprocessor.pkl"
            # model = load_object(os.path.join("artifacts", "model.pkl"))
            model = load_object(file_path=f"{ARTIFACTS_DIR}{MODEL_NAME}")
            preprocessor = load_object(file_path=f"{ARTIFACTS_DIR}{PREPROCESSOR_NAME}")
            logging.info("Model loaded successfully")
            logging.info("Transforming the data")
            data_scaled = preprocessor.transform(data)
            logging.info("Predicting the data")
            prediction = model.predict(data_scaled)
            logging.info("Prediction completed")
            return prediction
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
