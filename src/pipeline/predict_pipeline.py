import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object



class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            data_transformed = preprocessor.transform(features)
            pred = model.predict(data_transformed)
            return pred
        except Exception as e:
            raise CustomException(e, sys)
    
    
class CustomData:
    def __init__(self,
                 country:str,
                 Gender:str,
                 Age:str,
                 Marital_status:str,
                 Stratum_Urban:str,
                 Natur_of_work:str,
                 Level_of_Wealth:str,
                 Fathers_level_education:str,
                 Parent_affiliated_with_SS:str,
                 Participation_in_elections:str,
                 Freedom_to_speach_out:str,
                 Resort_to_nepotism:str,
                 Trust_in_Parliment:str,
                 Trust_in_employers:str,
                 Trust_in_associations:str,
                 Trust_in_political_parties:str,
                 Political_system:str):
        self.country = country   
        self.Gender = Gender
        self.Age = Age      
        self.Marital_status = Marital_status
        self.Stratum_Urban = Stratum_Urban      
        self.Natur_of_work = Natur_of_work
        self.Level_of_Wealth = Level_of_Wealth
        self.Fathers_level_education = Fathers_level_education
        self.Parent_affiliated_with_SS = Parent_affiliated_with_SS
        self.Participation_in_elections = Participation_in_elections    
        self.Freedom_to_speach_out = Freedom_to_speach_out
        self.Resort_to_nepotism = Resort_to_nepotism
        self.Trust_in_Parliment = Trust_in_Parliment
        self.Trust_in_employers = Trust_in_employers
        self.Trust_in_associations = Trust_in_associations
        self.Trust_in_political_parties = Trust_in_political_parties
        self.Political_system = Political_system
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'country': [self.country],
                'Gender': [self.Gender],    
                'Age': [self.Age],  
                'Stratum Urban': [self.Stratum_Urban],
                'Marital status': [self.Marital_status],
                'Natur of work': [self.Natur_of_work],
                'Level of Wealth': [self.Level_of_Wealth],
                'Fathers level education': [self.Fathers_level_education],
                'Parent_affiliated_with Social Security': [self.Parent_affiliated_with_SS],
                'Participation in elections': [self.Participation_in_elections],
                'Freedom to speach out about government': [self.Freedom_to_speach_out],
                'Resort to nepotism': [self.Resort_to_nepotism],
                'Trust in Parliment': [self.Trust_in_Parliment],
                'Trust in employers': [self.Trust_in_employers],
                'Trust in associations': [self.Trust_in_associations],
                'Trust in political_parties': [self.Trust_in_political_parties],
                'Political system': [self.Political_system]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
        
                
                                   
                 
