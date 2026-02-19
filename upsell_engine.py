from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from knowledge_base_connector import KnowledgeBaseConnector
from marketing_automation import MarketingAutomationAPI

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='upsell_engine.log'
)
logger = logging.getLogger(__name__)

class SubscriptionDataCollector:
    def __init__(self):
        self.crm_connector = CRMConnector()
        self.usage_analyzer = UsageAnalyzer()

    def collect_subscription_data(self) -> pd.DataFrame:
        # Collect data from various sources
        try:
            customer_data = self.crm_connector.get_customer_details()
            usage_data = self.usage_analyzer.get_usage_metrics()
            
            # Merge datasets on customer ID
            df = pd.merge(customer_data, usage_data, on='customer_id')
            return df
        except Exception as e:
            logger.error(f"Error collecting subscription data: {str(e)}")
            raise

class CustomerRiskPredictor:
    def __init__(self):
        self.model = LogisticRegression()
        
    def train_model(self, data: pd.DataFrame) -> None:
        # Prepare features and target
        X = data[['subscription_tenure', 'login_frequency', 'payment_failures']]
        y = data['churn_flag']
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        try:
            self.model.fit(X_train, y_train)
            accuracy = accuracy_score(self.model.predict(X_test), y_test)
            logger.info(f"Model trained with accuracy: {accuracy}")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def predict_risk(self, customer_data: pd.DataFrame) -> Dict[str, float]:
        try:
            # Preprocess data
            processed_data = self._preprocess_input(customer_data)
            
            # Predict risk scores
            risk_scores = dict(zip(
                customer_data['customer_id'],
                list(self.model.predict_proba(processed_data)[:, 1])
            ))
            
            return risk_scores
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def _preprocess_input(self, data: pd.DataFrame) -> np.ndarray:
        # Normalize features
        processed = (data - data.mean()) / (data.std())
        return processed.values

class OfferPersonalizer:
    def __init__(self):
        self.knowledge_base = KnowledgeBaseConnector()

    def get_personalized_offers(self, risk_scores: Dict[str, float]) -> Dict[str, str]:
        try:
            # Segment customers based on risk
            high_risk = [cid for cid, score in risk_scores.items() if score > 0.7]
            medium_risk = [cid for cid, score in risk_scores.items() if 0.3 < score <= 0.7]
            
            offers = {}
            for cid in high_risk:
                # Retrieve customer details from knowledge base
                details = self.knowledge_base.get_customer_details(cid)
                offer = f"Upgrade to Premium - {details['preferred_tier']}"
                offers[cid] = offer
                
            for cid in medium_risk:
                details = self.knowledge_base.get_customer_details(cid)
                offer = f"Try our Advanced features - {details['usage_metrics']}"
                offers[cid] = offer
            
            return offers
        except Exception as e:
            logger.error(f"Personalization failed: {str(e)}")
            raise

class UpsellEngine:
    def __init__(self):
        self.risk_predictor = CustomerRiskPredictor()
        self.offer_personalizer = OfferPersonalizer()
        self.marketing_automation = MarketingAutomationAPI()

    def run_engine(self) -> None:
        try:
            # Collect data
            collector = SubscriptionDataCollector()
            df = collector.collect_subscription_data()
            
            # Train model if not already trained
            self.risk_predictor.train_model(df)
            
            # Predict risks
            risk_scores = self.risk_predictor.predict_risk(df)
            
            # Generate offers
            offers = self.offer_personalizer.get_personalized_offers(risk_scores)
            
            # Execute offers through marketing automation
            self.marketing_automation.send_offers(offers)
            
            logger.info("Upsell engine executed successfully")
        except Exception as e:
            logger.error(f"Engine failed: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    upsell_engine = UpsellEngine()
    upsell_engine.run_engine()