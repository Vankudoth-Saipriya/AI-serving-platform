import random
from locust import HttpUser, task, between, events

REALISTIC_QUERIES = [
    "what is my credit card balance",
    "book a table for 4 at an Italian restaurant",
    "what is the weather like in New York today",
    "who won the 1998 World Cup in France",
    "blabla random nonsense quantum banana string",
    "how do I change my oil in my car",
    "please cancel my flight to London tomorrow",
    "what are the nutrition facts for an avocado",
    "how much interest will I earn on my savings account",
    "what gas type should I use for my vehicle",
    "where is the nearest ATM machine located",
    "can I redeem my reward points for cash",
    "why is my account blocked right now",
    "set a timer for 15 minutes for cooking",
    "translate this sentence into Spanish please",
    "what are the top rated movies playing near me"
]

class ModelServingUser(HttpUser):
    wait_time = between(0.1, 0.5)  # 100ms to 500ms delay between tasks

    @task(4)
    def predict_single(self):
        query = random.choice(REALISTIC_QUERIES)
        self.client.post(
            "/predict",
            json={"text": query},
            name="/predict (single)"
        )

    @task(1)
    def predict_batch(self):
        batch_sample = random.sample(REALISTIC_QUERIES, k=5)
        self.client.post(
            "/predict/batch",
            json={"inputs": batch_sample},
            name="/predict/batch (batch of 5)"
        )

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")
