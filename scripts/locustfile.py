import json
import os

from locust import HttpUser, task


class DreamUser(HttpUser):
    @task
    def predict(self):
        data = {"dream": "A purple cloud with Lightning", "high_quality": False}
        URL = f"{self.host}/api/predict"
        print(self.host)
        data = json.dumps(data)
        headers = {
            "accept": "application/json",
            "x-api-key": os.environ.get("RATE_LIMIT_KEY", ""),
            # Already added when you pass json= but not when you pass data=
            "Content-Type": "application/json",
        }
        response = self.client.post(URL, data=data, headers=headers, timeout=120)
        response.raise_for_status()
