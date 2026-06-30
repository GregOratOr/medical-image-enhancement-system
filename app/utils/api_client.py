# app/utils/api_clients.py

import os
import requests

class ApiClient:
    def __init__(self, base_url: str | None = None):
        """ Initialize the client with the address of your running FastAPI server."""

        if not base_url:
            self.base_url = os.getenv("API_URL", "http://127.0.0.1:8000")
        else:
            self.base_url = base_url.rstrip("/")

    def check_health(self) -> bool:
        """ Ping the backend to verify connection.

            Returns: True if server is online (200 OK), False otherwise.
        """
        try:
            response = requests.get(f"{self.base_url}/", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_config(self) -> dict:
        """ Fetch currently active config from server."""
        try:
            r = requests.get(f"{self.base_url}/", timeout=2)
            if r.status_code == 200:
                return r.json().get("active_config", {})
            return {}
        except:
            return {}

    def update_config(self, model_name: str, enable_cuda: bool, wrapped_model: bool) -> bool:
        """ Send request to swap the backend engine."""
        payload = {
            "model_name": model_name,
            "enable_cuda": enable_cuda,
            "wrapped_model": wrapped_model
        }
        try:
            r = requests.post(f"{self.base_url}/set_config", json=payload, timeout=30)
            return r.status_code == 200
        except Exception as e:
            print(f"Config Update Failed: {e}")
            return False

    def predict(self, image_bytes: bytes, filename: str) -> bytes | None:
        """ Send a raw image to the processing engine.
        
        Args:
            image_bytes: The raw bytes of the file.
            filename: The original name (e.g., 'scan_01.png').
            
        Returns:
            bytes: The denoised image bytes if successful.
            None: If the request failed.
        """

        try:
            # Prepare the multipart/form-data payload
            # We explicitly set the MIME type to generic 'image/png' to satisfy FastAPI
            files = {"file": (filename, image_bytes, "image/png")}

            # Send POST request
            response = requests.post(
                f"{self.base_url}/predict", 
                files=files,
                timeout=120
                )
            
            if response.status_code == 200:
                return response.content
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                return None
                
        except requests.RequestException as e:
            print(f"❌ Connection Error: {e}")
            return None
    
    def unload_model(self):
        """Tells the server to drop the model from VRAM."""
        
        import requests
        try:
            response = requests.post(f"{self.base_url}/unload")
            return response.status_code == 200
        except:
            return False