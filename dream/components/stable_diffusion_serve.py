import base64
import os.path
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from io import BytesIO

import lightning as L
import numpy as np
import torch
from PIL import Image
from torch import autocast

from dream.components.utils import Data, TimeoutException
from dream.CONST import REQUEST_TIMEOUT


@dataclass
class FastAPIBuildConfig(L.BuildConfig):
    requirements = ["fastapi==0.78.0", "uvicorn==0.17.6"]


class StableDiffusionServe(L.LightningWork):
    """Deploys the Stable Diffusion model with FastAPI.

    tolerable_failures: total number of failures after which the worker status becomes unhealthy.
    """

    def __init__(self, tolerable_failures=2, **kwargs):
        super().__init__(cloud_build_config=FastAPIBuildConfig(), **kwargs)
        self.num_failures = 0
        self._model = None
        self.tolerable_failures = tolerable_failures

    @staticmethod
    def download_weights(url: str, target_folder: str):
        dest = target_folder + f"/{os.path.basename(url)}"
        urllib.request.urlretrieve(url, dest)
        file = tarfile.open(dest)

        # extracting file
        file.extractall(target_folder)

    def build_model(self):
        """The `build_model(...)` method returns a model and the returned model is set to `self._model` state."""

        import os

        import torch
        from diffusers import StableDiffusionPipeline

        print("loading model...")
        if torch.cuda.is_available():
            weights_folder = "resources/stable-diffusion-v1-4"
            os.makedirs(weights_folder, exist_ok=True)

            print("Downloading weights...")
            self.download_weights(
                "https://lightning-dream-app-assets.s3.amazonaws.com/diffusers.tar.gz", weights_folder
            )

            repo_folder = f"{weights_folder}/Users/pritam/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-4/snapshots/a304b1ab1b59dd6c3ba9c40705c29c6de4144096"
            pipe = StableDiffusionPipeline.from_pretrained(
                repo_folder,
                revision="fp16",
                torch_dtype=torch.float16,
            )
            pipe = pipe.to("cuda")
            print("model loaded")
        else:
            pipe = None
            print("model set to None")
        return pipe

    def predict(self, dream: str, num_inference_steps: int, entry_time: int):
        if time.time() - entry_time > REQUEST_TIMEOUT:
            raise TimeoutException()

        height, width = 512, 512
        with autocast("cuda"):
            if torch.cuda.is_available():
                preds = self._model(
                    dream,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                )
                generated_image = preds.images[0]
                if preds.has_nsfw_concept:
                    generated_image = Image.open("./assets/nsfw-warning.png")
            else:
                generated_image = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype="uint8"))

        results = []
        buffered = BytesIO()
        generated_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        results.append(f"data:image/png;base64,{img_str}")
        return results

    @property
    def health_status(self):
        return self.num_failures < self.tolerable_failures

    def run(self):
        import subprocess

        import uvicorn
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        subprocess.run("nvidia-smi", shell=True)

        if self._model is None:
            self._model = self.build_model()

        self._fastapi_app = app = FastAPI()
        app.POOL: ThreadPoolExecutor = None

        @app.on_event("startup")
        def startup_event():
            app.POOL = ThreadPoolExecutor(max_workers=1)

        @app.on_event("shutdown")
        def shutdown_event():
            app.POOL.shutdown(wait=False)

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/api/health")
        def health():
            return self.health_status

        @app.post("/api/predict")
        def predict_api(data: Data):
            """Dream a dream. Defines the REST API which takes the text prompt, number of images and image size in the
            request body.

            This API returns an image generated by the model in base64 format.
            """
            try:
                entry_time = time.time()
                print(f"request: {data}")
                num_inference_steps = 50 if data.high_quality else 25
                result = app.POOL.submit(
                    self.predict,
                    data.dream,
                    num_inference_steps,
                    entry_time=entry_time,
                ).result(timeout=REQUEST_TIMEOUT)
                return result
            except (TimeoutError, TimeoutException):
                # hack: once there is a timeout then all requests after that is getting timedout
                old_pool = app.POOL
                app.POOL = ThreadPoolExecutor(max_workers=1)
                old_pool.shutdown(wait=False, cancel_futures=True)
                self.num_failures += 1
                raise TimeoutException()

        uvicorn.run(app, host=self.host, port=self.port)
