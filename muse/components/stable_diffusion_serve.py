import base64
import os.path
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List

import lightning as L
import numpy as np
import torch
from PIL import Image
from torch import autocast

from muse.CONST import IMAGE_SIZE, KEEP_ALIVE_TIMEOUT, REQUEST_TIMEOUT
from muse.utility.utils import Data, DataBatch, TimeoutException


@dataclass
class FastAPIBuildConfig(L.BuildConfig):
    requirements = ["fastapi==0.78.0", "uvicorn==0.17.6"]


class StableDiffusionServe(L.LightningWork):
    """The StableDiffusionServer handles the prediction.

    It initializes a model and expose an API to handle incoming requests and generate predictions.
    """

    def __init__(self, **kwargs):
        super().__init__(cloud_build_config=FastAPIBuildConfig(), **kwargs)
        self._model = None

    @staticmethod
    def download_weights(url: str, target_folder: str):
        dest = target_folder / f"{os.path.basename(url)}"
        if not os.path.exists(dest):
            urllib.request.urlretrieve(url, dest)
            file = tarfile.open(dest)

            # extracting file
            file.extractall(target_folder)

    def build_model(self):
        """The `build_model(...)` method returns a model and the returned model is set to `self._model` state."""

        import os

        import torch

        from dream.models import StableDiffusionPipelineTraced

        print("loading model...")
        if torch.cuda.is_available():
            weights_folder = Path("resources/stable-diffusion-v1-4")
            os.makedirs(weights_folder, exist_ok=True)

            print("Downloading weights...")
            self.download_weights(
                "https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/diffusers_traced.tar.gz", weights_folder
            )

            pipe = StableDiffusionPipelineTraced(weights_folder / "diffusers_traced")
            print("model loaded")
        else:
            pipe = None
            print("model set to None")
        return pipe

    @torch.inference_mode()
    def predict(self, dreams: List[Data], entry_time: int):
        if time.time() - entry_time > REQUEST_TIMEOUT:
            raise TimeoutException()

        height = width = IMAGE_SIZE
        num_inference_steps = 50 if dreams[0].high_quality else 25

        prompts = [dream.dream for dream in dreams]
        if torch.cuda.is_available():
            with autocast("cuda"):
                torch.cuda.empty_cache()
                pil_results = self._model(
                    prompts,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                )
        else:
            pil_results = [Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype="uint8"))] * len(prompts)

        results = []
        for image in pil_results:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            results.append(f"data:image/png;base64,{img_str}")

        return results

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
            return True

        @app.post("/api/predict")
        def predict_api(data: DataBatch):
            """Dream a muse. Defines the REST API which takes the text prompt, number of images and image size in the
            request body.

            This API returns an image generated by the model in base64 format.
            """
            try:
                entry_time = time.time()
                print(f"batch size: {len(data.batch)}")
                result = app.POOL.submit(
                    self.predict,
                    data.batch,
                    entry_time=entry_time,
                ).result(timeout=REQUEST_TIMEOUT)
                return result
            except (TimeoutError, TimeoutException):
                # hack: once there is a timeout then all requests after that is getting timeout
                # old_pool = app.POOL
                # app.POOL = ThreadPoolExecutor(max_workers=1)
                # old_pool.shutdown(wait=False)
                # signal.signal(signal.SIGINT, lambda sig, frame: exit_threads(old_pool))
                raise TimeoutException()

        uvicorn.run(
            app, host=self.host, port=self.port, timeout_keep_alive=KEEP_ALIVE_TIMEOUT, access_log=False, loop="uvloop"
        )
