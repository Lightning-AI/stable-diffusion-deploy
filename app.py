import os
import time
import uuid
from typing import List, Optional

import lightning as L
import requests
from lightning.app.frontend import StaticWebFrontend
from lightning.app.storage import Drive
from lightning_api_access import APIAccessFrontend
from lightning_app.utilities.frontend import AppInfo

from muse import (
    LoadBalancer,
    Locust,
    MuseSlackCommandBot,
    SafetyCheckerEmbedding,
    StableDiffusionServe,
)
from muse.CONST import ENABLE_TRACKERS
from muse.utility.trackers import trackers


class ReactUI(L.LightningFlow):
    def configure_layout(self):
        return StaticWebFrontend(os.path.join(os.path.dirname(__file__), "muse", "ui", "build"))


class APIUsageFlow(L.LightningFlow):
    def __init__(self, api_url: str = ""):
        super().__init__()
        self.api_url = api_url

    def configure_layout(self):
        return APIAccessFrontend(
            apis=[
                {
                    "name": "Generate Image",
                    "url": f"{self.api_url}/api/predict",
                    "method": "POST",
                    "request": {"dream": "cats in hats", "high_quality": "true"},
                    "response": {"image": "data:image/png;base64,<image-actual-content>"},
                }
            ]
        )


class MuseFlow(L.LightningFlow):
    """The MuseFlow is a LightningFlow component that handles all the servers and uses load balancer to spawn up and
    shutdown based on current requests in the queue.

    Args:
        initial_num_workers: Number of works to start when app initializes.
        autoscale_interval: Number of seconds to wait before checking whether to upscale or downscale the works.
        max_batch_size: Number of requests to process at once.
        batch_timeout_secs: Number of seconds to wait before sending the requests to process.
        gpu_type: GPU type to use for the works.
        max_workers: Max numbers of works to spawn to handle the incoming requests.
        autoscale_down_limit: Lower limit to determine when to stop works.
        autoscale_up_limit: Upper limit to determine when to spawn up a new work.
    """

    def __init__(
        self,
        initial_num_workers: int = 5,
        autoscale_interval: int = 1 * 30,
        max_batch_size: int = 4,
        batch_timeout_secs: int = 10,
        gpu_type: str = "gpu-fast",
        max_workers: int = 20,
        autoscale_down_limit: Optional[int] = None,
        autoscale_up_limit: Optional[int] = None,
        load_testing: Optional[bool] = False,
    ):
        super().__init__()
        self.hide_footer_shadow = True
        self.load_balancer_started = False
        self._initial_num_workers = initial_num_workers
        self._num_workers = 0
        self._work_registry = {}
        self.autoscale_interval = autoscale_interval
        self.max_workers = max_workers
        self.autoscale_down_limit = autoscale_down_limit or initial_num_workers
        self.autoscale_up_limit = autoscale_up_limit or initial_num_workers * max_batch_size
        self.load_testing = load_testing or os.getenv("MUSE_LOAD_TESTING", False)
        self.fake_trigger = 0
        self.gpu_type = gpu_type
        self.load_balancer_time = 60
        self._last_autoscale = time.time()

        # Create Drive to store Safety Checker embeddings
        self.safety_embeddings_drive = Drive("lit://embeddings")

        # Safety Checker Embedding Work to create and store embeddings in the Drive
        self.safety_checker_embedding_work = SafetyCheckerEmbedding(drive=self.safety_embeddings_drive)

        self.load_balancer = LoadBalancer(
            max_batch_size=max_batch_size, batch_timeout_secs=batch_timeout_secs, cache_calls=True, parallel=True
        )
        for i in range(initial_num_workers):
            work = StableDiffusionServe(
                safety_embeddings_drive=self.safety_embeddings_drive,
                safety_embeddings_filename=self.safety_checker_embedding_work.safety_embeddings_filename,
                cloud_compute=L.CloudCompute(gpu_type, disk_size=30),
                cache_calls=True,
                parallel=True,
            )
            self.add_work(work)

        self.slack_bot = MuseSlackCommandBot(command="/muse")
        if self.load_testing:
            self.locust = Locust(locustfile="./scripts/locustfile.py")
        self.printed_url = False
        self.slack_bot_url = ""
        self.dream_url = ""
        self.ui = ReactUI()
        self.api_component = APIUsageFlow()

        self.safety_embeddings_ready = False

    @property
    def model_servers(self) -> List[StableDiffusionServe]:
        works = []
        for i in range(self._num_workers):
            work: StableDiffusionServe = self.get_work(i)
            works.append(work)
        return works

    def add_work(self, work) -> str:
        work_attribute = uuid.uuid4().hex
        work_attribute = f"model_serve_{self._num_workers}_{str(work_attribute)}"
        setattr(self, work_attribute, work)
        self._work_registry[self._num_workers] = work_attribute
        self._num_workers += 1
        return work_attribute

    def remove_work(self, index: int) -> str:
        work_attribute = self._work_registry[index]
        del self._work_registry[index]
        work = getattr(self, work_attribute)
        work.stop()
        self._num_workers -= 1
        return work_attribute

    def get_work(self, index: int):
        work_attribute = self._work_registry[index]
        work = getattr(self, work_attribute)
        return work

    def run(self):  # noqa: C901
        if os.environ.get("TESTING_LAI"):
            print("⚡ Lightning Dream App! ⚡")

        # provision these works early
        if not self.load_balancer.is_running:
            self.load_balancer.run([])
        if not self.slack_bot.is_running:
            self.slack_bot.run("")

        if not self.safety_embeddings_ready:
            self.safety_checker_embedding_work.run()

        if not self.safety_embeddings_ready and self.safety_checker_embedding_work.has_succeeded:
            self.safety_embeddings_ready = True
            self.safety_checker_embedding_work.stop()

        for model_serve in self.model_servers:
            model_serve.run()

        if any(model_serve.url for model_serve in self.model_servers) and not self.load_balancer_started:
            # run the load balancer when one of the model servers is ready
            self.load_balancer.run([serve.url for serve in self.model_servers if serve.url])
            self.load_balancer_started = True

        if self.load_balancer.url:  # hack for getting the work url
            self.api_component.api_url = self.load_balancer.url
            self.dream_url = self.load_balancer.url
            self.load_balancer_time = self.set_response_time()
            if self.slack_bot is not None:
                self.slack_bot.run(self.load_balancer.url)
                self.slack_bot_url = self.slack_bot.url
                if self.slack_bot.url and not self.printed_url:
                    print("Slack Bot Work ready with URL=", self.slack_bot.url)
                    print("model serve url=", self.load_balancer.url)
                    print("API component url=", self.api_component.state_vars["vars"]["_layout"]["target"])
                    self.printed_url = True

        if self.load_testing and self.load_balancer.url:
            self.locust.run(self.load_balancer.url)

        if self.load_balancer.url:
            self.fake_trigger += 1
            self.autoscale()

    def configure_layout(self):
        ui = [{"name": "Muse App" if self.load_testing else None, "content": self.ui}]
        if self.load_testing:
            ui.append({"name": "Locust", "content": self.locust.url})

        return ui

    def autoscale(self):
        """Upscale and down scale model inference works based on the number of requests."""
        if time.time() - self._last_autoscale < self.autoscale_interval:
            return

        self.load_balancer.update_servers(self.model_servers)

        num_requests = int(requests.get(f"{self.load_balancer.url}/num-requests").json())
        num_workers = len(self.model_servers)

        # upscale
        if num_requests > self.autoscale_up_limit and num_workers < self.max_workers:
            idx = self._num_workers
            print(f"Upscale to {self._num_workers + 1}")
            work = StableDiffusionServe(
                safety_embeddings_drive=self.safety_embeddings_drive,
                safety_embeddings_filename=self.safety_checker_embedding_work.safety_embeddings_filename,
                cloud_compute=L.CloudCompute(self.gpu_type, disk_size=30),
                cache_calls=True,
                parallel=True,
            )
            new_work_id = self.add_work(work)
            print("new work id:", new_work_id)

        # downscale
        elif num_requests < self.autoscale_down_limit and num_workers > self._initial_num_workers:
            idx = self._num_workers - 1
            print(f"Downscale to {idx}")
            print("prev num servers:", len(self.model_servers))
            removed_id = self.remove_work(idx)
            print("removed:", removed_id)
            print("new num servers:", len(self.model_servers))
            self.load_balancer.update_servers(self.model_servers)
        self._last_autoscale = time.time()

    def set_response_time(self):
        if len(self.load_balancer._response_time_queue) == 0:
            return 60  # default to 60 seconds
        return sum(self.load_balancer._response_time_queue) // len(self.load_balancer._response_time_queue)


if __name__ == "__main__":
    app = L.LightningApp(
        MuseFlow(),
        info=AppInfo(
            title="Use AI to inspire your art.",
            favicon="https://storage.googleapis.com/grid-static/muse/favicon.ico",
            description="Bring your words to life in seconds - powered by Lightning AI and Stable Diffusion.",
            image="https://storage.googleapis.com/grid-static/header.png",
            meta_tags=[
                '<meta name="theme-color" content="#792EE5" />',
                '<meta name="image" content="https://storage.googleapis.com/grid-static/header.png">'
                '<meta itemprop="name" content="Use AI to inspire your art.">'
                '<meta itemprop="description" content="Bring your words to life in seconds - powered by Lightning AI and Stable Diffusion.">'  # noqa
                '<meta itemprop="image" content="https://storage.googleapis.com/grid-static/header.png">'
                # <!-- Twitter -->
                '<meta name="twitter:card" content="summary">'
                '<meta name="twitter:title" content="Use AI to inspire your art.">'
                '<meta name="twitter:description" content="Bring your words to life in seconds - powered by Lightning AI and Stable Diffusion.">'  # noqa
                '<meta name="twitter:site" content="https://lightning.ai/muse">'
                '<meta name="twitter:domain" content="https://lightning.ai/muse">'
                '<meta name="twitter:creator" content="@LightningAI">'
                '<meta name="twitter:image:src" content="https://storage.googleapis.com/grid-static/header.png">'
                # <!-- Open Graph general (Facebook, Pinterest & Google+) -->
                '<meta name="og:title" content="Use AI to inspire your art.">'
                '<meta name="og:description" content="Bring your words to life in seconds - powered by Lightning AI and Stable Diffusion.">'  # noqa
                '<meta name="og:url" content="https://lightning.ai/muse">'
                '<meta property="og:image" content="https://storage.googleapis.com/grid-static/header.png" />',
                '<meta property="og:image:type" content="image/png" />',
                '<meta property="og:image:height" content="1114" />'
                '<meta property="og:image:width" content="1112" />',
                *(trackers if ENABLE_TRACKERS else []),
            ],
        ),
        root_path=os.getenv("MUSE_ROOT_PATH", ""),
    )
