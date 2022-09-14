import os

import lightning as L
from lightning.app.frontend import StaticWebFrontend

from dream import DreamSlackCommandBot, StableDiffusionServe


class ReactUI(L.LightningFlow):
    def configure_layout(self):
        return StaticWebFrontend(os.path.join(os.path.dirname(__file__), "dream", "ui", "build"))


class RootWorkFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.model_serve = StableDiffusionServe(initial_num_workers=2, worker_compute_type="gpu")

        self.slack_bot = DreamSlackCommandBot(command="/dream")

        self.printed_url = False
        self.slack_bot_url = ""

        self.dream_url = ""
        self.ui = ReactUI()

    def run(self):
        if os.environ.get("TESTING_LAI"):
            print("⚡ Lightning Dream App! ⚡")
        # if self.model_serve.url:  # hack for getting the work url
        #     self.dream_url = self.model_serve.url

        #     if self.slack_bot is not None:
        #         self.slack_bot.run(self.model_serve.url)
        #         if self.slack_bot.url and not self.printed_url:
        #             print("Slack Bot Work ready with URL=", self.slack_bot.url)
        #             self.printed_url = True

        self.model_serve.run()

    def configure_api(self):
        return self.model_serve.configure_api()

    def configure_layout(self):
        return [
            {
                "name": None,
                "content": self.ui,
            },
        ]


if __name__ == "__main__":
    app = L.LightningApp(RootWorkFlow())
