import os

from arize.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor


class ArizeLangChainTracer:
    def __init__(self, project_name: str = "langGraph-demo") -> None:
        self.project_name = project_name
        self.tracer_provider = None

    def setup_instrumentation(self) -> None:
        print(os.environ["OPENAI_API_KEY"][:10])
        print(os.environ["ARIZE_API_KEY"][:10])
        print(os.environ["ARIZE_SPACE_ID"][:10])

        self.tracer_provider = register(
            space_id=os.environ["ARIZE_SPACE_ID"],
            api_key=os.environ["ARIZE_API_KEY"],
            project_name=self.project_name,
        )

        LangChainInstrumentor(tracer_provider=self.tracer_provider).instrument()

        print("Instrumentation active. Sending traces to Arize Cloud Project")


if __name__ == "__main__":
    ArizeLangChainTracer().setup_instrumentation()