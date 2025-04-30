from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import AgentRunner, FunctionCallingAgentWorker

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo", temperature=1)


def generate_unit_test(code: str) -> str:
    return f"""write unit tests in jest for React Javascript code:
    - Test must cover general logic
    - Use Jest liberary
    Return code.
    """


test_tool = FunctionTool.from_defaults(
    fn=generate_unit_test,
    name="generate_test",
    description="Generate test for this code",
)

worker = FunctionCallingAgentWorker.from_tools(
    tools=[test_tool],
    llm=llm,
    system_prompt="You are frontend developer, You write unit tests in jest for React Javascript code. Write test for this react component ",
)

agent = AgentRunner(worker)


if __name__ == "__main__":
    sample_code = """

      const InfoIconWithTooltip = ({ message }) => (
    <Tooltip arrow title={message}>
        <InfoIcon color='primary' />
    </Tooltip>
);
    """

    result = agent.chat(sample_code)
    print(result.response)
