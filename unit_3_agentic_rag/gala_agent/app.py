import random
from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool

from tools import WeatherInfoTool, HubStatsTool
from retriever import load_guest_dataset


# Initialize the Hugging Face model
model = InferenceClientModel()

# Initialize tools
search_tool = DuckDuckGoSearchTool()
weather_info_tool = WeatherInfoTool()
hub_stats_tool = HubStatsTool()
guest_info_tool = load_guest_dataset()

# Create Alfred, the gala agent
alfred = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool],
    model=model,
    add_base_tools=True,
    planning_interval=3,
)


if __name__ == "__main__":
    examples = [
        "Tell me about our guest named Lady Ada Lovelace.",
        "What's the weather like in Paris tonight? Will it be suitable for our fireworks display?",
        "One of our guests is from Qwen. Can you tell me about their most popular model?",
        "I need to speak with Dr. Nikola Tesla about recent advances in wireless energy. Can you help me prepare?",
    ]

    query = random.choice(examples)
    print(f"User query: {query}\n")

    response = alfred.run(query)

    print("\nAlfred's Response:")
    print(response)